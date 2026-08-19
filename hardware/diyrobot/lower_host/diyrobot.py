#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
DIYRobot bimanual mobile robot driver for LeRobot.

Hardware layout
---------------
Leader arms  (Feetech STS3215, USB-serial bus)
  Left  : IDs 1-7  → shoulder_pan, shoulder_lift, elbow_flex,
                      wrist_flex, wrist_roll, wrist_pitch, gripper
  Right : IDs 8-14 (same joint names, different IDs to avoid collision)

Follower arms (RobStride CAN, USB-CAN #1)
  Left  : CAN IDs 1-7
  Right : CAN IDs 11-17

Chassis / lift (Damiao CAN, USB-CAN #2)
  Wheel left   : CAN ID 0x21 (33)
  Wheel back   : CAN ID 0x22 (34)
  Wheel right  : CAN ID 0x23 (35)
  Lift         : CAN ID 0x24 (36)
"""

import logging
import math
import time
from functools import cached_property

import numpy as np

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.damiao import DamiaoMotorsBus
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode
from lerobot.motors.robstride import RobstrideMotorsBus
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_diyrobot import DIYRobotConfig
from .limit_reader import LiftLimitReader
from .robstride_at_bus import RobStrideAtMotorsBus

logger = logging.getLogger(__name__)

ARM_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "wrist_pitch",
    "gripper",
]

LEADER_LEFT_IDS = {name: i + 1 for i, name in enumerate(ARM_JOINT_NAMES)}
LEADER_RIGHT_IDS = {name: i + 8 for i, name in enumerate(ARM_JOINT_NAMES)}
FOLLOWER_LEFT_IDS = {name: i + 1 for i, name in enumerate(ARM_JOINT_NAMES)}
FOLLOWER_RIGHT_IDS = {name: i + 11 for i, name in enumerate(ARM_JOINT_NAMES)}

CHASSIS_IDS = {
    "wheel_left": 0x21,
    "wheel_back": 0x22,
    "wheel_right": 0x23,
    "lift": 0x24,
}
WHEEL_MOTORS = ["wheel_left", "wheel_back", "wheel_right"]
LIFT_MOTOR = "lift"


class DIYRobot(Robot):
    """Control the DIYRobot arms, mobile base, lift, and camera observations."""

    config_class = DIYRobotConfig
    name = "diyrobot"

    DEFAULT_KP = 20.0
    DEFAULT_KD = 0.5

    def __init__(self, config: DIYRobotConfig):
        super().__init__(config)
        self.config = config

        norm_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100

        leader_motors = {}
        for name, mid in LEADER_LEFT_IDS.items():
            nm = norm_body if name != "gripper" else MotorNormMode.RANGE_0_100
            leader_motors[f"left_{name}"] = Motor(mid, "sts3215", nm)
        for name, mid in LEADER_RIGHT_IDS.items():
            nm = norm_body if name != "gripper" else MotorNormMode.RANGE_0_100
            leader_motors[f"right_{name}"] = Motor(mid, "sts3215", nm)

        self.leader_bus = FeetechMotorsBus(
            port=config.leader_port,
            motors=leader_motors,
            calibration=self.calibration,
        )

        follower_motors = {}
        for name, mid in FOLLOWER_LEFT_IDS.items():
            follower_motors[f"left_{name}"] = Motor(mid, "O3", MotorNormMode.DEGREES)
        for name, mid in FOLLOWER_RIGHT_IDS.items():
            follower_motors[f"right_{name}"] = Motor(mid, "O3", MotorNormMode.DEGREES)

        if str(config.follower_port).startswith("/dev/"):
            self.follower_bus = RobStrideAtMotorsBus(
                port=config.follower_port,
                motors=follower_motors,
            )
        else:
            self.follower_bus = RobstrideMotorsBus(
                port=config.follower_port,
                motors=follower_motors,
            )

        chassis_motors = {
            "wheel_left": Motor(CHASSIS_IDS["wheel_left"], "dm4310", MotorNormMode.RANGE_M100_100, motor_type_str="dm4310", recv_id=0x31),
            "wheel_back": Motor(CHASSIS_IDS["wheel_back"], "dm4310", MotorNormMode.RANGE_M100_100, motor_type_str="dm4310", recv_id=0x32),
            "wheel_right": Motor(CHASSIS_IDS["wheel_right"], "dm4310", MotorNormMode.RANGE_M100_100, motor_type_str="dm4310", recv_id=0x33),
            "lift": Motor(CHASSIS_IDS["lift"], "dm4310", MotorNormMode.DEGREES, motor_type_str="dm4310", recv_id=0x34),
        }
        self.chassis_bus = DamiaoMotorsBus(
            port=config.chassis_port,
            motors=chassis_motors,
            can_interface="slcan" if str(config.chassis_port).startswith("/dev/") else "socketcan",
            use_can_fd=False,
            bitrate=1000000,
        )

        self.cameras = make_cameras_from_configs(config.cameras)

        self._follower_kp = {m: self.DEFAULT_KP for m in self.follower_bus.motors}
        self._follower_kd = {m: self.DEFAULT_KD for m in self.follower_bus.motors}

        self._lift_kp = 20.0
        self._lift_kd = 0.5
        self._last_lift_pos_deg = 0.0
        self._last_limit_state = {
            "top_limit": False,
            "bottom_limit": False,
            "last_update": 0.0,
            "stale": True,
            "last_line": "",
        }
        self.limit_reader = LiftLimitReader(
            port=config.lift_limit_port,
            baudrate=config.lift_limit_baudrate,
            stale_timeout_s=config.lift_limit_stale_timeout_s,
        )

    @property
    def _arm_motors_ft(self) -> dict[str, type]:
        return {f"{m}.pos": float for m in self.follower_bus.motors}

    @property
    def _chassis_ft(self) -> dict[str, type]:
        return {
            "x.vel": float,
            "y.vel": float,
            "theta.vel": float,
            "lift.pos": float,
            "lift.top_limit": bool,
            "lift.bottom_limit": bool,
            "lift.limit_stale": bool,
        }

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3)
            for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._arm_motors_ft, **self._chassis_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {**self._arm_motors_ft, "x.vel": float, "y.vel": float, "theta.vel": float, "lift.pos": float}

    @property
    def is_connected(self) -> bool:
        return (
            self.leader_bus.is_connected
            and self.follower_bus.is_connected
            and self.chassis_bus.is_connected
            and all(cam.is_connected for cam in self.cameras.values())
        )

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self.leader_bus.connect()
        self.follower_bus.connect(handshake=True)
        self.chassis_bus.connect(handshake=True)

        if not self.is_calibrated and calibrate:
            self.calibrate()

        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        self.limit_reader.start()
        if self.config.auto_home_lift_on_connect:
            self.home_lift()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return self.leader_bus.is_calibrated

    def calibrate(self) -> None:
        if self.calibration:
            user_input = input(
                f"Press ENTER to use existing calibration for '{self.id}', "
                "or type 'c' + ENTER to re-run calibration: "
            )
            if user_input.strip().lower() != "c":
                self.leader_bus.write_calibration(self.calibration)
                return

        logger.info("Running leader arm calibration…")
        self.leader_bus.disable_torque()
        for motor in self.leader_bus.motors:
            self.leader_bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

        input("Move both leader arms to the middle of their range of motion and press ENTER…")
        homing_offsets = self.leader_bus.set_half_turn_homings()

        full_turn_motors = [m for m in self.leader_bus.motors if "wrist_roll" in m]
        unknown_range_motors = [m for m in self.leader_bus.motors if m not in full_turn_motors]

        print("Move all joints (except wrist_roll) through their full range. Press ENTER to stop…")
        range_mins, range_maxes = self.leader_bus.record_ranges_of_motion(unknown_range_motors)
        for m in full_turn_motors:
            range_mins[m] = 0
            range_maxes[m] = 4095

        self.calibration = {}
        for motor, m_obj in self.leader_bus.motors.items():
            self.calibration[motor] = MotorCalibration(
                id=m_obj.id,
                drive_mode=0,
                homing_offset=homing_offsets[motor],
                range_min=range_mins[motor],
                range_max=range_maxes[motor],
            )

        self.leader_bus.write_calibration(self.calibration)
        self._save_calibration()
        print("Calibration saved to", self.calibration_fpath)

    def configure(self) -> None:
        with self.leader_bus.torque_disabled():
            self.leader_bus.configure_motors()
            for motor in self.leader_bus.motors:
                self.leader_bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
                self.leader_bus.write("P_Coefficient", motor, 16)
                self.leader_bus.write("I_Coefficient", motor, 0)
                self.leader_bus.write("D_Coefficient", motor, 32)
                if "gripper" in motor:
                    self.leader_bus.write("Max_Torque_Limit", motor, 500)
                    self.leader_bus.write("Protection_Current", motor, 250)
                    self.leader_bus.write("Overload_Torque", motor, 25)

        self.follower_bus.configure_motors()
        self.follower_bus.enable_torque()

        self.chassis_bus.configure_motors()
        self.chassis_bus.enable_torque()

    def setup_motors(self) -> None:
        for motor in reversed(self.leader_bus.motors):
            input(f"Connect controller to '{motor}' only and press ENTER.")
            self.leader_bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.leader_bus.motors[motor].id}")

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        obs: dict = {}

        arm_states = self.follower_bus.sync_read_all_states()
        for motor, state in arm_states.items():
            obs[f"{motor}.pos"] = state["position"]

        chassis_states = self.chassis_bus.sync_read_all_states(WHEEL_MOTORS)
        wheel_vel = {m: chassis_states[m]["velocity"] for m in WHEEL_MOTORS}
        body_vel = self._wheel_vel_to_body(
            wheel_vel["wheel_left"],
            wheel_vel["wheel_back"],
            wheel_vel["wheel_right"],
        )
        obs.update(body_vel)

        lift_states = self.chassis_bus.sync_read_all_states([LIFT_MOTOR])
        self._last_lift_pos_deg = float(lift_states[LIFT_MOTOR]["position"])
        obs["lift.pos"] = self._last_lift_pos_deg

        self._last_limit_state = self.limit_reader.get_state()
        obs["lift.top_limit"] = bool(self._last_limit_state["top_limit"])
        obs["lift.bottom_limit"] = bool(self._last_limit_state["bottom_limit"])
        obs["lift.limit_stale"] = bool(self._last_limit_state["stale"])

        for cam_key, cam in self.cameras.items():
            obs[cam_key] = cam.read_latest()

        return obs

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        arm_goal = {
            k.removesuffix(".pos"): v
            for k, v in action.items()
            if k.endswith(".pos") and not k.startswith("lift")
        }

        if self.config.max_relative_target is not None:
            present = self.follower_bus.sync_read("Present_Position")
            goal_present = {k: (arm_goal[k], present[k]) for k in arm_goal}
            arm_goal = ensure_safe_goal_position(goal_present, self.config.max_relative_target)

        mit_commands = {
            motor: (self._follower_kp[motor], self._follower_kd[motor], pos_deg, 0.0, 0.0)
            for motor, pos_deg in arm_goal.items()
        }
        self.follower_bus._mit_control_batch(mit_commands)

        try:
            lift_state = self.chassis_bus.sync_read_all_states([LIFT_MOTOR])
            self._last_lift_pos_deg = float(lift_state[LIFT_MOTOR]["position"])
        except Exception as exc:
            logger.warning("Failed to refresh lift position before send_action: %s", exc)

        self._last_limit_state = self.limit_reader.get_state()

        x_vel = float(action.get("x.vel", 0.0))
        y_vel = float(action.get("y.vel", 0.0))
        theta_vel = float(action.get("theta.vel", 0.0))
        lift_pos_deg = float(action.get("lift.pos", self._last_lift_pos_deg))

        safe_x_vel, safe_y_vel, safe_theta_vel, safe_lift_pos_deg = self._coordinate_base_and_lift(
            x_vel,
            y_vel,
            theta_vel,
            lift_pos_deg,
        )

        wheel_vel_rad = self._body_to_wheel_vel(safe_x_vel, safe_y_vel, safe_theta_vel)
        wheel_mit = {
            motor: (
                self.config.chassis_wheel_kp,
                self.config.chassis_wheel_kd,
                0.0,
                vel_rad,
                self.config.chassis_wheel_torque_ff,
            )
            for motor, vel_rad in wheel_vel_rad.items()
        }
        self.chassis_bus._mit_control_batch(wheel_mit)

        self.chassis_bus._mit_control(
            LIFT_MOTOR,
            kp=self._lift_kp,
            kd=self._lift_kd,
            position_degrees=safe_lift_pos_deg,
            velocity_deg_per_sec=0.0,
            torque=0.0,
        )

        sent = {f"{m}.pos": v for m, v in arm_goal.items()}
        sent.update({"x.vel": safe_x_vel, "y.vel": safe_y_vel, "theta.vel": safe_theta_vel})
        sent["lift.pos"] = safe_lift_pos_deg
        return sent

    @check_if_not_connected
    def disconnect(self) -> None:
        stop_cmds = {m: (0.0, 0.1, 0.0, 0.0, 0.0) for m in WHEEL_MOTORS}
        try:
            self.chassis_bus._mit_control_batch(stop_cmds)
        except Exception:
            pass

        try:
            self.limit_reader.stop()
        except Exception:
            pass

        if self.config.disable_torque_on_disconnect:
            self.follower_bus.disable_torque()
            self.chassis_bus.disable_torque()

        self.leader_bus.disconnect(self.config.disable_torque_on_disconnect)
        self.follower_bus.disconnect(disable_torque=False)
        self.chassis_bus.disconnect(disable_torque=False)

        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")

    def _speed_scale_by_lift_pos(self, lift_pos_deg: float) -> float:
        if lift_pos_deg < self.config.lift_low_pos_deg:
            return 1.0
        if lift_pos_deg < self.config.lift_high_pos_deg:
            return self.config.lift_mid_speed_scale
        return self.config.lift_high_speed_scale

    def _coordinate_base_and_lift(
        self,
        x_vel: float,
        y_vel: float,
        theta_vel: float,
        target_lift_pos_deg: float,
    ) -> tuple[float, float, float, float]:
        current_lift_pos_deg = self._last_lift_pos_deg
        top_limit = bool(self._last_limit_state.get("top_limit", False))
        bottom_limit = bool(self._last_limit_state.get("bottom_limit", False))
        limit_stale = bool(self._last_limit_state.get("stale", True))

        scale = self._speed_scale_by_lift_pos(current_lift_pos_deg)
        safe_x_vel = x_vel * scale
        safe_y_vel = y_vel * scale
        safe_theta_vel = theta_vel * scale

        base_speed = math.sqrt(safe_x_vel**2 + safe_y_vel**2)
        safe_lift_pos_deg = target_lift_pos_deg
        if base_speed > self.config.base_fast_speed_mps and target_lift_pos_deg > current_lift_pos_deg:
            safe_lift_pos_deg = current_lift_pos_deg

        if top_limit and safe_lift_pos_deg > current_lift_pos_deg:
            safe_lift_pos_deg = current_lift_pos_deg
        if bottom_limit and safe_lift_pos_deg < current_lift_pos_deg:
            safe_lift_pos_deg = current_lift_pos_deg

        if limit_stale and self.config.stop_lift_on_limit_serial_loss:
            safe_lift_pos_deg = current_lift_pos_deg

        return safe_x_vel, safe_y_vel, safe_theta_vel, safe_lift_pos_deg

    def _body_to_wheel_vel(self, x: float, y: float, theta_deg_per_s: float) -> dict[str, float]:
        r = self.config.wheel_radius
        L = self.config.base_radius
        theta_rad_s = np.radians(theta_deg_per_s)

        angles = np.radians(np.array([240, 0, 120]) - 90)
        M = np.array([[np.cos(a), np.sin(a), L] for a in angles])

        body_vel = np.array([x, y, theta_rad_s])
        wheel_linear = M @ body_vel
        wheel_angular = wheel_linear / r

        max_rad_s = self.config.max_wheel_rpm * (2 * np.pi / 60)
        max_computed = np.max(np.abs(wheel_angular))
        if max_computed > max_rad_s:
            wheel_angular = wheel_angular * (max_rad_s / max_computed)

        signs = self.config.wheel_direction_signs
        return {
            "wheel_left": float(wheel_angular[0]) * float(signs.get("wheel_left", 1)),
            "wheel_back": float(wheel_angular[1]) * float(signs.get("wheel_back", 1)),
            "wheel_right": float(wheel_angular[2]) * float(signs.get("wheel_right", 1)),
        }

    def _wheel_vel_to_body(self, left_rad_s: float, back_rad_s: float, right_rad_s: float) -> dict[str, float]:
        r = self.config.wheel_radius
        L = self.config.base_radius

        signs = self.config.wheel_direction_signs
        left_rad_s = left_rad_s * float(signs.get("wheel_left", 1))
        back_rad_s = back_rad_s * float(signs.get("wheel_back", 1))
        right_rad_s = right_rad_s * float(signs.get("wheel_right", 1))

        angles = np.radians(np.array([240, 0, 120]) - 90)
        M = np.array([[np.cos(a), np.sin(a), L] for a in angles])

        wheel_linear = np.array([left_rad_s, back_rad_s, right_rad_s]) * r
        body_vel = np.linalg.inv(M) @ wheel_linear
        x, y, theta_rad_s = body_vel

        return {
            "x.vel": float(x),
            "y.vel": float(y),
            "theta.vel": float(np.degrees(theta_rad_s)),
        }
