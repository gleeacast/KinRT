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

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("diyrobot")
@dataclass
class DIYRobotConfig(RobotConfig):
    """
    Configuration for the DIYRobot bimanual mobile platform:
      - 14x Feetech STS3215 servos as leader arms (left + right, 7 each)
        connected via a single USB-to-servo bus
      - 14x RobStride joint motors as follower arms (left + right, 7 each)
        connected via a USB-to-CAN adapter
      - 3x Damiao motors for the omni-wheel chassis (velocity control)
        + 1x Damiao motor for the lift platform (position control)
        connected via a second USB-to-CAN adapter

    Port assignments:
      leader_port   : USB serial port for Feetech leader servos
      follower_port : CH340 USB-CAN AT port for RobStride followers
      chassis_port  : USB-CAN port for Damiao chassis and lift motors
    """

    # --- Serial / CAN ports ---
    leader_port: str = "/dev/diyrobot/leader"
    follower_port: str = "/dev/diyrobot/follower"
    chassis_port: str = "/dev/diyrobot/chassis"

    # --- Lift limit sensor serial (ESP32 + U-shaped opto sensors) ---
    lift_limit_port: str = "/dev/diyrobot/lift-limits"
    lift_limit_baudrate: int = 115200
    lift_limit_stale_timeout_s: float = 1.0
    stop_lift_on_limit_serial_loss: bool = True

    # --- Lift homing / zeroing ---
    auto_home_lift_on_connect: bool = False
    lift_home_direction: int = -1
    lift_home_step_deg: float = 2.0
    lift_home_kp: float = 8.0
    lift_home_kd: float = 0.3
    lift_home_settle_s: float = 0.2
    lift_zero_pos_deg: float = 0.0

    # --- Safety ---
    max_relative_target: float | dict[str, float] | None = None
    disable_torque_on_disconnect: bool = True

    # --- Normalization ---
    # Use degrees for leader/follower arm joints
    use_degrees: bool = True

    # --- Cameras ---
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # --- Chassis kinematics ---
    # Physical dimensions of the omni-wheel base
    wheel_radius: float = 0.05    # metres
    base_radius: float = 0.125    # metres, centre-to-wheel distance
    max_wheel_rpm: float = 100.0  # safety cap for wheel speed commands

    # Effective MIT wheel-control defaults validated on the live chassis.
    # These are intentionally stronger than the initial conservative values,
    # because low-gain / zero-torque settings caused the wheels to stall when touched.
    chassis_wheel_kp: float = 0.0
    chassis_wheel_kd: float = 3.0
    chassis_wheel_torque_ff: float = 3.0
    chassis_test_velocity_rad_s: float = 5.0

    # Per-wheel command sign for real-world direction alignment.
    # Use +1 when positive command already matches expectation, -1 when reversed.
    wheel_direction_signs: dict[str, int] = field(
        default_factory=lambda: {
            "wheel_left": 1,
            "wheel_back": 1,
            "wheel_right": 1,
        }
    )

    # --- Base / lift coordination ---
    lift_low_pos_deg: float = 30.0
    lift_high_pos_deg: float = 90.0
    lift_mid_speed_scale: float = 0.6
    lift_high_speed_scale: float = 0.3
    base_fast_speed_mps: float = 0.2
