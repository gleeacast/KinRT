#!/usr/bin/env python3
"""
RobStride official CH340 USB-CAN driver.

This driver is for RobStride's official serial USB-CAN module, whose wire format
is an AT-style serial envelope:

    b"AT" + ((can_id << 3) | 0x04).to_bytes(4, "big") + dlc + data + b"\r\n"

It is not SocketCAN and not Lawicel/SLCAN.  The class below keeps one serial
reader thread alive, parses every incoming frame once, stores the latest motor
feedback, and lets request/response code match frames from a shared queue.
"""

from __future__ import annotations

import math
import os
import json
import select
import struct
import termios
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Iterable


P_MIN, P_MAX = -12.5, 12.5
V_MIN, V_MAX = -44.0, 44.0
KP_MIN, KP_MAX = 0.0, 500.0
KD_MIN, KD_MAX = 0.0, 5.0
T_MIN, T_MAX = -17.0, 17.0

BAUDS = {
    115200: termios.B115200,
    460800: getattr(termios, "B460800", termios.B115200),
    921600: getattr(termios, "B921600", termios.B115200),
    1000000: getattr(termios, "B1000000", termios.B115200),
    2000000: getattr(termios, "B2000000", termios.B115200),
}


def _float_to_uint(x: float, x_min: float, x_max: float, bits: int) -> int:
    x = min(max(float(x), x_min), x_max)
    return int((x - x_min) * ((1 << bits) - 1) / (x_max - x_min))


def _uint_to_float(x: int, x_min: float, x_max: float, bits: int) -> float:
    return (x_max - x_min) * (x & ((1 << bits) - 1)) / ((1 << bits) - 1) + x_min


def _can_id(comm_type: int, data_area2: int, motor_id: int) -> int:
    return ((comm_type & 0x1F) << 24) | ((data_area2 & 0xFFFF) << 8) | (motor_id & 0xFF)


def _at_frame(can_id: int, data: bytes) -> bytes:
    if len(data) != 8:
        raise ValueError("RobStride AT frames require exactly 8 data bytes")
    encoded_id = ((can_id & 0x1FFFFFFF) << 3) | 0x04
    return b"AT" + encoded_id.to_bytes(4, "big") + b"\x08" + data + b"\r\n"


@dataclass(frozen=True)
class RobStrideFrame:
    can_id: int
    data: bytes
    timestamp: float

    @property
    def comm_type(self) -> int:
        return (self.can_id >> 24) & 0x1F

    @property
    def area2(self) -> int:
        return (self.can_id >> 8) & 0xFFFF

    @property
    def target(self) -> int:
        return self.can_id & 0xFF


class RobStrideAtBus:
    """Robust low-level bus for RobStride official CH340 USB-CAN adapters."""

    def __init__(
        self,
        port: str = "/dev/diyrobot/follower",
        baudrate: int = 921600,
        host_id: int = 0xFD,
        receive_history: int = 4096,
        min_tx_gap_s: float = 0.012,
    ) -> None:
        if baudrate not in BAUDS:
            raise ValueError(f"Unsupported baudrate {baudrate}; known={sorted(BAUDS)}")
        self.port = port
        self.baudrate = baudrate
        self.host_id = host_id
        self.min_tx_gap_s = max(0.0, float(min_tx_gap_s))
        self._fd: int | None = None
        self._reader: threading.Thread | None = None
        self._stop = threading.Event()
        self._write_lock = threading.Lock()
        self._request_lock = threading.Lock()
        self._last_tx_monotonic = 0.0
        self._cv = threading.Condition()
        self._frames: deque[RobStrideFrame] = deque(maxlen=receive_history)
        self._latest_feedback: dict[int, dict[str, float | int]] = {}
        self._rx_errors = 0

    @property
    def is_connected(self) -> bool:
        return self._fd is not None and self._reader is not None and self._reader.is_alive()

    def connect(self) -> None:
        if self.is_connected:
            return
        fd = os.open(self.port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
        self._configure(fd)
        self._drain_fd_until_quiet(fd, timeout=2.0, quiet_s=0.15)
        self._fd = fd
        self._stop.clear()
        self._reader = threading.Thread(target=self._read_loop, name="robstride-at-reader", daemon=True)
        self._reader.start()
        time.sleep(0.05)
        self.flush()

    def disconnect(self) -> None:
        self._stop.set()
        if self._reader is not None:
            self._reader.join(timeout=1.0)
        if self._fd is not None:
            os.close(self._fd)
        self._fd = None
        self._reader = None

    def flush(self) -> None:
        with self._cv:
            self._frames.clear()
        if self._fd is not None:
            try:
                termios.tcflush(self._fd, termios.TCIOFLUSH)
            except termios.error:
                pass
            self.drain_until_quiet(timeout=1.0, quiet_s=0.10)

    def _drain_fd_until_quiet(self, fd: int, timeout: float, quiet_s: float) -> int:
        deadline = time.monotonic() + timeout
        quiet_deadline = time.monotonic() + quiet_s
        drained = 0
        while time.monotonic() < deadline:
            try:
                readable, _, _ = select.select([fd], [], [], 0.02)
                if not readable:
                    if time.monotonic() >= quiet_deadline:
                        break
                    continue
                chunk = os.read(fd, 4096)
            except BlockingIOError:
                continue
            except OSError:
                break
            if chunk:
                drained += len(chunk)
                quiet_deadline = time.monotonic() + quiet_s
        return drained

    def drain_until_quiet(self, timeout: float = 1.0, quiet_s: float = 0.10) -> bool:
        deadline = time.monotonic() + timeout
        quiet_deadline = time.monotonic() + quiet_s
        last_count = -1
        while time.monotonic() < deadline:
            with self._cv:
                count = len(self._frames)
                self._frames.clear()
            if count != last_count and count > 0:
                quiet_deadline = time.monotonic() + quiet_s
                last_count = count
            if time.monotonic() >= quiet_deadline:
                return True
            time.sleep(0.02)
        with self._cv:
            self._frames.clear()
        return False

    def _configure(self, fd: int) -> None:
        baud = BAUDS[self.baudrate]
        attrs = termios.tcgetattr(fd)
        attrs[0] = 0
        attrs[1] = 0
        attrs[2] = baud | termios.CS8 | termios.CLOCAL | termios.CREAD
        attrs[3] = 0
        attrs[4] = baud
        attrs[5] = baud
        attrs[6][termios.VMIN] = 0
        attrs[6][termios.VTIME] = 1
        termios.tcsetattr(fd, termios.TCSANOW, attrs)
        termios.tcflush(fd, termios.TCIOFLUSH)

    def _read_loop(self) -> None:
        buf = bytearray()
        while not self._stop.is_set() and self._fd is not None:
            try:
                readable, _, _ = select.select([self._fd], [], [], 0.02)
                if not readable:
                    continue
                chunk = os.read(self._fd, 4096)
            except BlockingIOError:
                continue
            except OSError:
                break
            if not chunk:
                continue
            buf.extend(chunk)
            self._parse_buffer(buf)

    def _parse_buffer(self, buf: bytearray) -> None:
        while True:
            start = buf.find(b"AT")
            if start < 0:
                if len(buf) > 64:
                    del buf[:-1]
                return
            if start:
                del buf[:start]
            if len(buf) < 17:
                return
            if buf[15:17] != b"\r\n":
                del buf[0]
                self._rx_errors += 1
                continue
            raw_id = int.from_bytes(buf[2:6], "big")
            dlc = buf[6]
            data = bytes(buf[7:15])
            del buf[:17]
            if dlc != 8:
                self._rx_errors += 1
                continue
            frame = RobStrideFrame(can_id=raw_id >> 3, data=data, timestamp=time.monotonic())
            self._ingest_frame(frame)

    def _ingest_frame(self, frame: RobStrideFrame) -> None:
        if frame.comm_type in (2, 0x18):
            motor_id = frame.area2 & 0xFF
            previous = self._latest_feedback.get(motor_id, {})
            if frame.comm_type == 2:
                fault = (frame.area2 >> 8) & 0x3F
                mode = (frame.area2 >> 14) & 0x03
            else:
                # Active-report frames carry the same kinematic payload but no
                # status bits in area2, so preserve the most recent status.
                fault = int(previous.get("fault", 0))
                mode = int(previous.get("mode_status", 0))
            data = frame.data
            self._latest_feedback[motor_id] = {
                "position_rad": _uint_to_float((data[0] << 8) | data[1], P_MIN, P_MAX, 16),
                "velocity_rad_s": _uint_to_float((data[2] << 8) | data[3], V_MIN, V_MAX, 16),
                "torque_nm": _uint_to_float((data[4] << 8) | data[5], T_MIN, T_MAX, 16),
                "temperature_c": ((data[6] << 8) | data[7]) * 0.1,
                "fault": fault,
                "mode_status": mode,
                "timestamp": frame.timestamp,
            }
        with self._cv:
            self._frames.append(frame)
            self._cv.notify_all()

    def send_raw(self, can_id: int, data: bytes) -> None:
        if self._fd is None:
            raise RuntimeError("RobStrideAtBus is not connected")
        packet = _at_frame(can_id, data)
        with self._write_lock:
            now = time.monotonic()
            gap = now - self._last_tx_monotonic
            if gap < self.min_tx_gap_s:
                time.sleep(self.min_tx_gap_s - gap)
            os.write(self._fd, packet)
            self._last_tx_monotonic = time.monotonic()

    def wait_for(
        self,
        predicate: Callable[[RobStrideFrame], bool],
        timeout: float = 0.2,
        since: float | None = None,
    ) -> RobStrideFrame | None:
        deadline = time.monotonic() + timeout
        with self._cv:
            while True:
                for frame in reversed(self._frames):
                    if since is not None and frame.timestamp < since:
                        continue
                    if predicate(frame):
                        return frame
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._cv.wait(remaining)

    def request(
        self,
        motor_id: int,
        comm_type: int,
        data: bytes,
        response_predicate: Callable[[RobStrideFrame], bool],
        timeout: float = 0.25,
        retries: int = 3,
        inter_retry_delay: float = 0.04,
    ) -> RobStrideFrame | None:
        with self._request_lock:
            for _ in range(retries):
                with self._cv:
                    self._frames.clear()
                since = time.monotonic()
                self.send_raw(_can_id(comm_type, self.host_id, motor_id), data)
                frame = self.wait_for(response_predicate, timeout=timeout, since=since)
                if frame is not None:
                    return frame
                if inter_retry_delay:
                    time.sleep(inter_retry_delay)
        return None

    def get_device_id(self, motor_id: int, timeout: float = 0.35) -> bytes | None:
        frame = self.request(
            motor_id,
            0x00,
            bytes(8),
            lambda f: f.comm_type == 0x00 and f.area2 == motor_id and f.target == 0xFE,
            timeout=timeout,
            retries=5,
        )
        return None if frame is None else frame.data

    def read_parameter(self, motor_id: int, index: int, timeout: float = 0.35) -> bytes | None:
        req = bytes([index & 0xFF, (index >> 8) & 0xFF, 0, 0, 0, 0, 0, 0])
        frame = self.request(
            motor_id,
            0x11,
            req,
            lambda f: (
                f.comm_type == 0x11
                and f.area2 == motor_id
                and f.target == self.host_id
                and f.data[:2] == req[:2]
            ),
            timeout=timeout,
            retries=8,
        )
        return None if frame is None else frame.data

    def write_parameter_u8(self, motor_id: int, index: int, value: int) -> None:
        data = bytes([index & 0xFF, (index >> 8) & 0xFF, 0, 0, value & 0xFF, 0, 0, 0])
        self.send_raw(_can_id(0x12, self.host_id, motor_id), data)

    def write_parameter_float(self, motor_id: int, index: int, value: float) -> None:
        data = bytes([index & 0xFF, (index >> 8) & 0xFF, 0, 0]) + struct.pack("<f", float(value))
        self.send_raw(_can_id(0x12, self.host_id, motor_id), data)

    def set_motion_mode(self, motor_id: int) -> None:
        self.write_parameter_u8(motor_id, 0x7005, 0)

    def read_run_mode(self, motor_id: int) -> int | None:
        data = self.read_parameter(motor_id, 0x7005)
        return None if data is None else data[4]

    def read_run_mode_confirmed(self, motor_id: int, attempts: int = 3) -> int | None:
        for _ in range(attempts):
            mode = self.read_run_mode(motor_id)
            if mode is not None:
                return mode
            time.sleep(0.04)
        return None

    def enable(self, motor_id: int, timeout: float = 0.25) -> dict[str, float | int] | None:
        since = time.monotonic()
        self.send_raw(_can_id(0x03, self.host_id, motor_id), bytes(8))
        self.wait_for(lambda f: f.comm_type == 2 and (f.area2 & 0xFF) == motor_id, timeout=timeout, since=since)
        return self.latest_feedback(motor_id)

    def disable(self, motor_id: int, clear_error: bool = False) -> None:
        data = bytes([1 if clear_error else 0, 0, 0, 0, 0, 0, 0, 0])
        self.send_raw(_can_id(0x04, self.host_id, motor_id), data)

    def disable_confirmed(
        self,
        motor_id: int,
        clear_error: bool = False,
        timeout: float = 0.20,
        retries: int = 4,
        inter_retry_delay: float = 0.03,
    ) -> dict[str, float | int] | None:
        last_state = self.latest_feedback(motor_id, max_age_s=1.0)
        for attempt in range(max(1, retries)):
            since = time.monotonic()
            self.disable(motor_id, clear_error=clear_error and attempt == 0)
            self.wait_for(
                lambda f: f.comm_type == 2 and (f.area2 & 0xFF) == motor_id,
                timeout=timeout,
                since=since,
            )
            state = self.latest_feedback(motor_id, max_age_s=1.0)
            if state is not None:
                last_state = state
                if int(state["mode_status"]) == 0:
                    return dict(state)
            time.sleep(inter_retry_delay)
        return None if last_state is None else dict(last_state)

    def zero_position(self, motor_id: int) -> None:
        # Official Set_ZeroPos sends communication type 0x06 with data[0] = 1.
        self.disable(motor_id)
        time.sleep(0.03)
        self.send_raw(_can_id(0x06, self.host_id, motor_id), bytes([1, 0, 0, 0, 0, 0, 0, 0]))
        time.sleep(0.08)
        self.enable(motor_id)

    def motion_control(
        self,
        motor_id: int,
        position_rad: float,
        velocity_rad_s: float = 0.0,
        kp: float = 20.0,
        kd: float = 0.5,
        torque_nm: float = 0.0,
    ) -> None:
        torque_raw = _float_to_uint(torque_nm, T_MIN, T_MAX, 16)
        pos_raw = _float_to_uint(position_rad, P_MIN, P_MAX, 16)
        vel_raw = _float_to_uint(velocity_rad_s, V_MIN, V_MAX, 16)
        kp_raw = _float_to_uint(kp, KP_MIN, KP_MAX, 16)
        kd_raw = _float_to_uint(kd, KD_MIN, KD_MAX, 16)
        data = bytes(
            [
                (pos_raw >> 8) & 0xFF,
                pos_raw & 0xFF,
                (vel_raw >> 8) & 0xFF,
                vel_raw & 0xFF,
                (kp_raw >> 8) & 0xFF,
                kp_raw & 0xFF,
                (kd_raw >> 8) & 0xFF,
                kd_raw & 0xFF,
            ]
        )
        self.send_raw(_can_id(0x01, torque_raw, motor_id), data)

    def set_active_report(self, motor_id: int, enable: bool = True) -> None:
        # RobStride active-report toggle. This only changes feedback reporting;
        # it does not command motion or torque.
        f_cmd = 1 if enable else 0
        self.send_raw(_can_id(0x18, self.host_id, motor_id), bytes([1, 2, 3, 4, 5, 6, f_cmd, 8]))

    def latest_feedback(self, motor_id: int, max_age_s: float | None = None) -> dict[str, float | int] | None:
        state = self._latest_feedback.get(motor_id)
        if state is None:
            return None
        if max_age_s is not None and time.monotonic() - float(state["timestamp"]) > max_age_s:
            return None
        return dict(state)

    def wait_feedback(self, motor_id: int, timeout: float = 0.25) -> dict[str, float | int] | None:
        state = self.latest_feedback(motor_id, max_age_s=timeout)
        if state is not None:
            return state
        self.enable(motor_id, timeout=timeout)
        return self.latest_feedback(motor_id, max_age_s=timeout * 2)

    def scan_online(self, motor_ids: Iterable[int], timeout: float = 0.25) -> dict[int, bool]:
        online: dict[int, bool] = {}
        for mid in motor_ids:
            online[mid] = (
                self.read_run_mode_confirmed(mid, attempts=2) is not None
                or self.get_device_id(mid, timeout=timeout) is not None
            )
            time.sleep(0.04)
        return online


class RobStrideAtMotorsBus:
    """
    LeRobot-like motor bus wrapper for DIYRobot follower motors.

    Positions exposed to DIYRobot are degrees. Internally RobStride private
    protocol uses radians in the motion-control frame.
    """

    def __init__(
        self,
        port: str,
        motors: dict,
        baudrate: int = 921600,
        host_id: int = 0xFD,
        min_tx_gap_s: float = 0.012,
    ) -> None:
        self.port = port
        self.motors = motors
        self.bus = RobStrideAtBus(port=port, baudrate=baudrate, host_id=host_id, min_tx_gap_s=min_tx_gap_s)
        self._zero_offsets_rad = self._load_zero_offsets()

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected

    def connect(self, handshake: bool = True) -> None:
        self.bus.connect()
        if handshake:
            missing = set(self.motors)
            for _ in range(3):
                for name in list(missing):
                    motor_id = self.motors[name].id
                    online = (
                        self.bus.read_run_mode_confirmed(motor_id, attempts=2) is not None
                        or self.bus.get_device_id(motor_id) is not None
                    )
                    if online:
                        missing.remove(name)
                    time.sleep(0.04)
                if not missing:
                    break
            if missing:
                raise RuntimeError(f"RobStride motors did not respond: {sorted(missing)}")

    def disconnect(self, disable_torque: bool = True) -> None:
        if disable_torque:
            self.disable_torque()
        self.bus.disconnect()

    def configure_motors(self) -> None:
        self._zero_offsets_rad = self._load_zero_offsets()
        for motor in self.motors.values():
            self.bus.set_motion_mode(motor.id)
            time.sleep(0.005)

    def enable_torque(self) -> None:
        for motor in self.motors.values():
            self.bus.enable(motor.id)
            time.sleep(0.005)

    def disable_torque(self) -> None:
        for motor in self.motors.values():
            self.bus.disable_confirmed(motor.id, clear_error=True, timeout=0.20, retries=4, inter_retry_delay=0.03)
            time.sleep(0.01)

    def sync_read_all_states(self, motors: Iterable[str] | None = None) -> dict[str, dict[str, float | int]]:
        names = list(self.motors if motors is None else motors)
        states: dict[str, dict[str, float | int]] = {}
        for name in names:
            motor_id = self.motors[name].id
            state = self.bus.wait_feedback(motor_id, timeout=0.2)
            if state is None:
                raise RuntimeError(f"No RobStride feedback for {name} id={motor_id}")
            states[name] = {
                "position": math.degrees(float(state["position_rad"]) - self._zero_offsets_rad.get(name, 0.0)),
                "velocity": math.degrees(float(state["velocity_rad_s"])),
                "torque": float(state["torque_nm"]),
                "temperature": float(state["temperature_c"]),
                "fault": int(state["fault"]),
                "mode_status": int(state["mode_status"]),
            }
        return states

    def sync_read(self, data_name: str) -> dict[str, float]:
        if data_name not in {"Present_Position", "position", "Position"}:
            raise NotImplementedError(f"RobStrideAtMotorsBus only supports position reads, got {data_name!r}")
        states = self.sync_read_all_states()
        return {name: float(state["position"]) for name, state in states.items()}

    def read_run_modes(self) -> dict[str, int | None]:
        return {name: self.bus.read_run_mode_confirmed(motor.id) for name, motor in self.motors.items()}

    def _mit_control_batch(self, mit_commands: dict[str, tuple[float, float, float, float, float]]) -> None:
        for name, command in mit_commands.items():
            kp, kd, position_deg, velocity_deg_s, torque = command
            motor_id = self.motors[name].id
            self.bus.motion_control(
                motor_id,
                position_rad=math.radians(float(position_deg)) + self._zero_offsets_rad.get(name, 0.0),
                velocity_rad_s=math.radians(float(velocity_deg_s)),
                kp=float(kp),
                kd=float(kd),
                torque_nm=float(torque),
            )

    def _mit_control(
        self,
        motor: str,
        kp: float,
        kd: float,
        position_degrees: float,
        velocity_deg_per_sec: float = 0.0,
        torque: float = 0.0,
    ) -> None:
        self._mit_control_batch({motor: (kp, kd, position_degrees, velocity_deg_per_sec, torque)})

    def _load_zero_offsets(self) -> dict[str, float]:
        path = os.path.join(os.path.dirname(__file__), "follower_calibration.json")
        try:
            with open(path, "r", encoding="utf-8") as f:
                calib = json.load(f)
        except (OSError, json.JSONDecodeError):
            return {}
        offsets: dict[str, float] = {}
        for name in self.motors:
            value = calib.get(name, {}).get("software_zero_rad")
            if isinstance(value, (int, float)):
                offsets[name] = float(value)
        return offsets
