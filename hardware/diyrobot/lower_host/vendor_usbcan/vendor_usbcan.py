#!/usr/bin/env python3
"""Implement the vendor CDC framing used by DIYRobot base and lift motors."""

from __future__ import annotations

import json
import logging
import os
import struct
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import serial

logger = logging.getLogger(__name__)

P_MIN = -3.141593
P_MAX = 3.141593
V_MIN = -45.0
V_MAX = 45.0
KP_MIN = 0.0
KP_MAX = 500.0
KD_MIN = 0.0
KD_MAX = 5.0
T_MIN = -10.0
T_MAX = 10.0
A_MIN = 0.0
A_MAX = 20.0
MONITOR_LOG_PATH = Path(
    os.environ.get(
        "DIYROBOT_VENDOR_MONITOR_LOG",
        "/tmp/diyrobot/vendor_usbcan_monitor.jsonl",
    )
)


@dataclass
class VendorMotorState:
    position: float = 0.0
    velocity: float = 0.0
    torque: float = 0.0
    temp_mos: float = 0.0
    temp_rotor: float = 0.0
    last_update: float = 0.0


class VendorUsbCanBus:
    """Damiao USB-CAN vendor-CDC transport.

    This bus speaks the verified vendor serial framing used by the HDSC CDC device.
    It is intentionally minimal: just enough for DIYRobot chassis and lift use.
    """

    def __init__(self, port: str = "/dev/diyrobot/chassis", baudrate: int = 921600):
        self.port = port
        self.baudrate = baudrate
        self.ser: serial.Serial | None = None
        self._lock = threading.Lock()
        self._states: dict[int, VendorMotorState] = {}
        self._monitor_lock = threading.Lock()

    def _monitor_event(self, event: str, data: bytes | None = None, **extra) -> None:
        try:
            MONITOR_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            rec = {
                "ts": time.time(),
                "event": event,
                "port": self.port,
                "baudrate": self.baudrate,
            }
            if data is not None:
                rec["hex"] = data.hex(" ")
                rec["len"] = len(data)
            rec.update(extra)
            with self._monitor_lock:
                with MONITOR_LOG_PATH.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def connect(self) -> None:
        if self.ser and self.ser.is_open:
            return
        self.ser = serial.Serial(self.port, self.baudrate, timeout=0.05)
        time.sleep(0.1)
        self._monitor_event("connect")
        self._drain()
        self.send_can_baudrate_1m()
        self.send_heartbeat()
        logger.info("VendorUsbCanBus connected on %s @ %s", self.port, self.baudrate)

    def disconnect(self) -> None:
        if self.ser and self.ser.is_open:
            self.ser.close()
        self._monitor_event("disconnect")
        self.ser = None

    def ensure_connected(self) -> None:
        if not self.ser or not self.ser.is_open:
            raise RuntimeError("VendorUsbCanBus is not connected")

    def _drain(self) -> None:
        if not self.ser:
            return
        while self.ser.in_waiting:
            drained = self.ser.read(self.ser.in_waiting)
            if drained:
                self._monitor_event("drain", drained)
            time.sleep(0.01)

    @staticmethod
    def _clamp(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    @staticmethod
    def _float_to_uint(x: float, x_min: float, x_max: float, bits: int) -> int:
        span = x_max - x_min
        offset = x_min
        return int((x - offset) * ((1 << bits) - 1) / span)

    @staticmethod
    def _uint_to_float(x_int: int, x_min: float, x_max: float, bits: int) -> float:
        span = x_max - x_min
        offset = x_min
        return float(x_int * span / ((1 << bits) - 1) + offset)

    def _write(self, data: bytes) -> bytes:
        self.ensure_connected()
        with self._lock:
            assert self.ser is not None
            self._monitor_event("tx", data)
            self.ser.write(data)
            self.ser.flush()
            time.sleep(0.02)
            rx = bytearray()
            end = time.time() + 0.08
            while time.time() < end:
                n = self.ser.in_waiting
                if n:
                    rx.extend(self.ser.read(n))
                time.sleep(0.005)
            if rx:
                self._monitor_event("rx", bytes(rx))
                self._parse_rx(bytes(rx))
            else:
                self._monitor_event("rx_timeout")
            return bytes(rx)

    def _parse_rx(self, data: bytes) -> None:
        i = 0
        while i < len(data):
            if data[i] != 0xAA:
                i += 1
                continue
            if i + 16 > len(data):
                break
            if data[i + 15] != 0x55:
                i += 1
                continue
            frame = data[i : i + 16]
            cmd = frame[1]
            fmt = frame[2]
            payload = frame[7:15]
            self._monitor_event("frame", frame, cmd=cmd, fmt=fmt, payload_hex=payload.hex(" "))
            if cmd == 0x11 and (fmt & 0x3F) == 8:
                self._update_state_from_feedback(payload)
            i += 16

    def _update_state_from_feedback(self, payload: bytes) -> None:
        if len(payload) < 8:
            return
        motor_id = int(payload[0])
        p_uint = (int(payload[1]) << 8) | int(payload[2])
        v_uint = (int(payload[3]) << 4) | (int(payload[4]) >> 4)
        t_uint = ((int(payload[4]) & 0x0F) << 8) | int(payload[5])

        pos = self._uint_to_float(p_uint, P_MIN, P_MAX, 16)
        vel = self._uint_to_float(v_uint, V_MIN, V_MAX, 12)
        tq = 0.28 * self._uint_to_float(t_uint, A_MIN, A_MAX, 12)

        state = self._states.setdefault(motor_id, VendorMotorState())
        state.position = pos
        state.velocity = vel
        state.torque = tq
        state.temp_mos = float(payload[6])
        state.temp_rotor = float(payload[7])
        state.last_update = time.time()
        self._monitor_event(
            "feedback",
            payload,
            motor_id=motor_id,
            position=pos,
            velocity=vel,
            torque=tq,
            temp_mos=state.temp_mos,
            temp_rotor=state.temp_rotor,
        )

    def send_can_baudrate_1m(self) -> bytes:
        return self._write(bytes([0x55, 0x05, 0x00, 0xAA, 0x55]))

    def send_heartbeat(self) -> bytes:
        return self._write(bytes([0x55, 0x04, 0xAA, 0x55]))

    def send_usart_baudrate(self, baudrate: int | None = None) -> bytes:
        baud = self.baudrate if baudrate is None else baudrate
        data = bytearray([0x55, 0xAA, 0, 0, 0, 0, 0, 0, 0, 0xAA, 0x55])
        data[2:6] = struct.pack('<i', baud)
        return self._write(bytes(data))

    def _wrap_can(self, can_id: int, payload: list[int]) -> bytes:
        return bytes([
            0x55, 0xAA, 0x1E, 0x01, 0x01, 0x00, 0x00, 0x00,
            0x0A, 0x00, 0x00, 0x00, 0x00,
            can_id & 0xFF, (can_id >> 8) & 0xFF, 0x00, 0x00, 0x00,
            len(payload) & 0xFF, 0x00, 0x00,
            *payload,
            0x88,
        ])

    def _wrap_can8(self, motor_id: int, payload8: list[int]) -> bytes:
        return self._wrap_can(motor_id, payload8)

    def motor_enable(self, motor_id: int) -> bytes:
        return self._write(self._wrap_can8(motor_id, [0xFF] * 7 + [0xFC]))

    def motor_disable(self, motor_id: int) -> bytes:
        return self._write(self._wrap_can8(motor_id, [0xFF] * 7 + [0xFD]))

    def motor_set_zero(self, motor_id: int) -> bytes:
        return self._write(self._wrap_can8(motor_id, [0xFF] * 7 + [0xFE]))

    def mit_control(self, motor_id: int, p: float, v: float, kp: float, kd: float, t: float) -> bytes:
        p = self._clamp(p, P_MIN, P_MAX)
        v = self._clamp(v, V_MIN, V_MAX)
        kp = self._clamp(kp, KP_MIN, KP_MAX)
        kd = self._clamp(kd, KD_MIN, KD_MAX)
        t = self._clamp(t, T_MIN, T_MAX)

        p_u = self._float_to_uint(p, P_MIN, P_MAX, 16)
        v_u = self._float_to_uint(v, V_MIN, V_MAX, 12)
        kp_u = self._float_to_uint(kp, KP_MIN, KP_MAX, 12)
        kd_u = self._float_to_uint(kd, KD_MIN, KD_MAX, 12)
        t_u = self._float_to_uint(t, T_MIN, T_MAX, 12)

        payload = [0] * 8
        payload[0] = (p_u >> 8) & 0xFF
        payload[1] = p_u & 0xFF
        payload[2] = (v_u >> 4) & 0xFF
        payload[3] = ((v_u & 0x0F) << 4) | ((kp_u >> 8) & 0x0F)
        payload[4] = kp_u & 0xFF
        payload[5] = (kd_u >> 4) & 0xFF
        payload[6] = ((kd_u & 0x0F) << 4) | ((t_u >> 8) & 0x0F)
        payload[7] = t_u & 0xFF
        return self._write(self._wrap_can8(motor_id, payload))

    def control_pos_vel(self, motor_id: int, pos: float, vel: float) -> bytes:
        payload = list(struct.pack('<ff', pos, vel))
        return self._write(self._wrap_can(motor_id + 0x100, payload))

    def control_vel(self, motor_id: int, vel: float) -> bytes:
        payload = list(struct.pack('<f', vel))
        return self._write(self._wrap_can(motor_id + 0x200, payload))

    def get_state(self, motor_id: int) -> VendorMotorState:
        return self._states.get(motor_id, VendorMotorState())
