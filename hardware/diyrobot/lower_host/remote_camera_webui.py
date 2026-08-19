#!/usr/bin/env python3
"""Serve a dependency-light V4L2 camera preview for remote alignment checks."""

import argparse
import ctypes
import fcntl
import json
import mmap
import os
import select
import signal
import socketserver
import struct
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse


VIDIOC_QUERYCAP = 0x80685600
VIDIOC_ENUM_FMT = 0xC0405602
VIDIOC_S_FMT = 0xC0D05605
VIDIOC_REQBUFS = 0xC0145608
VIDIOC_QUERYBUF = 0xC0585609
VIDIOC_QBUF = 0xC058560F
VIDIOC_DQBUF = 0xC0585611
VIDIOC_STREAMON = 0x40045612
VIDIOC_STREAMOFF = 0x40045613

V4L2_BUF_TYPE_VIDEO_CAPTURE = 1
V4L2_MEMORY_MMAP = 1
V4L2_FIELD_ANY = 0
V4L2_PIX_FMT_MJPEG = 0x47504A4D
V4L2_CAP_VIDEO_CAPTURE = 0x00000001
V4L2_CAP_STREAMING = 0x04000000


def xioctl(fd, req, arg):
    while True:
        try:
            return fcntl.ioctl(fd, req, arg)
        except InterruptedError:
            continue


def cstr(raw):
    return raw.rstrip(b"\0").decode("utf-8", errors="ignore")


def fourcc(value):
    return "".join(chr((value >> (8 * i)) & 255) for i in range(4))


class V4L2Capability(ctypes.Structure):
    _fields_ = [
        ("driver", ctypes.c_char * 16),
        ("card", ctypes.c_char * 32),
        ("bus_info", ctypes.c_char * 32),
        ("version", ctypes.c_uint32),
        ("capabilities", ctypes.c_uint32),
        ("device_caps", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32 * 3),
    ]


class V4L2FmtDesc(ctypes.Structure):
    _fields_ = [
        ("index", ctypes.c_uint32),
        ("type", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("description", ctypes.c_char * 32),
        ("pixelformat", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32 * 4),
    ]


class V4L2Format(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        ("pixelformat", ctypes.c_uint32),
        ("field", ctypes.c_uint32),
        ("bytesperline", ctypes.c_uint32),
        ("sizeimage", ctypes.c_uint32),
        ("colorspace", ctypes.c_uint32),
        ("priv", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("ycbcr_enc", ctypes.c_uint32),
        ("quantization", ctypes.c_uint32),
        ("xfer_func", ctypes.c_uint32),
    ]


class V4L2RequestBuffers(ctypes.Structure):
    _fields_ = [
        ("count", ctypes.c_uint32),
        ("type", ctypes.c_uint32),
        ("memory", ctypes.c_uint32),
        ("capabilities", ctypes.c_uint32),
        ("flags", ctypes.c_uint8),
        ("reserved", ctypes.c_uint8 * 3),
    ]


class TimeVal(ctypes.Structure):
    _fields_ = [("tv_sec", ctypes.c_long), ("tv_usec", ctypes.c_long)]


class V4L2Timecode(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("frames", ctypes.c_uint8),
        ("seconds", ctypes.c_uint8),
        ("minutes", ctypes.c_uint8),
        ("hours", ctypes.c_uint8),
        ("userbits", ctypes.c_uint8 * 4),
    ]


class V4L2Buffer(ctypes.Structure):
    _fields_ = [
        ("index", ctypes.c_uint32),
        ("type", ctypes.c_uint32),
        ("bytesused", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("field", ctypes.c_uint32),
        ("timestamp", TimeVal),
        ("timecode", V4L2Timecode),
        ("sequence", ctypes.c_uint32),
        ("memory", ctypes.c_uint32),
        ("m_offset", ctypes.c_uint32),
        ("m_padding", ctypes.c_uint32),
        ("length", ctypes.c_uint32),
        ("reserved2", ctypes.c_uint32),
        ("request_fd", ctypes.c_int32),
    ]


class Camera:
    def __init__(self, device, width, height, buffers=4):
        self.device = device
        self.width = width
        self.height = height
        self.buffer_count = buffers
        self.fd = None
        self.maps = []
        self.running = False

    def open(self):
        self.fd = os.open(self.device, os.O_RDWR | os.O_NONBLOCK)

        fmt = bytearray(208)
        struct.pack_into(
            "I I I I I I I I I I I I",
            fmt,
            8,
            self.width,
            self.height,
            V4L2_PIX_FMT_MJPEG,
            V4L2_FIELD_ANY,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        )
        struct.pack_into("I", fmt, 0, V4L2_BUF_TYPE_VIDEO_CAPTURE)
        xioctl(self.fd, VIDIOC_S_FMT, fmt)
        self.width, self.height = struct.unpack_from("II", fmt, 8)

        req = V4L2RequestBuffers()
        req.count = self.buffer_count
        req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
        req.memory = V4L2_MEMORY_MMAP
        xioctl(self.fd, VIDIOC_REQBUFS, req)
        if req.count < 2:
            raise RuntimeError("camera did not allocate enough mmap buffers")

        for index in range(req.count):
            buf = V4L2Buffer()
            buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
            buf.memory = V4L2_MEMORY_MMAP
            buf.index = index
            xioctl(self.fd, VIDIOC_QUERYBUF, buf)
            mm = mmap.mmap(
                self.fd,
                buf.length,
                mmap.MAP_SHARED,
                mmap.PROT_READ | mmap.PROT_WRITE,
                offset=buf.m_offset,
            )
            self.maps.append(mm)
            xioctl(self.fd, VIDIOC_QBUF, buf)

        typ = struct.pack("I", V4L2_BUF_TYPE_VIDEO_CAPTURE)
        xioctl(self.fd, VIDIOC_STREAMON, typ)
        self.running = True

    def read_frame(self):
        if not self.running:
            return None
        readable, _, _ = select.select([self.fd], [], [], 1.0)
        if not readable:
            return None
        buf = V4L2Buffer()
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
        buf.memory = V4L2_MEMORY_MMAP
        try:
            xioctl(self.fd, VIDIOC_DQBUF, buf)
        except BlockingIOError:
            return None
        try:
            return bytes(self.maps[buf.index][: buf.bytesused])
        finally:
            xioctl(self.fd, VIDIOC_QBUF, buf)

    def close(self):
        if self.fd is not None:
            if self.running:
                typ = struct.pack("I", V4L2_BUF_TYPE_VIDEO_CAPTURE)
                try:
                    xioctl(self.fd, VIDIOC_STREAMOFF, typ)
                except OSError:
                    pass
            for mm in self.maps:
                mm.close()
            os.close(self.fd)
        self.fd = None
        self.running = False


def scan_cameras():
    devices = []
    for name in sorted(os.listdir("/dev")):
        if not name.startswith("video"):
            continue
        path = "/dev/" + name
        try:
            fd = os.open(path, os.O_RDWR | os.O_NONBLOCK)
        except OSError as exc:
            devices.append({"device": path, "error": str(exc), "usable": False})
            continue

        try:
            cap = V4L2Capability()
            xioctl(fd, VIDIOC_QUERYCAP, cap)
            caps = cap.device_caps or cap.capabilities
            formats = []
            index = 0
            while True:
                desc = V4L2FmtDesc()
                desc.index = index
                desc.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
                try:
                    xioctl(fd, VIDIOC_ENUM_FMT, desc)
                except OSError:
                    break
                formats.append(
                    {
                        "fourcc": fourcc(desc.pixelformat),
                        "description": cstr(desc.description),
                    }
                )
                index += 1
            has_mjpeg = any(item["fourcc"] == "MJPG" for item in formats)
            devices.append(
                {
                    "device": path,
                    "driver": cstr(cap.driver),
                    "card": cstr(cap.card),
                    "bus": cstr(cap.bus_info),
                    "formats": formats,
                    "usable": bool(
                        caps & V4L2_CAP_VIDEO_CAPTURE
                        and caps & V4L2_CAP_STREAMING
                        and has_mjpeg
                    ),
                }
            )
        except OSError as exc:
            devices.append({"device": path, "error": str(exc), "usable": False})
        finally:
            os.close(fd)
    return devices


class FrameHub:
    def __init__(self, camera):
        self.camera = camera
        self.condition = threading.Condition()
        self.frame = None
        self.frame_id = 0
        self.error = None
        self.started_at = time.time()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def _run(self):
        try:
            self.camera.open()
            while True:
                frame = self.camera.read_frame()
                if frame:
                    with self.condition:
                        self.frame = frame
                        self.frame_id += 1
                        self.condition.notify_all()
        except Exception as exc:
            with self.condition:
                self.error = str(exc)
                self.condition.notify_all()

    def wait_frame(self, last_id, timeout=5.0):
        with self.condition:
            self.condition.wait_for(
                lambda: self.frame_id != last_id or self.error, timeout=timeout
            )
            return self.frame_id, self.frame, self.error

    def close(self):
        self.camera.close()
        with self.condition:
            self.error = self.error or "camera switched"
            self.condition.notify_all()


class CameraManager:
    def __init__(self, device, width, height):
        self.width = width
        self.height = height
        self.lock = threading.Lock()
        self.hub = None
        self.device = None
        self.switch(device)

    def switch(self, device):
        if not device.startswith("/dev/video"):
            raise ValueError("device must be /dev/video*")
        if not os.path.exists(device):
            raise FileNotFoundError(device)
        with self.lock:
            if self.device == device and self.hub:
                hub = self.hub
                return {
                    "device": self.device,
                    "size": f"{hub.camera.width}x{hub.camera.height}",
                    "width": hub.camera.width,
                    "height": hub.camera.height,
                    "frames": hub.frame_id,
                    "uptime": int(time.time() - hub.started_at),
                    "error": hub.error or "",
                }
        new_hub = FrameHub(Camera(device, self.width, self.height))
        new_hub.start()
        with self.lock:
            old_hub = self.hub
            self.hub = new_hub
            self.device = device
        if old_hub:
            old_hub.close()
        return self.status()

    def current(self):
        with self.lock:
            return self.hub

    def status(self):
        hub = self.current()
        return {
            "device": self.device,
            "size": f"{hub.camera.width}x{hub.camera.height}",
            "width": hub.camera.width,
            "height": hub.camera.height,
            "frames": hub.frame_id,
            "uptime": int(time.time() - hub.started_at),
            "error": hub.error or "",
        }

    def close(self):
        hub = self.current()
        if hub:
            hub.close()


def make_handler(manager):
    class Handler(BaseHTTPRequestHandler):
        server_version = "CameraWebUI/1.0"

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path in ("/", "/index.html"):
                self._index()
            elif parsed.path == "/stream.mjpg":
                self._stream()
            elif parsed.path == "/status":
                self._status()
            elif parsed.path == "/api/devices":
                self._json({"devices": scan_cameras(), "current": manager.device})
            elif parsed.path == "/api/select":
                self._select(parsed.query)
            elif parsed.path == "/api/status":
                self._json(manager.status())
            else:
                self.send_error(404)

        def _index(self):
            html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Camera Live</title>
  <style>
    :root {{ color-scheme: dark; font-family: Arial, sans-serif; }}
    body {{ margin: 0; min-height: 100vh; background: #101418; color: #f2f5f7; }}
    header {{ min-height: 56px; display: flex; align-items: center; gap: 12px; flex-wrap: wrap; padding: 10px 18px; background: #171d22; border-bottom: 1px solid #2a333a; box-sizing: border-box; }}
    h1 {{ font-size: 18px; margin: 0; font-weight: 700; }}
    .pill {{ padding: 4px 9px; border: 1px solid #3d4a54; border-radius: 999px; color: #b9c6ce; font-size: 13px; }}
    select, button {{ height: 34px; border: 1px solid #3d4a54; background: #202830; color: #f2f5f7; border-radius: 6px; padding: 0 10px; }}
    button {{ cursor: pointer; font-weight: 700; }}
    button:disabled {{ opacity: .5; cursor: default; }}
    main {{ height: calc(100vh - 77px); display: grid; place-items: center; padding: 16px; box-sizing: border-box; }}
    img {{ max-width: 100%; max-height: 100%; background: #050607; border: 1px solid #2a333a; }}
    #cameraSelect {{ min-width: 270px; max-width: min(560px, 100%); }}
  </style>
</head>
<body>
  <header>
    <h1>Camera Live</h1>
    <select id="cameraSelect" aria-label="USB camera"></select>
    <button id="scanBtn" type="button">Scan</button>
    <button id="selectBtn" type="button">Switch</button>
    <span class="pill" id="statusPill">--</span>
  </header>
  <main><img id="stream" src="/stream.mjpg" alt="live camera stream"></main>
  <script>
    const select = document.getElementById('cameraSelect');
    const statusPill = document.getElementById('statusPill');
    const stream = document.getElementById('stream');
    const scanBtn = document.getElementById('scanBtn');
    const selectBtn = document.getElementById('selectBtn');

    function optionLabel(item) {{
      const formats = (item.formats || []).map(f => f.fourcc).join('/');
      const name = item.card || item.driver || item.device;
      return `${{item.device}}  ${{name}}  ${{formats}}`;
    }}

    async function scan() {{
      scanBtn.disabled = true;
      try {{
        const res = await fetch('/api/devices', {{ cache: 'no-store' }});
        const data = await res.json();
        select.innerHTML = '';
        for (const item of data.devices) {{
          const opt = document.createElement('option');
          opt.value = item.device;
          opt.textContent = item.usable ? optionLabel(item) : `${{item.device}}  unavailable`;
          opt.disabled = !item.usable;
          if (item.device === data.current) opt.selected = true;
          select.appendChild(opt);
        }}
        await refreshStatus();
      }} finally {{
        scanBtn.disabled = false;
      }}
    }}

    async function choose() {{
      const device = select.value;
      if (!device) return;
      selectBtn.disabled = true;
      statusPill.textContent = 'switching';
      try {{
        const res = await fetch('/api/select?device=' + encodeURIComponent(device), {{ cache: 'no-store' }});
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'switch failed');
        stream.src = '/stream.mjpg?t=' + Date.now();
        await refreshStatus();
      }} catch (err) {{
        statusPill.textContent = err.message;
      }} finally {{
        selectBtn.disabled = false;
      }}
    }}

    async function refreshStatus() {{
      const res = await fetch('/api/status', {{ cache: 'no-store' }});
      const data = await res.json();
      statusPill.textContent = `${{data.device}}  ${{data.size}}  frames ${{data.frames}}${{data.error ? '  ' + data.error : ''}}`;
    }}

    scanBtn.addEventListener('click', scan);
    selectBtn.addEventListener('click', choose);
    setInterval(refreshStatus, 2000);
    scan();
  </script>
</body>
</html>"""
            data = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _status(self):
            status = manager.status()
            body = (
                f"device={status['device']}\n"
                f"size={status['size']}\n"
                f"frames={status['frames']}\n"
                f"uptime={status['uptime']}\n"
                f"error={status['error']}\n"
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, data, code=200):
            body = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _select(self, query):
            device = parse_qs(query).get("device", [""])[0]
            try:
                status = manager.switch(device)
            except Exception as exc:
                self._json({"error": str(exc)}, code=400)
                return
            self._json(status)

        def _stream(self):
            hub = manager.current()
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            last_id = -1
            while True:
                frame_id, frame, error = hub.wait_frame(last_id)
                if error:
                    break
                if frame_id == last_id or not frame:
                    continue
                last_id = frame_id
                try:
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii"))
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    break

        def log_message(self, fmt, *args):
            print("%s - %s" % (self.address_string(), fmt % args), flush=True)

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="/dev/video5")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()

    manager = CameraManager(args.device, args.width, args.height)

    class Server(ThreadingHTTPServer):
        daemon_threads = True
        allow_reuse_address = True

    httpd = Server((args.host, args.port), make_handler(manager))

    def shutdown(_signum, _frame):
        manager.close()
        httpd.shutdown()

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    print(f"serving http://{args.host}:{args.port}/ from {args.device}", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
