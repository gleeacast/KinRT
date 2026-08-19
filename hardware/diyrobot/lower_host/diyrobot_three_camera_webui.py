#!/usr/bin/env python3
"""Three independent DIYRobot camera previews.

One lightweight HTTP service exposes:

- /right_gripper
- /left_gripper
- /overhead

Each page streams one camera through MJPEG without touching the robot motors.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import threading
import time
from dataclasses import dataclass
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, quote, urlparse

import cv2

from remote_camera_webui import Camera, FrameHub


DEFAULT_CAMERAS = [
    "right_gripper=/dev/diyrobot/camera-right-wrist:640x480:mjpeg",
    "left_gripper=/dev/diyrobot/camera-left-wrist:640x480:mjpeg",
    "overhead=/dev/diyrobot/camera-overhead:640x480:opencv",
]


@dataclass(frozen=True)
class CameraConfig:
    name: str
    device: str
    width: int
    height: int
    backend: str


def parse_camera(value: str) -> CameraConfig:
    # NAME=DEVICE:WIDTHxHEIGHT[:mjpeg|opencv]
    if "=" not in value:
        raise ValueError(f"Bad camera spec {value!r}; expected NAME=DEVICE:WIDTHxHEIGHT[:BACKEND]")
    name, rest = value.split("=", 1)
    if ":" not in rest:
        raise ValueError(f"Bad camera spec {value!r}; expected NAME=DEVICE:WIDTHxHEIGHT[:BACKEND]")
    device_and_size, maybe_backend = rest.rsplit(":", 1)
    if "x" in maybe_backend.lower():
        device = device_and_size
        size = maybe_backend
        backend = "mjpeg"
    else:
        if ":" not in device_and_size:
            raise ValueError(f"Bad camera spec {value!r}; expected NAME=DEVICE:WIDTHxHEIGHT[:BACKEND]")
        device, size = device_and_size.rsplit(":", 1)
        backend = maybe_backend.strip().lower()
    if backend not in {"mjpeg", "opencv"}:
        raise ValueError(f"Bad backend {backend!r}; expected mjpeg or opencv")
    if "x" not in size.lower():
        raise ValueError(f"Bad camera size {size!r}; expected WIDTHxHEIGHT")
    width_s, height_s = size.lower().split("x", 1)
    name = name.strip()
    if not name:
        raise ValueError("Camera name cannot be empty")
    return CameraConfig(name=name, device=device.strip(), width=int(width_s), height=int(height_s), backend=backend)


class OpenCVJpegHub:
    def __init__(self, device: str, width: int, height: int, jpeg_quality: int = 80) -> None:
        self.device = device
        self.width = width
        self.height = height
        self.jpeg_quality = jpeg_quality
        self.condition = threading.Condition()
        self.frame: bytes | None = None
        self.frame_id = 0
        self.error: str | None = None
        self.started_at = time.time()
        self._stop = False
        self._cap: cv2.VideoCapture | None = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def _run(self) -> None:
        try:
            cap = cv2.VideoCapture(self.device)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._cap = cap
            if not cap.isOpened():
                raise RuntimeError(f"failed to open {self.device}")
            encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jpeg_quality)]
            while not self._stop:
                ok, image = cap.read()
                if not ok or image is None:
                    time.sleep(0.02)
                    continue
                if image.shape[1] != self.width or image.shape[0] != self.height:
                    image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_AREA)
                ok, encoded = cv2.imencode(".jpg", image, encode_params)
                if not ok:
                    continue
                with self.condition:
                    self.frame = encoded.tobytes()
                    self.frame_id += 1
                    self.condition.notify_all()
        except Exception as exc:
            with self.condition:
                self.error = str(exc)
                self.condition.notify_all()

    def wait_frame(self, last_id: int, timeout: float = 5.0):
        with self.condition:
            self.condition.wait_for(lambda: self.frame_id != last_id or self.error, timeout=timeout)
            return self.frame_id, self.frame, self.error

    def close(self) -> None:
        self._stop = True
        if self._cap is not None:
            self._cap.release()
        with self.condition:
            self.error = self.error or "camera closed"
            self.condition.notify_all()


class MultiCameraManager:
    def __init__(self, configs: list[CameraConfig]) -> None:
        self.configs = {cfg.name: cfg for cfg in configs}
        self.hubs: dict[str, FrameHub | OpenCVJpegHub] = {}

    def start(self) -> None:
        for name, cfg in self.configs.items():
            if cfg.backend == "opencv":
                hub = OpenCVJpegHub(cfg.device, cfg.width, cfg.height)
            else:
                hub = FrameHub(Camera(cfg.device, cfg.width, cfg.height))
            hub.start()
            self.hubs[name] = hub

    def names(self) -> list[str]:
        return list(self.configs)

    def hub(self, name: str) -> FrameHub | OpenCVJpegHub:
        if name not in self.hubs:
            raise KeyError(name)
        return self.hubs[name]

    def status(self, name: str) -> dict:
        cfg = self.configs[name]
        hub = self.hubs[name]
        return {
            "name": name,
            "device": cfg.device,
            "backend": cfg.backend,
            "width": cfg.width,
            "height": cfg.height,
            "frames": hub.frame_id,
            "uptime": int(time.time() - hub.started_at),
            "error": hub.error or "",
        }

    def all_status(self) -> dict[str, dict]:
        return {name: self.status(name) for name in self.names()}

    def close(self) -> None:
        for hub in self.hubs.values():
            hub.close()


class CalibrationStore:
    def __init__(self, path: str) -> None:
        self.path = path
        self.lock = threading.Lock()

    def load(self) -> dict:
        with self.lock:
            if not os.path.exists(self.path):
                return {"exists": False}
            with open(self.path, "r", encoding="utf-8") as file:
                data = json.load(file)
            data["exists"] = True
            return data

    def save(self, data: dict) -> dict:
        points = data.get("points")
        width = int(data.get("image_width") or 0)
        height = int(data.get("image_height") or 0)
        if width <= 0 or height <= 0:
            raise ValueError("image_width and image_height are required")
        if not isinstance(points, list) or len(points) != 4:
            raise ValueError("exactly 4 points are required")

        clean_points = []
        for point in points:
            clean_points.append({"x": float(point.get("x")), "y": float(point.get("y"))})

        saved = {
            "camera": str(data.get("camera") or "overhead"),
            "image_width": width,
            "image_height": height,
            "points": clean_points,
            "normalized_points": [
                {"x": point["x"] / width, "y": point["y"] / height}
                for point in clean_points
            ],
            "updated_at": int(time.time()),
        }
        with self.lock:
            directory = os.path.dirname(self.path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            tmp_path = self.path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as file:
                json.dump(saved, file, ensure_ascii=False, indent=2)
                file.write("\n")
            os.replace(tmp_path, self.path)
        result = dict(saved)
        result["exists"] = True
        return result

    def delete(self) -> None:
        with self.lock:
            if os.path.exists(self.path):
                os.remove(self.path)


def make_handler(manager: MultiCameraManager, calibration_store: CalibrationStore):
    class Handler(BaseHTTPRequestHandler):
        server_version = "DIYRobotThreeCameraWebUI/1.0"

        def do_GET(self):
            parsed = urlparse(self.path)
            path = parsed.path.rstrip("/") or "/"
            if path in ("/", "/index.html"):
                self._index()
                return
            if path == "/api/status":
                self._json(manager.all_status())
                return

            parts = [part for part in path.split("/") if part]
            if len(parts) == 1 and parts[0] in manager.configs:
                query = parse_qs(parsed.query)
                self._camera_page(parts[0], calibrate=query.get("calibrate", ["0"])[0] == "1")
                return
            if len(parts) == 2 and parts[0] in manager.configs and parts[1] == "stream.mjpg":
                self._stream(parts[0])
                return
            if len(parts) == 2 and parts[0] in manager.configs and parts[1] == "status":
                self._json(manager.status(parts[0]))
                return
            if len(parts) == 2 and parts[0] == "overhead" and parts[1] == "calibration":
                self._json(calibration_store.load())
                return
            self.send_error(404)

        def do_POST(self):
            parsed = urlparse(self.path)
            path = parsed.path.rstrip("/") or "/"
            parts = [part for part in path.split("/") if part]
            if len(parts) == 2 and parts[0] == "overhead" and parts[1] == "calibration":
                try:
                    length = int(self.headers.get("Content-Length") or "0")
                    body = self.rfile.read(length)
                    payload = json.loads(body.decode("utf-8"))
                    self._json(calibration_store.save(payload))
                except Exception as exc:
                    self._json({"ok": False, "error": str(exc)}, code=400)
                return
            if len(parts) == 3 and parts[0] == "overhead" and parts[1] == "calibration" and parts[2] == "delete":
                calibration_store.delete()
                self._json({"ok": True, "exists": False})
                return
            self.send_error(404)

        def _index(self):
            cards = "\n".join(
                f"""
                <a class="card" href="/{quote(name)}">
                  <img src="/{quote(name)}/stream.mjpg" alt="{escape(name)} live stream">
                  <div class="label">
                    <strong>{escape(name)}</strong>
                    <span id="status-{escape(name)}">connecting</span>
                  </div>
                </a>
                """
                for name in manager.names()
            )
            html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>DIYRobot Cameras</title>
  <style>
    :root {{ color-scheme: dark; font-family: ui-sans-serif, system-ui, sans-serif; }}
    body {{ margin:0; min-height:100vh; background:#111815; color:#f3f5ef; }}
    header {{ padding:18px 22px; border-bottom:1px solid #2b3933; background:linear-gradient(135deg,#17221d,#0f1513); }}
    h1 {{ margin:0; font-size:22px; }}
    p {{ margin:6px 0 0; color:#aab8af; }}
    main {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:18px; padding:18px; }}
    .card {{ color:inherit; text-decoration:none; border:1px solid #2b3933; border-radius:14px; overflow:hidden; background:#17201c; box-shadow:0 16px 50px #0006; }}
    img {{ display:block; width:100%; aspect-ratio:4/3; object-fit:contain; background:#050706; }}
    .label {{ display:flex; justify-content:space-between; gap:12px; padding:12px 14px; color:#cbd6cf; }}
    .label strong {{ color:#fff; }}
  </style>
</head>
<body>
  <header>
    <h1>DIYRobot three-camera preview</h1>
    <p>Select any view for a single-camera page. This preview does not control motors.</p>
  </header>
  <main>{cards}</main>
  <script>
    async function refreshStatus() {{
      try {{
        const res = await fetch('/api/status', {{cache:'no-store'}});
        const data = await res.json();
        for (const [name, item] of Object.entries(data)) {{
          const el = document.getElementById('status-' + name);
          if (el) el.textContent = `${{item.width}}x${{item.height}}  frames ${{item.frames}}${{item.error ? '  ' + item.error : ''}}`;
        }}
      }} catch (err) {{}}
    }}
    setInterval(refreshStatus, 1500);
    refreshStatus();
  </script>
</body>
</html>"""
            self._html(html)

        def _camera_page(self, name: str, calibrate: bool = False):
            safe_name = escape(name)
            if name == "overhead" and calibrate:
                self._overhead_calibration_page()
                return
            html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>{safe_name} | DIYRobot Camera</title>
  <style>
    :root {{ color-scheme: dark; font-family: ui-sans-serif, system-ui, sans-serif; }}
    body {{ margin:0; min-height:100vh; background:#080b0a; color:#f5f7f2; display:grid; grid-template-rows:auto 1fr; }}
    header {{ display:flex; align-items:center; justify-content:space-between; gap:12px; padding:12px 16px; background:#121916; border-bottom:1px solid #2b3933; }}
    a {{ color:#9ed8b3; text-decoration:none; }}
    h1 {{ margin:0; font-size:18px; }}
    .status {{ color:#aab8af; font-size:14px; }}
    main {{ min-height:0; display:grid; place-items:center; padding:14px; }}
    img {{ max-width:100%; max-height:calc(100vh - 78px); background:#000; border:1px solid #2b3933; }}
  </style>
</head>
<body>
  <header>
    <div><a href="/">&larr; All cameras</a><h1>{safe_name}</h1></div>
    <div class="status" id="status">connecting</div>
  </header>
  <main><img src="/{quote(name)}/stream.mjpg" alt="{safe_name} live stream"></main>
  <script>
    async function refreshStatus() {{
      try {{
        const res = await fetch('/{quote(name)}/status', {{cache:'no-store'}});
        const item = await res.json();
        document.getElementById('status').textContent =
          `${{item.device}}  ${{item.width}}x${{item.height}}  frames ${{item.frames}}${{item.error ? '  ' + item.error : ''}}`;
      }} catch (err) {{
        document.getElementById('status').textContent = err.message;
      }}
    }}
    setInterval(refreshStatus, 1500);
    refreshStatus();
  </script>
</body>
</html>"""
            self._html(html)

        def _overhead_calibration_page(self):
            html = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Overhead reference frame | DIYRobot Camera</title>
  <style>
    :root { color-scheme: dark; font-family: ui-sans-serif, system-ui, sans-serif; }
    body { margin:0; min-height:100vh; background:#080b0a; color:#f5f7f2; display:grid; grid-template-rows:auto 1fr; }
    header { display:flex; align-items:flex-start; justify-content:space-between; gap:12px; padding:12px 16px; background:#121916; border-bottom:1px solid #2b3933; }
    a { color:#9ed8b3; text-decoration:none; }
    h1 { margin:0; font-size:18px; }
    .status { color:#aab8af; font-size:14px; text-align:right; }
    .tools { display:flex; align-items:center; gap:8px; flex-wrap:wrap; margin-top:8px; }
    .group { display:flex; align-items:center; gap:4px; padding:4px; border:1px solid #26372f; border-radius:6px; }
    button { height:32px; border:1px solid #395047; border-radius:6px; background:#17231e; color:#e7f3eb; padding:0 10px; cursor:pointer; }
    .small { width:36px; padding:0; font-weight:700; }
    main { min-height:0; display:grid; place-items:center; padding:14px; }
    .stage { position:relative; display:inline-block; max-width:100%; max-height:calc(100vh - 106px); }
    img { display:block; max-width:100%; max-height:calc(100vh - 106px); background:#000; border:1px solid #2b3933; box-sizing:border-box; }
    canvas { position:absolute; left:0; top:0; width:100%; height:100%; pointer-events:none; }
    .hint { color:#d7e0da; font-size:13px; }
  </style>
</head>
<body>
  <header>
    <div>
      <a href="/overhead">Back to clean overhead</a>
      <h1>overhead zero frame</h1>
      <div class="tools">
        <div class="group">
          <button class="small" data-move="up" type="button">UP</button>
          <button class="small" data-move="down" type="button">DN</button>
          <button class="small" data-move="left" type="button">LT</button>
          <button class="small" data-move="right" type="button">RT</button>
        </div>
        <div class="group">
          <button class="small" data-size="w-" type="button">W-</button>
          <button class="small" data-size="w+" type="button">W+</button>
          <button class="small" data-size="h-" type="button">H-</button>
          <button class="small" data-size="h+" type="button">H+</button>
        </div>
        <button id="saveBtn" type="button">Save frame</button>
        <button id="resetBtn" type="button">Reset default</button>
        <button id="deleteBtn" type="button">Delete saved</button>
        <span class="hint" id="hint">Red frame is browser-only. Use camera mount to align the mat to it.</span>
      </div>
    </div>
    <div class="status" id="status">connecting</div>
  </header>
  <main>
    <div class="stage" id="stage">
      <img id="stream" src="/overhead/stream.mjpg" alt="overhead live stream">
      <canvas id="overlay"></canvas>
    </div>
  </main>
  <script>
    const stream = document.getElementById('stream');
    const overlay = document.getElementById('overlay');
    const statusEl = document.getElementById('status');
    const hint = document.getElementById('hint');
    const saveBtn = document.getElementById('saveBtn');
    const resetBtn = document.getElementById('resetBtn');
    const deleteBtn = document.getElementById('deleteBtn');
    let imageWidth = 640;
    let imageHeight = 480;
    const defaultRect = {x:128, y:151, w:330, h:265};
    let rect = {...defaultRect};

    function resizeCanvas() {
      const rect = stream.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      overlay.width = Math.max(1, Math.round(rect.width * dpr));
      overlay.height = Math.max(1, Math.round(rect.height * dpr));
      overlay.style.width = rect.width + 'px';
      overlay.style.height = rect.height + 'px';
      draw();
    }

    function scaleRect() {
      const view = stream.getBoundingClientRect();
      return {
        x: rect.x / imageWidth * view.width,
        y: rect.y / imageHeight * view.height,
        w: rect.w / imageWidth * view.width,
        h: rect.h / imageHeight * view.height,
      };
    }

    function draw() {
      const ctx = overlay.getContext('2d');
      ctx.clearRect(0, 0, overlay.width, overlay.height);
      const dpr = window.devicePixelRatio || 1;
      const r = scaleRect();
      ctx.save();
      ctx.scale(dpr, dpr);
      ctx.lineWidth = 4;
      ctx.strokeStyle = '#ff2424';
      ctx.fillStyle = '#ff2424';
      ctx.font = '13px ui-sans-serif, system-ui, sans-serif';
      ctx.strokeRect(r.x, r.y, r.w, r.h);
      ctx.fillText(`x=${Math.round(rect.x)} y=${Math.round(rect.y)} w=${Math.round(rect.w)} h=${Math.round(rect.h)}`, r.x + 8, Math.max(18, r.y - 8));
      ctx.restore();
      hint.textContent = `Frame: x=${Math.round(rect.x)}, y=${Math.round(rect.y)}, w=${Math.round(rect.w)}, h=${Math.round(rect.h)}. Align the mat to this red frame.`;
    }

    function rectToPoints() {
      return [
        {x:rect.x, y:rect.y},
        {x:rect.x + rect.w, y:rect.y},
        {x:rect.x + rect.w, y:rect.y + rect.h},
        {x:rect.x, y:rect.y + rect.h},
      ];
    }

    function pointsToRect(points) {
      const xs = points.map(p => p.x);
      const ys = points.map(p => p.y);
      const minX = Math.min(...xs);
      const maxX = Math.max(...xs);
      const minY = Math.min(...ys);
      const maxY = Math.max(...ys);
      return {x:minX, y:minY, w:maxX - minX, h:maxY - minY};
    }

    function clampRect() {
      rect.w = Math.max(20, Math.min(imageWidth, rect.w));
      rect.h = Math.max(20, Math.min(imageHeight, rect.h));
      rect.x = Math.max(0, Math.min(imageWidth - rect.w, rect.x));
      rect.y = Math.max(0, Math.min(imageHeight - rect.h, rect.y));
    }

    async function loadCalibration() {
      const res = await fetch('/overhead/calibration', {cache:'no-store'});
      const data = await res.json();
      if (data.exists) {
        imageWidth = data.image_width || imageWidth;
        imageHeight = data.image_height || imageHeight;
        let points = [];
        if (Array.isArray(data.normalized_points)) {
          points = data.normalized_points.map(p => ({x:p.x * imageWidth, y:p.y * imageHeight}));
        } else {
          points = data.points || [];
        }
        if (points.length === 4) rect = pointsToRect(points);
      }
      clampRect();
      draw();
    }

    async function refreshStatus() {
      try {
        const res = await fetch('/overhead/status', {cache:'no-store'});
        const item = await res.json();
        imageWidth = item.width || imageWidth;
        imageHeight = item.height || imageHeight;
        statusEl.textContent = `${item.device}  ${item.width}x${item.height}  frames ${item.frames}${item.error ? '  ' + item.error : ''}`;
      } catch (err) {
        statusEl.textContent = err.message;
      }
      resizeCanvas();
    }

    saveBtn.addEventListener('click', async () => {
      clampRect();
      const res = await fetch('/overhead/calibration', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify({
          camera:'overhead',
          image_width:imageWidth,
          image_height:imageHeight,
          points:rectToPoints(),
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'save failed');
      if (Array.isArray(data.points)) rect = pointsToRect(data.points);
      draw();
    });
    resetBtn.addEventListener('click', () => {
      rect = {...defaultRect};
      clampRect();
      draw();
    });
    deleteBtn.addEventListener('click', async () => {
      await fetch('/overhead/calibration/delete', {method:'POST'});
      rect = {...defaultRect};
      clampRect();
      draw();
    });
    document.querySelectorAll('[data-move]').forEach(btn => btn.addEventListener('click', () => {
      const step = 5;
      if (btn.dataset.move === 'up') rect.y -= step;
      if (btn.dataset.move === 'down') rect.y += step;
      if (btn.dataset.move === 'left') rect.x -= step;
      if (btn.dataset.move === 'right') rect.x += step;
      clampRect();
      draw();
    }));
    document.querySelectorAll('[data-size]').forEach(btn => btn.addEventListener('click', () => {
      const step = 5;
      if (btn.dataset.size === 'w-') { rect.x += step / 2; rect.w -= step; }
      if (btn.dataset.size === 'w+') { rect.x -= step / 2; rect.w += step; }
      if (btn.dataset.size === 'h-') { rect.y += step / 2; rect.h -= step; }
      if (btn.dataset.size === 'h+') { rect.y -= step / 2; rect.h += step; }
      clampRect();
      draw();
    }));
    stream.addEventListener('load', resizeCanvas);
    window.addEventListener('resize', resizeCanvas);
    setInterval(refreshStatus, 1500);
    refreshStatus();
    loadCalibration();
  </script>
</body>
</html>"""
            self._html(html)

        def _stream(self, name: str):
            hub = manager.hub(name)
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

        def _html(self, html: str):
            body = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
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

        def log_message(self, fmt, *args):
            print("%s - %s" % (self.address_string(), fmt % args), flush=True)

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument(
        "--camera",
        action="append",
        default=[],
        help="Camera spec NAME=DEVICE:WIDTHxHEIGHT. Repeat for multiple cameras.",
    )
    parser.add_argument(
        "--calibration-file",
        default="overhead_calibration.json",
        help="Path for the overhead calibration reference points JSON.",
    )
    args = parser.parse_args()

    configs = [parse_camera(item) for item in (args.camera or DEFAULT_CAMERAS)]
    manager = MultiCameraManager(configs)
    calibration_store = CalibrationStore(args.calibration_file)
    manager.start()

    class Server(ThreadingHTTPServer):
        daemon_threads = True
        allow_reuse_address = True

    httpd = Server((args.host, args.port), make_handler(manager, calibration_store))

    def shutdown(_signum, _frame):
        manager.close()
        httpd.shutdown()

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    print(f"serving DIYRobot cameras at http://{args.host}:{args.port}/", flush=True)
    for cfg in configs:
        print(f"  /{cfg.name} -> {cfg.device} ({cfg.width}x{cfg.height})", flush=True)
    httpd.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
