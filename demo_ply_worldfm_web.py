#!/usr/bin/env python3
"""
Browser-based WorldFM interaction from an existing PLY scene.

This is the headless/server variant of demo_ply_worldfm.py. It does not create
an OpenCV desktop window. A tiny HTTP server serves a browser UI; the browser
sends camera controls, and the Python process returns the WorldFM-generated
image for the updated camera pose.
"""

from __future__ import annotations

import base64
import json
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from demo_ply_worldfm import (
    InteractiveWorldFMApp,
    TorchPointCloudRenderer,
    WorldFMInprocessConfig,
    WorldFMTriConditionInprocess,
    _build_condition_db_from_source_frames,
    _decode_worldfm_output,
    _default_model_path,
    _load_points,
    _load_source_cameras,
    _make_debug_panel,
    _make_input_output_panel,
    _move_camera,
    _nearest_source_index,
    _resolve,
    _rotate_camera_local,
    build_parser as build_base_parser,
)
from modules.transforms_io import scale_K_for_resize


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>WorldFM PLY Web</title>
  <style>
    :root { color-scheme: dark; font-family: system-ui, -apple-system, Segoe UI, sans-serif; }
    body { margin: 0; background: #101114; color: #eceff4; overflow: hidden; }
    #wrap { display: grid; grid-template-columns: 1fr 310px; height: 100vh; }
    #stage { position: relative; display: grid; place-items: center; background: #08090b; }
    #img { max-width: 100%; max-height: 100%; object-fit: contain; image-rendering: auto; }
    #busy { position: absolute; top: 14px; left: 14px; padding: 6px 10px; background: rgba(0,0,0,.65); border-radius: 6px; }
    #side { border-left: 1px solid #2a2d34; padding: 14px; background: #17191f; overflow: auto; }
    button, input { font: inherit; }
    button { width: 100%; margin: 4px 0; padding: 8px 10px; border: 1px solid #3a3f4a; border-radius: 6px; background: #242832; color: #fff; cursor: pointer; }
    button:hover { background: #303642; }
    .grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 4px; margin: 8px 0 12px; }
    .grid button { margin: 0; }
    label { display: block; margin-top: 10px; color: #aeb6c6; font-size: 13px; }
    input { width: 100%; box-sizing: border-box; margin-top: 4px; padding: 6px; border-radius: 6px; border: 1px solid #3a3f4a; background: #0f1117; color: #fff; }
    #meta { margin-top: 12px; white-space: pre-wrap; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; color: #c5ccda; }
    @media (max-width: 900px) { #wrap { grid-template-columns: 1fr; grid-template-rows: 1fr auto; } #side { max-height: 42vh; border-left: 0; border-top: 1px solid #2a2d34; } }
  </style>
</head>
<body>
  <div id="wrap">
    <main id="stage">
      <img id="img" draggable="false" alt="WorldFM splat input and output">
      <div id="busy">loading</div>
    </main>
    <aside id="side">
      <div class="grid">
        <button data-key="q">Q down</button><button data-key="w">W forward</button><button data-key="e">E up</button>
        <button data-key="a">A left</button><button data-key="s">S back</button><button data-key="d">D right</button>
        <button data-key="i">I pitch</button><button data-key="j">J yaw</button><button data-key="l">L yaw</button>
      </div>
      <button data-action="snap">Snap nearest source pose</button>
      <button data-action="reset">Reset start pose</button>
      <button data-action="save">Save current frame</button>
      <label>Move step <input id="move" type="number" min="0.001" step="0.01" value=""></label>
      <label>Turn step <input id="turn" type="number" min="0.001" step="0.01" value=""></label>
      <label>Mouse sensitivity <input id="sens" type="number" min="0.0001" step="0.0005" value=""></label>
      <div id="meta"></div>
    </aside>
  </div>
<script>
const img = document.getElementById('img');
const busy = document.getElementById('busy');
const meta = document.getElementById('meta');
const moveInput = document.getElementById('move');
const turnInput = document.getElementById('turn');
const sensInput = document.getElementById('sens');
let inFlight = false;
let queued = null;
let dragging = false;
let lastX = 0, lastY = 0;

function settings() {
  const out = {};
  if (moveInput.value !== '') out.move_step = Number(moveInput.value);
  if (turnInput.value !== '') out.key_turn_step = Number(turnInput.value);
  if (sensInput.value !== '') out.mouse_sensitivity = Number(sensInput.value);
  return out;
}

async function render(action) {
  if (inFlight) {
    if (queued && queued.type === 'drag' && action.type === 'drag') {
      queued.dx += action.dx;
      queued.dy += action.dy;
    } else if (queued && queued.type === 'key' && action.type === 'key' && queued.key === action.key) {
      queued.count = (queued.count || 1) + 1;
    } else {
      queued = action;
    }
    return;
  }
  inFlight = true;
  busy.textContent = 'generating';
  try {
    const res = await fetch('/api/render', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({action, settings: settings()})
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || res.statusText);
    img.src = data.image;
    meta.textContent = JSON.stringify(data.meta, null, 2);
    if (!moveInput.value) moveInput.value = data.meta.move_step;
    if (!turnInput.value) turnInput.value = data.meta.key_turn_step;
    if (!sensInput.value) sensInput.value = data.meta.mouse_sensitivity;
    busy.textContent = `ready ${data.meta.elapsed_ms.toFixed(0)} ms`;
  } catch (err) {
    busy.textContent = 'error';
    meta.textContent = String(err);
  } finally {
    inFlight = false;
    if (queued) {
      const next = queued;
      queued = null;
      render(next);
    }
  }
}

document.querySelectorAll('[data-key]').forEach(btn => {
  btn.addEventListener('click', () => render({type: 'key', key: btn.dataset.key}));
});
document.querySelectorAll('[data-action]').forEach(btn => {
  btn.addEventListener('click', () => render({type: btn.dataset.action}));
});

window.addEventListener('keydown', ev => {
  const k = ev.key.toLowerCase();
  if ('wasdqerijkl'.includes(k)) {
    ev.preventDefault();
    render({type: 'key', key: k});
  } else if (k === 'p') {
    render({type: 'save'});
  }
});

img.addEventListener('pointerdown', ev => {
  dragging = true;
  lastX = ev.clientX;
  lastY = ev.clientY;
  img.setPointerCapture(ev.pointerId);
});
img.addEventListener('pointermove', ev => {
  if (!dragging) return;
  const dx = ev.clientX - lastX;
  const dy = ev.clientY - lastY;
  lastX = ev.clientX;
  lastY = ev.clientY;
  if (Math.abs(dx) + Math.abs(dy) >= 4) render({type: 'drag', dx, dy});
});
img.addEventListener('pointerup', ev => {
  dragging = false;
  img.releasePointerCapture(ev.pointerId);
});

render({type: 'none'});
</script>
</body>
</html>
"""


class WebWorldFMEngine:
    def __init__(self, args) -> None:
        self.args = args
        if torch.cuda.is_available() and int(args.gpu_index) >= 0:
            torch.cuda.set_device(int(args.gpu_index))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_str = f"cuda:{torch.cuda.current_device()}" if self.device.type == "cuda" else "cpu"

        scene_dir = _resolve(args.scene_dir)
        ply_path = _resolve(args.ply) if args.ply else scene_dir / "points" / "point_cloud.ply"
        image_dir = _resolve(args.image_dir) if args.image_dir else scene_dir / "images"
        default_orig = scene_dir / "runtime" / "extracted_frames"
        original_frame_dir = _resolve(args.original_frame_dir) if args.original_frame_dir else (
            default_orig if default_orig.exists() else None
        )
        intri_yml = _resolve(args.intri_yml) if args.intri_yml else scene_dir / "intri.yml"
        extri_yml = _resolve(args.extri_yml) if args.extri_yml else scene_dir / "extri.yml"
        self.output_dir = _resolve(args.output_dir)

        self.cameras = _load_source_cameras(
            scene_dir=scene_dir,
            image_dir=image_dir,
            original_frame_dir=original_frame_dir,
            intri_yml=intri_yml,
            extri_yml=extri_yml,
            frame_stride=int(args.frame_stride),
            max_frames=int(args.max_frames),
        )

        xyz, rgb = _load_points(ply_path)
        self.renderer = TorchPointCloudRenderer(
            points_xyz=xyz,
            points_rgb=rgb,
            width=int(args.render_size),
            height=int(args.render_size),
            device=self.device_str,
            mode="fast",
            max_points=(int(args.max_points) if int(args.max_points) > 0 else None),
            splat_radius=int(args.splat_radius),
        )
        self.cond_db = _build_condition_db_from_source_frames(
            cameras=self.cameras,
            renderer=self.renderer,
            render_size=int(args.render_size),
            device=self.device,
        )

        model_path = _resolve(args.model_path) if args.model_path else _default_model_path(int(args.step)).resolve()
        vae_path = _resolve(args.vae_path)
        self.svc = WorldFMTriConditionInprocess(
            WorldFMInprocessConfig(
                model_path=str(model_path),
                vae_path=str(vae_path),
                image_size=int(args.image_size),
                version=str(args.version),
                disable_cross_attn=True,
                step=(int(args.step) if int(args.step) in (1, 2) else 2),
                cfg_scale=float(args.cfg_scale),
                device=self.device_str,
                weight_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                profile=bool(args.profile),
            )
        )
        self.svc.set_cond2_candidates_from_paths([str(p) for p in self.cameras.image_paths])

        self.start = int(np.clip(int(args.start_frame), 0, len(self.cameras.c2w) - 1))
        self.K_render = scale_K_for_resize(
            self.cameras.K[self.start],
            src_wh=self.cameras.source_wh,
            dst_wh=(int(args.render_size), int(args.render_size)),
        )
        self.c2w_start = self.cameras.c2w[self.start].copy()
        self.c2w = self.c2w_start.copy()
        self.last_generated: np.ndarray | None = None
        self.last_cond_idx = self.start
        self.frame_id = 0
        self.lock = threading.Lock()

        print(
            f"[PLY-WorldFM-Web] ready: {len(self.cameras.names)} cameras, "
            f"{self.renderer.num_points_used:,}/{self.renderer.num_points_total:,} points, "
            f"splat_radius={int(args.splat_radius)}, {self.device_str}",
            flush=True,
        )

    def _select_condition(self, depth: torch.Tensor) -> tuple[int, int, int]:
        return InteractiveWorldFMApp._select_condition(self, depth)

    def _apply_settings(self, settings: dict[str, Any]) -> None:
        for key in ("move_step", "key_turn_step", "mouse_sensitivity"):
            if key in settings and settings[key] not in (None, ""):
                value = float(settings[key])
                if value > 0.0:
                    setattr(self.args, key, value)

    def _apply_action(self, action: dict[str, Any]) -> str:
        typ = str(action.get("type", "none"))
        move = float(self.args.move_step)
        rot = float(self.args.key_turn_step)

        if typ == "key":
            key = str(action.get("key", "")).lower()
            count = max(1, min(20, int(action.get("count", 1) or 1)))
            move *= count
            rot *= count
            if key == "w":
                self.c2w = _move_camera(self.c2w, np.array([0.0, 0.0, move]))
            elif key == "s":
                self.c2w = _move_camera(self.c2w, np.array([0.0, 0.0, -move]))
            elif key == "a":
                self.c2w = _move_camera(self.c2w, np.array([-move, 0.0, 0.0]))
            elif key == "d":
                self.c2w = _move_camera(self.c2w, np.array([move, 0.0, 0.0]))
            elif key == "q":
                self.c2w = _move_camera(self.c2w, np.array([0.0, -move, 0.0]))
            elif key == "e":
                self.c2w = _move_camera(self.c2w, np.array([0.0, move, 0.0]))
            elif key == "j":
                self.c2w = _rotate_camera_local(self.c2w, yaw=-rot, pitch=0.0)
            elif key == "l":
                self.c2w = _rotate_camera_local(self.c2w, yaw=rot, pitch=0.0)
            elif key == "i":
                self.c2w = _rotate_camera_local(self.c2w, yaw=0.0, pitch=-rot)
            elif key == "k":
                self.c2w = _rotate_camera_local(self.c2w, yaw=0.0, pitch=rot)
            elif key == "r":
                idx = _nearest_source_index(self.cameras, self.c2w)
                self.c2w = self.cameras.c2w[idx].copy()
            return f"key:{key}"
        if typ == "drag":
            sens = float(self.args.mouse_sensitivity)
            self.c2w = _rotate_camera_local(
                self.c2w,
                yaw=float(action.get("dx", 0.0)) * sens,
                pitch=float(action.get("dy", 0.0)) * sens,
            )
            return "drag"
        if typ == "snap":
            idx = _nearest_source_index(self.cameras, self.c2w)
            self.c2w = self.cameras.c2w[idx].copy()
            return "snap"
        if typ == "reset":
            self.c2w = self.c2w_start.copy()
            return "reset"
        if typ == "save":
            self._save_current()
            return "save"
        return "none"

    def _save_current(self) -> Path | None:
        if self.last_generated is None:
            return None
        self.output_dir.mkdir(parents=True, exist_ok=True)
        out = self.output_dir / f"web_{self.frame_id:06d}.png"
        Image.fromarray(self.last_generated).save(out)
        meta = {
            "frame_id": self.frame_id,
            "cond_name": self.cameras.names[self.last_cond_idx],
            "cond_image": str(self.cameras.image_paths[self.last_cond_idx]),
            "K": self.K_render.tolist(),
            "c2w": self.c2w.tolist(),
        }
        out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return out

    def _generate_locked(self) -> dict[str, Any]:
        t0 = time.perf_counter()
        render = self.renderer.render_torch(K_3x3=self.K_render, c2w_4x4=self.c2w)
        idx, hits, samples = self._select_condition(render.depth_f32)
        if samples == 0:
            idx = _nearest_source_index(self.cameras, self.c2w)

        if int(self.args.step) in (1, 2):
            decoded = self.svc.infer_from_render_u8(
                render.rgb_u8,
                cond2_index=idx,
                profile=bool(self.args.profile),
                profile_tag=str(self.frame_id),
            )
        else:
            decoded = self.svc.infer_from_render_u8_multistep(
                render.rgb_u8,
                sample_steps=int(self.args.step),
                cfg_scale=float(self.args.cfg_scale),
                cond2_index=idx,
            )

        gen = _decode_worldfm_output(decoded)
        self.last_generated = gen
        self.last_cond_idx = int(idx)
        self.frame_id += 1

        raw_render = render.rgb_u8.detach().cpu().numpy()
        display = _make_input_output_panel(raw_render, gen)
        if bool(self.args.debug_panel):
            cond = np.array(Image.open(self.cameras.image_paths[idx]).convert("RGB"))
            display = _make_debug_panel(raw_render, cond, gen)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        ok, enc = cv2.imencode(
            ".jpg",
            cv2.cvtColor(display, cv2.COLOR_RGB2BGR),
            [int(cv2.IMWRITE_JPEG_QUALITY), int(self.args.jpeg_quality)],
        )
        if not ok:
            raise RuntimeError("Failed to JPEG-encode generated frame")

        return {
            "image": "data:image/jpeg;base64," + base64.b64encode(enc.tobytes()).decode("ascii"),
            "meta": {
                "frame_id": self.frame_id,
                "elapsed_ms": elapsed_ms,
                "cond_index": int(idx),
                "cond_name": self.cameras.names[idx],
                "display": "splat_input|worldfm_output" if not bool(self.args.debug_panel) else "splat_input|source_frame|worldfm_output",
                "splat_radius": int(self.args.splat_radius),
                "hits": int(hits),
                "samples": int(samples),
                "position": [float(x) for x in self.c2w[:3, 3]],
                "move_step": float(self.args.move_step),
                "key_turn_step": float(self.args.key_turn_step),
                "mouse_sensitivity": float(self.args.mouse_sensitivity),
            },
        }

    def render(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            self._apply_settings(payload.get("settings") or {})
            action_name = self._apply_action(payload.get("action") or {})
            out = self._generate_locked()
            out["meta"]["action"] = action_name
            return out


def build_parser():
    parser = build_base_parser()
    parser.description = "Headless browser demo for WorldFM from existing PLY + source frames."
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--jpeg_quality", type=int, default=90)
    return parser


def make_handler(engine: WebWorldFMEngine):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            print(f"[PLY-WorldFM-Web] {self.address_string()} - {fmt % args}", flush=True)

        def _send(self, status: int, data: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _send_json(self, status: int, obj: dict[str, Any]) -> None:
            self._send(status, json.dumps(obj).encode("utf-8"), "application/json")

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(HTTPStatus.OK, INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
            else:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self):
            if self.path != "/api/render":
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            try:
                n = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(n).decode("utf-8") or "{}")
                self._send_json(HTTPStatus.OK, engine.render(payload))
            except Exception as exc:
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

    return Handler


def main() -> int:
    args = build_parser().parse_args()
    engine = WebWorldFMEngine(args)
    server = ThreadingHTTPServer((str(args.host), int(args.port)), make_handler(engine))
    url = f"http://{args.host}:{args.port}/"
    print(f"[PLY-WorldFM-Web] serving {url}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[PLY-WorldFM-Web] stopped", flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
