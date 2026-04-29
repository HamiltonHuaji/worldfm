#!/usr/bin/env python3
"""
Interactive WorldFM demo from an existing point cloud and source images.

This path intentionally skips HunyuanWorld and MoGe. It assumes geometry has
already been reconstructed, for example by LingBot-MAP, and uses:

  * point_cloud.ply as the geometric proxy
  * source video frames as WorldFM cond2 candidates
  * intri.yml/extri.yml as OpenCV camera intrinsics and c2w poses

The displayed frame is the WorldFM-generated image for the current interactive
camera, not the raw point-cloud render.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
import torch
from PIL import Image

from modules.depth_selector import ConditionDB, select_best_condition_index
from modules.ply_io import load_ply_xyz_rgb
from modules.point_renderer import TorchPointCloudRenderer
from modules.transforms_io import scale_K_for_resize
from modules.worldfm_infer import WorldFMInprocessConfig, WorldFMTriConditionInprocess


WORLDFM_ROOT = Path(__file__).resolve().parent
DEFAULT_SCENE_DIR = WORLDFM_ROOT / "../../outputs/ict_5floor/lingbot-map"


@dataclass(frozen=True)
class SourceCameraSet:
    names: list[str]
    image_paths: list[Path]
    K: list[np.ndarray]
    c2w: list[np.ndarray]
    source_wh: tuple[int, int]


def _log(msg: str) -> None:
    print(f"[PLY-WorldFM] {msg}", flush=True)


def _as_4x4(mat: np.ndarray) -> np.ndarray:
    m = np.asarray(mat, dtype=np.float64)
    if m.shape == (4, 4):
        return m.copy()
    if m.shape == (3, 4):
        out = np.eye(4, dtype=np.float64)
        out[:3, :4] = m
        return out
    raise ValueError(f"Expected (3,4) or (4,4), got {m.shape}")


def _resolve(path: str | Path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else (Path.cwd() / p).resolve()


def _default_model_path(step: int) -> Path:
    ckpt_step = 1 if int(step) == 1 else 2
    return Path(f"worldfm_{ckpt_step}-step.pth")


def _model_ref_from_arg(value: str, step: int) -> str:
    if not value:
        return str(_default_model_path(step))
    local = Path(value).expanduser()
    if local.exists():
        return str(_resolve(value))
    if "/" in value and ":" not in value and not value.startswith("hf://"):
        ckpt_step = 1 if int(step) == 1 else 2
        return f"{value}:worldfm_{ckpt_step}-step.pth"
    return value


def _vae_ref_from_arg(value: str) -> str:
    value = value or "vae"
    local = Path(value).expanduser()
    if local.exists():
        return str(_resolve(value))
    if "/" in value and ":" not in value and not value.startswith("hf://"):
        return f"{value}:vae"
    return value


def _parse_opencv_matrix_yml(path: Path, node_name: str) -> tuple[list[str], dict[str, np.ndarray]]:
    """Parse the compact OpenCV YAML emitted by LingBot export.

    cv2.FileStorage is strict about indentation, while the export format is easy
    to parse directly: names followed by a map of opencv-matrix entries.
    """
    text = path.read_text(encoding="utf-8")
    node_pos = text.find(f"{node_name}:")
    if node_pos < 0:
        raise ValueError(f"Missing YAML node {node_name!r}: {path}")

    names_text = text[:node_pos]
    names = re.findall(r'^\s*-\s*"([^"]+)"\s*$', names_text, flags=re.MULTILINE)
    if not names:
        names = re.findall(r"^\s*-\s*([A-Za-z0-9_.-]+)\s*$", names_text, flags=re.MULTILINE)

    body = text[node_pos:]
    pat = re.compile(
        r'^\s*"([^"]+)":\s*!!opencv-matrix\s*'
        r"^\s*rows:\s*(\d+)\s*"
        r"^\s*cols:\s*(\d+)\s*"
        r"^\s*dt:\s*\S+\s*"
        r"^\s*data:\s*\[([^\]]+)\]",
        flags=re.MULTILINE,
    )
    mats: dict[str, np.ndarray] = {}
    for key, rows_s, cols_s, data_s in pat.findall(body):
        rows, cols = int(rows_s), int(cols_s)
        vals = np.fromstring(data_s.replace("\n", " "), sep=",", dtype=np.float64)
        if vals.size != rows * cols:
            raise ValueError(f"Bad matrix data for {key!r} in {path}")
        mats[key] = vals.reshape(rows, cols)

    if not mats:
        raise ValueError(f"No matrices parsed from {path}")
    if not names:
        names = sorted(mats)
    return names, mats


def _load_source_cameras(
    *,
    scene_dir: Path,
    image_dir: Path,
    original_frame_dir: Path | None,
    intri_yml: Path,
    extri_yml: Path,
    frame_stride: int,
    max_frames: int,
) -> SourceCameraSet:
    names_i, intrinsics = _parse_opencv_matrix_yml(intri_yml, "intrinsics")
    names_e, extrinsics = _parse_opencv_matrix_yml(extri_yml, "extrinsics")
    ordered = [n for n in names_i if n in intrinsics and n in extrinsics]
    if not ordered:
        ordered = [n for n in names_e if n in intrinsics and n in extrinsics]
    if not ordered:
        raise RuntimeError("No shared camera names in intri/extri YAML files.")

    stride = max(1, int(frame_stride))
    ordered = ordered[::stride]
    if int(max_frames) > 0:
        ordered = ordered[:int(max_frames)]

    def _image_for_name(name: str) -> Path:
        search_dirs = []
        if original_frame_dir is not None:
            search_dirs.append(original_frame_dir)
        search_dirs.append(image_dir)
        for root in search_dirs:
            for ext in (".png", ".jpg", ".jpeg", ".webp"):
                p = root / f"{name}{ext}"
                if p.exists():
                    return p
        raise FileNotFoundError(f"No source frame found for camera {name!r}")

    image_paths = [_image_for_name(n) for n in ordered]
    if not image_paths:
        raise RuntimeError("No source images found.")

    with Image.open(image_dir / f"{ordered[0]}.png") if (image_dir / f"{ordered[0]}.png").exists() else Image.open(image_paths[0]) as im:
        source_wh = im.size

    K = [np.asarray(intrinsics[n], dtype=np.float64) for n in ordered]
    c2w = [_as_4x4(extrinsics[n]) for n in ordered]
    return SourceCameraSet(names=ordered, image_paths=image_paths, K=K, c2w=c2w, source_wh=source_wh)


def _load_points(ply_path: Path) -> tuple[np.ndarray, np.ndarray]:
    _log(f"Loading point cloud: {ply_path}")
    xyz, rgb = load_ply_xyz_rgb(str(ply_path))
    ok = np.isfinite(xyz).all(axis=1)
    xyz, rgb = xyz[ok], rgb[ok]
    _log(f"Loaded {xyz.shape[0]:,} finite points")
    return xyz, rgb


def _P_from_K_c2w(K: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    w2c = np.linalg.inv(_as_4x4(c2w)).astype(np.float32)
    return (np.asarray(K, dtype=np.float32) @ w2c[:3, :4]).astype(np.float32)


@torch.inference_mode()
def _build_condition_db_from_source_frames(
    *,
    cameras: SourceCameraSet,
    renderer: TorchPointCloudRenderer,
    render_size: int,
    device: torch.device,
) -> ConditionDB:
    P_list: list[torch.Tensor] = []
    D_list: list[torch.Tensor] = []
    C_list: list[torch.Tensor] = []
    dst_wh = (int(render_size), int(render_size))

    _log(f"Building source-frame condition DB: {len(cameras.names)} views")
    for idx, (K_src, c2w) in enumerate(zip(cameras.K, cameras.c2w)):
        K_use = scale_K_for_resize(K_src, src_wh=cameras.source_wh, dst_wh=dst_wh)
        P_list.append(torch.from_numpy(_P_from_K_c2w(K_use, c2w)).to(device=device))
        C_list.append(torch.from_numpy(c2w[:3, 3].astype(np.float32)).to(device=device))
        D_list.append(renderer.render_torch(K_3x3=K_use, c2w_4x4=c2w).depth_f32)
        if (idx + 1) % 25 == 0 or idx + 1 == len(cameras.names):
            _log(f"  cached depths {idx + 1}/{len(cameras.names)}")

    return ConditionDB(
        cond_paths=[str(p) for p in cameras.image_paths],
        P_views=torch.stack(P_list),
        depth_views=torch.stack(D_list),
        C_views=torch.stack(C_list),
        width=int(render_size),
        height=int(render_size),
    )


def _decode_worldfm_output(decoded: torch.Tensor) -> np.ndarray:
    return (
        torch.clamp(127.5 * decoded[0] + 128.0, 0, 255)
        .permute(1, 2, 0)
        .to(torch.uint8)
        .cpu()
        .numpy()
    )


def _rodrigues(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    x, y, z = axis
    c, s = math.cos(float(angle)), math.sin(float(angle))
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=np.float64,
    )


def _local_rotation(rx: float, ry: float, rz: float = 0.0) -> np.ndarray:
    Rx = _rodrigues(np.array([1.0, 0.0, 0.0]), rx)
    Ry = _rodrigues(np.array([0.0, 1.0, 0.0]), ry)
    Rz = _rodrigues(np.array([0.0, 0.0, 1.0]), rz)
    return Rz @ Ry @ Rx


def _move_camera(c2w: np.ndarray, delta_local: np.ndarray) -> np.ndarray:
    out = c2w.copy()
    out[:3, 3] += out[:3, :3] @ delta_local.astype(np.float64)
    return out


def _rotate_camera_local(c2w: np.ndarray, *, yaw: float, pitch: float, roll: float = 0.0) -> np.ndarray:
    out = c2w.copy()
    out[:3, :3] = out[:3, :3] @ _local_rotation(pitch, yaw, roll)
    return out


def _nearest_source_index(cameras: SourceCameraSet, c2w: np.ndarray) -> int:
    C = c2w[:3, 3]
    Cs = np.stack([p[:3, 3] for p in cameras.c2w], axis=0)
    return int(np.argmin(np.linalg.norm(Cs - C[None, :], axis=1)))


def _make_debug_panel(render_rgb: np.ndarray, cond_rgb: np.ndarray, gen_rgb: np.ndarray) -> np.ndarray:
    h, w = gen_rgb.shape[:2]
    render = np.array(Image.fromarray(render_rgb).resize((w, h), Image.BILINEAR))
    cond = np.array(Image.fromarray(cond_rgb).resize((w, h), Image.BILINEAR))
    return np.concatenate([render, cond, gen_rgb], axis=1)


def _make_input_output_panel(render_rgb: np.ndarray, gen_rgb: np.ndarray) -> np.ndarray:
    h, w = gen_rgb.shape[:2]
    render = np.array(Image.fromarray(render_rgb).resize((w, h), Image.BILINEAR))
    return np.concatenate([render, gen_rgb], axis=1)


class InteractiveWorldFMApp:
    def __init__(
        self,
        *,
        renderer: TorchPointCloudRenderer,
        cond_db: ConditionDB,
        cameras: SourceCameraSet,
        svc: WorldFMTriConditionInprocess,
        K_render: np.ndarray,
        c2w_init: np.ndarray,
        args: argparse.Namespace,
        output_dir: Path,
    ) -> None:
        self.renderer = renderer
        self.cond_db = cond_db
        self.cameras = cameras
        self.svc = svc
        self.K_render = K_render
        self.c2w = c2w_init.copy()
        self.args = args
        self.output_dir = output_dir
        self.window = str(args.window_name)
        self.dragging = False
        self.last_mouse: tuple[int, int] | None = None
        self.frame_id = 0
        self.last_generated: np.ndarray | None = None
        self.last_render: np.ndarray | None = None
        self.last_cond_idx = 0
        self.dirty = False

    def _select_condition(self, depth: torch.Tensor) -> tuple[int, int, int]:
        return select_best_condition_index(
            depth_cur=depth,
            K_cur=self.K_render,
            c2w_cur=self.c2w,
            cond_db=self.cond_db,
            sample_grid=int(self.args.sample_grid),
            center_grid=int(self.args.center_grid),
            center_frac=float(self.args.center_frac),
            eps_rel=float(self.args.eps_rel),
            eps_abs=float(self.args.eps_abs),
            px_radius=int(self.args.px_radius),
            max_view_angle_deg=float(self.args.max_view_angle_deg),
            use_distance_weight=bool(self.args.use_distance_weight),
            dist_min_m=float(self.args.dist_min_m),
            dist_max_m=float(self.args.dist_max_m),
            weight_near=float(self.args.weight_near),
            weight_far=float(self.args.weight_far),
        )

    def generate_current(self) -> np.ndarray:
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
        self.last_render = render.rgb_u8.detach().cpu().numpy()
        self.last_cond_idx = int(idx)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        hud = f"WorldFM {elapsed_ms:.0f} ms  cond={self.cameras.names[idx]}  hits={hits}/{samples}"
        display = _make_input_output_panel(self.last_render, gen)
        hud = "splat input | WorldFM output    " + hud
        if bool(self.args.debug_panel):
            cond = np.array(Image.open(self.cameras.image_paths[idx]).convert("RGB"))
            display = _make_debug_panel(self.last_render, cond, gen)
            hud = "render | source frame | WorldFM    " + hud
        cv2.putText(display, hud, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(display, hud, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)

        self.frame_id += 1
        return display

    def save_current(self) -> None:
        if self.last_generated is None:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        out = self.output_dir / f"interactive_{self.frame_id:06d}.png"
        Image.fromarray(self.last_generated).save(out)
        meta = {
            "frame_id": self.frame_id,
            "cond_name": self.cameras.names[self.last_cond_idx],
            "cond_image": str(self.cameras.image_paths[self.last_cond_idx]),
            "K": self.K_render.tolist(),
            "c2w": self.c2w.tolist(),
        }
        (out.with_suffix(".json")).write_text(json.dumps(meta, indent=2), encoding="utf-8")
        _log(f"Saved {out}")

    def on_mouse(self, event: int, x: int, y: int, flags: int, param) -> None:
        del flags, param
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.last_mouse = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
            self.last_mouse = None
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging and self.last_mouse is not None:
            lx, ly = self.last_mouse
            dx, dy = x - lx, y - ly
            self.last_mouse = (x, y)
            self.c2w = _rotate_camera_local(
                self.c2w,
                yaw=float(dx) * float(self.args.mouse_sensitivity),
                pitch=float(dy) * float(self.args.mouse_sensitivity),
            )
            self.dirty = True

    def apply_key(self, key: int) -> bool:
        if key < 0:
            return False
        ch = chr(key & 0xFF).lower() if 0 <= (key & 0xFF) < 128 else ""
        move = float(self.args.move_step)
        rot = float(self.args.key_turn_step)
        changed = False

        if ch == "w":
            self.c2w = _move_camera(self.c2w, np.array([0.0, 0.0, move]))
            changed = True
        elif ch == "s":
            self.c2w = _move_camera(self.c2w, np.array([0.0, 0.0, -move]))
            changed = True
        elif ch == "a":
            self.c2w = _move_camera(self.c2w, np.array([-move, 0.0, 0.0]))
            changed = True
        elif ch == "d":
            self.c2w = _move_camera(self.c2w, np.array([move, 0.0, 0.0]))
            changed = True
        elif ch == "q":
            self.c2w = _move_camera(self.c2w, np.array([0.0, -move, 0.0]))
            changed = True
        elif ch == "e":
            self.c2w = _move_camera(self.c2w, np.array([0.0, move, 0.0]))
            changed = True
        elif ch == "j":
            self.c2w = _rotate_camera_local(self.c2w, yaw=-rot, pitch=0.0)
            changed = True
        elif ch == "l":
            self.c2w = _rotate_camera_local(self.c2w, yaw=rot, pitch=0.0)
            changed = True
        elif ch == "i":
            self.c2w = _rotate_camera_local(self.c2w, yaw=0.0, pitch=-rot)
            changed = True
        elif ch == "k":
            self.c2w = _rotate_camera_local(self.c2w, yaw=0.0, pitch=rot)
            changed = True
        elif ch == "r":
            idx = _nearest_source_index(self.cameras, self.c2w)
            self.c2w = self.cameras.c2w[idx].copy()
            changed = True
        elif ch == "p":
            self.save_current()
        return changed

    def run(self) -> None:
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window, self.on_mouse)
        _log("Controls: WASD move, QE vertical, mouse drag look, IJKL rotate, R snap nearest source pose, P save, Esc quit")

        display = self.generate_current()
        cv2.imshow(self.window, cv2.cvtColor(display, cv2.COLOR_RGB2BGR))
        prev_drag_state = self.dragging
        while True:
            key = cv2.waitKey(10)
            if key in (27, ord("x")):
                break
            changed = self.apply_key(key)
            if self.dragging != prev_drag_state:
                prev_drag_state = self.dragging
            if changed or self.dirty:
                self.dirty = False
                display = self.generate_current()
                cv2.imshow(self.window, cv2.cvtColor(display, cv2.COLOR_RGB2BGR))
        cv2.destroyWindow(self.window)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Interactive WorldFM from existing PLY + source frames, without HunyuanWorld/MoGe.",
    )
    p.add_argument("--scene_dir", type=str, default=str(DEFAULT_SCENE_DIR),
                   help="LingBot-MAP export directory containing points/, images/, intri.yml, extri.yml")
    p.add_argument("--ply", type=str, default="",
                   help="Point cloud PLY. Defaults to <scene_dir>/points/point_cloud.ply")
    p.add_argument("--image_dir", type=str, default="",
                   help="Resized geometry frame directory. Defaults to <scene_dir>/images")
    p.add_argument("--original_frame_dir", type=str, default="",
                   help="Original video frame directory for cond2 images. Defaults to <scene_dir>/runtime/extracted_frames if present")
    p.add_argument("--intri_yml", type=str, default="",
                   help="Intrinsics YAML. Defaults to <scene_dir>/intri.yml")
    p.add_argument("--extri_yml", type=str, default="",
                   help="Camera-to-world YAML. Defaults to <scene_dir>/extri.yml")
    p.add_argument("--output_dir", type=str, default="outputs/ply_worldfm_interactive")

    p.add_argument("--render_size", type=int, default=512)
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--max_points", type=int, default=0)
    p.add_argument("--splat_radius", type=int, default=2,
                   help="Disk radius in pixels for point splatting. Larger values fill holes but blur geometry.")
    p.add_argument("--frame_stride", type=int, default=1,
                   help="Use every Nth source frame as a cond2 candidate.")
    p.add_argument("--max_frames", type=int, default=0,
                   help="Limit source-frame candidates; 0 means all.")
    p.add_argument("--start_frame", type=int, default=0)

    p.add_argument("--step", type=int, default=2,
                   help="WorldFM sampling steps. 1/2 use DMD fast path; larger values use DPM solver.")
    p.add_argument("--model_path", type=str, default="",
                   help="Local checkpoint or HF ref. Defaults to worldfm_1-step.pth for --step=1, otherwise worldfm_2-step.pth from inspatio/worldfm cache.")
    p.add_argument("--vae_path", type=str, default="vae",
                   help="Local VAE dir or HF ref. Defaults to vae from inspatio/worldfm cache.")
    p.add_argument("--version", type=str, default="sigma", choices=["sigma", "alpha"])
    p.add_argument("--cfg_scale", type=float, default=4.5)
    p.add_argument("--gpu_index", type=int, default=0)

    p.add_argument("--sample_grid", type=int, default=10)
    p.add_argument("--center_grid", type=int, default=15)
    p.add_argument("--center_frac", type=float, default=0.5)
    p.add_argument("--eps_rel", type=float, default=0.02)
    p.add_argument("--eps_abs", type=float, default=0.0)
    p.add_argument("--px_radius", type=int, default=1)
    p.add_argument("--max_view_angle_deg", type=float, default=180.0)
    p.add_argument("--use_distance_weight", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--dist_min_m", type=float, default=1.0)
    p.add_argument("--dist_max_m", type=float, default=20.0)
    p.add_argument("--weight_near", type=float, default=1.0)
    p.add_argument("--weight_far", type=float, default=0.0)

    p.add_argument("--move_step", type=float, default=0.25)
    p.add_argument("--key_turn_step", type=float, default=0.12)
    p.add_argument("--mouse_sensitivity", type=float, default=0.003)
    p.add_argument("--window_name", type=str, default="WorldFM PLY Interactive")
    p.add_argument("--debug_panel", action="store_true",
                   help="Show raw point render and selected source frame next to the generated frame.")
    p.add_argument("--profile", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()
    if torch.cuda.is_available() and int(args.gpu_index) >= 0:
        torch.cuda.set_device(int(args.gpu_index))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_str = f"cuda:{torch.cuda.current_device()}" if device.type == "cuda" else "cpu"

    scene_dir = _resolve(args.scene_dir)
    ply_path = _resolve(args.ply) if args.ply else scene_dir / "points" / "point_cloud.ply"
    image_dir = _resolve(args.image_dir) if args.image_dir else scene_dir / "images"
    default_orig = scene_dir / "runtime" / "extracted_frames"
    original_frame_dir = _resolve(args.original_frame_dir) if args.original_frame_dir else (default_orig if default_orig.exists() else None)
    intri_yml = _resolve(args.intri_yml) if args.intri_yml else scene_dir / "intri.yml"
    extri_yml = _resolve(args.extri_yml) if args.extri_yml else scene_dir / "extri.yml"
    output_dir = _resolve(args.output_dir)

    for pth, label in (
        (ply_path, "PLY"),
        (image_dir, "image_dir"),
        (intri_yml, "intri_yml"),
        (extri_yml, "extri_yml"),
    ):
        if not pth.exists():
            raise FileNotFoundError(f"{label} not found: {pth}")

    cameras = _load_source_cameras(
        scene_dir=scene_dir,
        image_dir=image_dir,
        original_frame_dir=original_frame_dir,
        intri_yml=intri_yml,
        extri_yml=extri_yml,
        frame_stride=int(args.frame_stride),
        max_frames=int(args.max_frames),
    )
    _log(f"Loaded {len(cameras.names)} source cameras; cond2 frames from {original_frame_dir or image_dir}")

    xyz, rgb = _load_points(ply_path)
    renderer = TorchPointCloudRenderer(
        points_xyz=xyz,
        points_rgb=rgb,
        width=int(args.render_size),
        height=int(args.render_size),
        device=device_str,
        mode="fast",
        max_points=(int(args.max_points) if int(args.max_points) > 0 else None),
        splat_radius=int(args.splat_radius),
    )
    _log(
        f"Renderer uses {renderer.num_points_used:,}/{renderer.num_points_total:,} "
        f"points on {device_str}, splat_radius={int(args.splat_radius)}"
    )

    cond_db = _build_condition_db_from_source_frames(
        cameras=cameras,
        renderer=renderer,
        render_size=int(args.render_size),
        device=device,
    )

    model_path = _model_ref_from_arg(args.model_path, int(args.step))
    vae_path = _vae_ref_from_arg(args.vae_path)
    _log(f"Loading WorldFM: {model_path}")
    svc = WorldFMTriConditionInprocess(
        WorldFMInprocessConfig(
            model_path=str(model_path),
            vae_path=str(vae_path),
            image_size=int(args.image_size),
            version=str(args.version),
            disable_cross_attn=True,
            step=(int(args.step) if int(args.step) in (1, 2) else 2),
            cfg_scale=float(args.cfg_scale),
            device=device_str,
            weight_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            profile=bool(args.profile),
        )
    )
    _log("Encoding source-frame cond2 candidates")
    svc.set_cond2_candidates_from_paths([str(p) for p in cameras.image_paths])

    start = int(np.clip(int(args.start_frame), 0, len(cameras.c2w) - 1))
    K_render = scale_K_for_resize(
        cameras.K[start],
        src_wh=cameras.source_wh,
        dst_wh=(int(args.render_size), int(args.render_size)),
    )
    app = InteractiveWorldFMApp(
        renderer=renderer,
        cond_db=cond_db,
        cameras=cameras,
        svc=svc,
        K_render=K_render,
        c2w_init=cameras.c2w[start],
        args=args,
        output_dir=output_dir,
    )
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
