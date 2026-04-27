#!/usr/bin/env python3
"""
Export a Pi3X reconstruction in the WorldFM point-cloud input layout.

The script samples at most 16 frames from a video, runs Pi3X, and writes:

  images/                  resized frames used by Pi3X and camera intrinsics
  runtime/extracted_frames original selected video frames for WorldFM cond2
  points/point_cloud.ply   filtered colored global point cloud
  intri.yml / extri.yml    OpenCV intrinsics and camera-to-world poses

It does not depend on HunyuanWorld, MoGe, or LingBot-MAP.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


WORLDFM_ROOT = Path(__file__).resolve().parent
DEFAULT_PI3_ROOT = WORLDFM_ROOT / "../../../Pi3"
DEFAULT_VIDEO = WORLDFM_ROOT / "../../benchmarks/ict_5floor_panorama.mp4"
DEFAULT_OUTPUT = WORLDFM_ROOT / "../../outputs/ict_5floor/pi3x"


def _resolve(path: str | Path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else (Path.cwd() / p).resolve()


def _log(msg: str) -> None:
    print(f"[Pi3X-WorldFM] {msg}", flush=True)


def _target_size(width: int, height: int, pixel_limit: int) -> tuple[int, int]:
    scale = math.sqrt(float(pixel_limit) / float(width * height)) if width * height > 0 else 1.0
    w_target, h_target = width * scale, height * scale
    k, m = round(w_target / 14), round(h_target / 14)
    while (k * 14) * (m * 14) > int(pixel_limit):
        if k / max(m, 1) > w_target / max(h_target, 1e-6):
            k -= 1
        else:
            m -= 1
    return max(1, k) * 14, max(1, m) * 14


def _sample_video_frames(
    video_path: Path,
    *,
    max_frames: int,
    pixel_limit: int,
    output_dir: Path,
) -> tuple[torch.Tensor, list[str], list[int], tuple[int, int], tuple[int, int]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise ValueError(f"Video has no readable frame count: {video_path}")

    n = min(max(1, int(max_frames)), total)
    indices = np.linspace(0, total, n, endpoint=False, dtype=np.int64).tolist()
    indices = sorted(set(int(i) for i in indices))
    n = len(indices)

    orig_dir = output_dir / "runtime" / "extracted_frames"
    img_dir = output_dir / "images"
    orig_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    names = []
    orig_wh: tuple[int, int] | None = None
    resized_wh: tuple[int, int] | None = None

    for out_i, frame_i in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_i)
        ok, bgr = cap.read()
        if not ok:
            raise RuntimeError(f"Failed to read frame {frame_i} from {video_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        if orig_wh is None:
            orig_wh = (w, h)
            resized_wh = _target_size(w, h, pixel_limit)
            _log(f"video={w}x{h}, sampled={n}, Pi3X resize={resized_wh[0]}x{resized_wh[1]}")

        name = f"{out_i:06d}"
        names.append(name)
        Image.fromarray(rgb).save(orig_dir / f"{name}.jpg", quality=95)
        resized = Image.fromarray(rgb).resize(resized_wh, Image.Resampling.LANCZOS)
        resized.save(img_dir / f"{name}.png")
        arr = np.asarray(resized, dtype=np.float32) / 255.0
        frames.append(torch.from_numpy(arr).permute(2, 0, 1))

    cap.release()
    if not frames or orig_wh is None or resized_wh is None:
        raise RuntimeError("No frames extracted.")
    return torch.stack(frames, dim=0), names, indices, orig_wh, resized_wh


def _write_binary_ply(path: Path, xyz: np.ndarray, rgb_u8: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    xyz = np.asarray(xyz, dtype=np.float32)
    rgb_u8 = np.asarray(rgb_u8, dtype=np.uint8)
    if xyz.shape[0] != rgb_u8.shape[0]:
        raise ValueError(f"PLY xyz/rgb length mismatch: {xyz.shape[0]} vs {rgb_u8.shape[0]}")
    vertex = np.empty(
        xyz.shape[0],
        dtype=[
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ],
    )
    vertex["x"], vertex["y"], vertex["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    vertex["red"], vertex["green"], vertex["blue"] = rgb_u8[:, 0], rgb_u8[:, 1], rgb_u8[:, 2]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {xyz.shape[0]}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    with path.open("wb") as f:
        f.write(header)
        vertex.tofile(f)


def _write_opencv_matrix_yml(path: Path, names: list[str], matrices: np.ndarray, node_name: str) -> None:
    lines = ["%YAML:1.0", "---", "names:"]
    for name in names:
        lines.append(f'  - "{name}"')
    lines.append(f"{node_name}:")
    for name, mat in zip(names, matrices):
        arr = np.asarray(mat, dtype=np.float64)
        flat = ", ".join(f"{v:.10g}" for v in arr.reshape(-1))
        lines.extend([
            f'  "{name}": !!opencv-matrix',
            f"    rows: {arr.shape[0]}",
            f"    cols: {arr.shape[1]}",
            "    dt: d",
            f"    data: [{flat}]",
        ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_mat_txt(path: Path, c2w: np.ndarray) -> None:
    lines = []
    for pose in c2w:
        pose4 = pose if pose.shape == (4, 4) else np.vstack([pose, [0, 0, 0, 1]])
        lines.append(" ".join(f"{v:.6f}" for v in pose4.reshape(-1)))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _copy_reference(output_dir: Path, names: list[str]) -> None:
    src = output_dir / "runtime" / "extracted_frames" / f"{names[0]}.jpg"
    if src.exists():
        shutil.copyfile(src, output_dir / "reference.png")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run Pi3X on <=16 video frames and export a WorldFM-ready point cloud.")
    p.add_argument("--video_path", type=str, default=str(DEFAULT_VIDEO))
    p.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT))
    p.add_argument("--pi3_root", type=str, default=str(DEFAULT_PI3_ROOT))
    p.add_argument("--ckpt", type=str, default="", help="Optional local Pi3X safetensors/pth checkpoint.")
    p.add_argument("--max_frames", type=int, default=16)
    p.add_argument("--pixel_limit", type=int, default=255000)
    p.add_argument("--conf_threshold", type=float, default=0.1)
    p.add_argument("--edge_rtol", type=float, default=0.03)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--gpu_index", type=int, default=0)
    p.add_argument("--force", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()
    video_path = _resolve(args.video_path)
    output_dir = _resolve(args.output_dir)
    pi3_root = _resolve(args.pi3_root)
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    if not pi3_root.is_dir():
        raise FileNotFoundError(pi3_root)
    if output_dir.exists() and args.force:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "runtime").mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(pi3_root))
    from pi3.models.pi3x import Pi3X
    from pi3.utils.geometry import depth_edge, recover_intrinsic_from_rays_d

    if torch.cuda.is_available() and args.device.startswith("cuda") and int(args.gpu_index) >= 0:
        torch.cuda.set_device(int(args.gpu_index))
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    imgs_cpu, names, frame_indices, orig_wh, resized_wh = _sample_video_frames(
        video_path,
        max_frames=min(16, int(args.max_frames)),
        pixel_limit=int(args.pixel_limit),
        output_dir=output_dir,
    )
    imgs = imgs_cpu.unsqueeze(0).to(device)

    use_multimodal = False
    _log("Loading Pi3X")
    if args.ckpt:
        model = Pi3X(use_multimodal=use_multimodal).eval()
        ckpt = _resolve(args.ckpt)
        if ckpt.suffix == ".safetensors":
            from safetensors.torch import load_file
            weight = load_file(str(ckpt))
        else:
            weight = torch.load(str(ckpt), map_location=device, weights_only=False)
        model.load_state_dict(weight, strict=False)
    else:
        model = Pi3X.from_pretrained("yyfz233/Pi3X").eval()
        model.disable_multimodal()
    model = model.to(device)

    dtype = torch.float32
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
    _log(f"Running Pi3X on {imgs.shape[1]} frames at {resized_wh[0]}x{resized_wh[1]} ({device}, {dtype})")
    t0 = time.perf_counter()
    with torch.no_grad():
        if device.type == "cuda":
            with torch.amp.autocast("cuda", dtype=dtype):
                res = model(imgs=imgs)
        else:
            res = model(imgs=imgs)
    infer_sec = time.perf_counter() - t0
    _log(f"Pi3X inference done in {infer_sec:.2f}s")

    rays_d = torch.nn.functional.normalize(res["local_points"], dim=-1)
    K = recover_intrinsic_from_rays_d(rays_d, force_center_principal_point=True)[0].detach().float().cpu().numpy()
    c2w = res["camera_poses"][0].detach().float().cpu().numpy()
    points = res["points"][0].detach().float().cpu()
    conf = torch.sigmoid(res["conf"][..., 0])[0].detach().float().cpu()
    edge = depth_edge(res["local_points"][..., 2], rtol=float(args.edge_rtol))[0].detach().cpu()
    mask = (conf > float(args.conf_threshold)) & (~edge) & torch.isfinite(points).all(dim=-1)

    colors = imgs_cpu.permute(0, 2, 3, 1)
    xyz = points[mask].numpy().astype(np.float32)
    rgb = (colors[mask].numpy().clip(0, 1) * 255.0 + 0.5).astype(np.uint8)
    conf_np = conf[mask].numpy().astype(np.float32)
    if xyz.size == 0:
        raise RuntimeError("Pi3X produced no valid points after filtering.")

    points_dir = output_dir / "points"
    _write_binary_ply(points_dir / "point_cloud.ply", xyz, rgb)
    _write_binary_ply(points_dir / "whole.ply", xyz, rgb)
    np.savez_compressed(points_dir / "point_cloud.npz", vertices=xyz, colors=rgb, confidence=conf_np)

    _write_opencv_matrix_yml(output_dir / "intri.yml", names, K, "intrinsics")
    _write_opencv_matrix_yml(output_dir / "extri.yml", names, c2w[:, :3, :4], "extrinsics")
    _write_mat_txt(output_dir / "mat.txt", c2w)
    _copy_reference(output_dir, names)

    manifest = [str((output_dir / "runtime" / "extracted_frames" / f"{name}.jpg").resolve()) for name in names]
    (output_dir / "runtime" / "image_manifest.txt").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    run_plan = {
        "status": "completed",
        "model": "yyfz233/Pi3X",
        "video_path": str(video_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "frame_indices": frame_indices,
        "image_count": len(names),
        "original_size": list(orig_wh),
        "resized_size": list(resized_wh),
        "pixel_limit": int(args.pixel_limit),
        "conf_threshold": float(args.conf_threshold),
        "edge_rtol": float(args.edge_rtol),
        "point_count": int(xyz.shape[0]),
        "inference_seconds": infer_sec,
        "coordinate_mode": "pi3x_global_opencv_c2w",
    }
    (output_dir / "runtime" / "run_plan.json").write_text(json.dumps(run_plan, indent=2), encoding="utf-8")
    _log(f"Exported {len(names)} frames to {output_dir}")
    _log(f"point cloud: {points_dir / 'point_cloud.ply'} ({xyz.shape[0]:,} points)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
