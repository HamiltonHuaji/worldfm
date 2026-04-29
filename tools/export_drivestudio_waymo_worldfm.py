#!/usr/bin/env python3
"""Export a DriveStudio-processed Waymo scene for the WorldFM PLY demo."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


OPENCV2DATASET = np.array(
    [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]],
    dtype=np.float64,
)


def _resolve(path: str | Path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else (Path.cwd() / p).resolve()


def _log(msg: str) -> None:
    print(f"[DriveStudio-Waymo->WorldFM] {msg}", flush=True)


def _read_env_value(env_path: Path, key: str) -> str | None:
    if not env_path.exists():
        return None
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k.strip() == key:
            return v.strip().strip("'\"")
    return None


def _scene_root_from_args(args: argparse.Namespace) -> Path:
    if args.scene_root:
        return _resolve(args.scene_root)
    env_path = _resolve(args.env)
    data_root = _read_env_value(env_path, "waymo_training_root")
    if not data_root:
        raise ValueError(f"Missing waymo_training_root in {env_path}")
    return _resolve(data_root) / f"{int(args.scene_idx):03d}"


def _matrix_yml(path: Path, node_name: str, names: list[str], mats: dict[str, np.ndarray]) -> None:
    lines = ["%YAML:1.0", "---", "names:"]
    for name in names:
        lines.append(f'  - "{name}"')
    lines.append(f"{node_name}:")
    for name in names:
        arr = np.asarray(mats[name], dtype=np.float64)
        flat = ", ".join(f"{v:.12g}" for v in arr.reshape(-1))
        lines.extend(
            [
                f'  "{name}": !!opencv-matrix',
                f"    rows: {arr.shape[0]}",
                f"    cols: {arr.shape[1]}",
                "    dt: d",
                f"    data: [{flat}]",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_binary_ply(path: Path, xyz: np.ndarray, rgb_u8: np.ndarray) -> None:
    xyz = np.asarray(xyz, dtype=np.float32)
    rgb_u8 = np.asarray(rgb_u8, dtype=np.uint8)
    vertex = np.empty(
        xyz.shape[0],
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(header)
        vertex.tofile(f)


def _scaled_intrinsic(raw_intrinsic: np.ndarray, src_wh: tuple[int, int], dst_wh: tuple[int, int]) -> np.ndarray:
    fx, fy, cx, cy = raw_intrinsic[:4].astype(np.float64)
    sx = float(dst_wh[0]) / float(src_wh[0])
    sy = float(dst_wh[1]) / float(src_wh[1])
    return np.array([[fx * sx, 0.0, cx * sx], [0.0, fy * sy, cy * sy], [0.0, 0.0, 1.0]], dtype=np.float64)


def _cam_to_world(scene_root: Path, cam_id: int, timestep: int, start_timestep: int) -> np.ndarray:
    cam_to_ego = np.loadtxt(scene_root / "extrinsics" / f"{cam_id}.txt").astype(np.float64) @ OPENCV2DATASET
    ego_start = np.loadtxt(scene_root / "ego_pose" / f"{start_timestep:03d}.txt").astype(np.float64)
    ego_cur = np.loadtxt(scene_root / "ego_pose" / f"{timestep:03d}.txt").astype(np.float64)
    return np.linalg.inv(ego_start) @ ego_cur @ cam_to_ego


def _ego_to_world(scene_root: Path, timestep: int, start_timestep: int) -> np.ndarray:
    ego_start = np.loadtxt(scene_root / "ego_pose" / f"{start_timestep:03d}.txt").astype(np.float64)
    ego_cur = np.loadtxt(scene_root / "ego_pose" / f"{timestep:03d}.txt").astype(np.float64)
    return np.linalg.inv(ego_start) @ ego_cur


def _project_color(points_world: np.ndarray, c2w: np.ndarray, K: np.ndarray, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    w2c = np.linalg.inv(c2w)
    pts_h = np.concatenate([points_world, np.ones((points_world.shape[0], 1), dtype=np.float64)], axis=1)
    pts_cam = (w2c @ pts_h.T).T[:, :3]
    z = pts_cam[:, 2]
    valid_z = z > 1e-3
    uvw = (K @ pts_cam.T).T
    u = uvw[:, 0] / np.maximum(uvw[:, 2], 1e-6)
    v = uvw[:, 1] / np.maximum(uvw[:, 2], 1e-6)
    h, w = image.shape[:2]
    ui = np.rint(u).astype(np.int64)
    vi = np.rint(v).astype(np.int64)
    valid = valid_z & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    colors = np.zeros((points_world.shape[0], 3), dtype=np.uint8)
    colors[valid] = image[vi[valid], ui[valid]]
    return valid, colors


def _load_camera_image_and_intrinsics(
    image_path: Path,
    raw_intrinsic: np.ndarray,
    *,
    target_width: int,
    undistort: bool,
) -> tuple[np.ndarray, np.ndarray]:
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(image_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    dst_w = int(target_width)
    dst_h = int(round(h * dst_w / w))
    rgb = cv2.resize(rgb, (dst_w, dst_h), interpolation=cv2.INTER_LANCZOS4)
    K = _scaled_intrinsic(raw_intrinsic, (w, h), (dst_w, dst_h))
    if not undistort:
        return rgb, K

    D = np.asarray(raw_intrinsic[4:9], dtype=np.float64)
    if D.shape[0] != 5 or not np.any(D):
        return rgb, K
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (dst_w, dst_h), alpha=1)
    rgb = cv2.undistort(rgb, K, D, None, new_K)
    return rgb, new_K.astype(np.float64)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Export DriveStudio Waymo processed data to WorldFM PLY demo layout.")
    p.add_argument("--env", default="../drivestudio/.env")
    p.add_argument("--scene_root", default="", help="Processed Waymo scene dir. Overrides --env/--scene_idx.")
    p.add_argument("--scene_idx", type=int, default=5)
    p.add_argument("--output_dir", default="outputs/waymo_worldfm_scene005")
    p.add_argument("--cameras", default="0,1,2", help="Comma-separated Waymo camera ids.")
    p.add_argument("--start_timestep", type=int, default=0)
    p.add_argument("--num_timesteps", type=int, default=20)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--max_points", type=int, default=1200000)
    p.add_argument("--truncated_min_x", type=float, default=-2.0)
    p.add_argument("--truncated_max_x", type=float, default=80.0)
    p.add_argument("--undistort", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--keep_uncolored",
        action="store_true",
        help="Keep LiDAR points that are outside all selected camera views, coloring them gray.",
    )
    p.add_argument("--force", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()
    scene_root = _scene_root_from_args(args)
    output_dir = _resolve(args.output_dir)
    if not scene_root.is_dir():
        raise FileNotFoundError(scene_root)
    if output_dir.exists() and args.force:
        shutil.rmtree(output_dir)
    for sub in ("images", "runtime/extracted_frames", "points"):
        (output_dir / sub).mkdir(parents=True, exist_ok=True)

    cameras = [int(x) for x in args.cameras.split(",") if x.strip()]
    timesteps = [
        int(args.start_timestep) + i * max(1, int(args.stride))
        for i in range(max(1, int(args.num_timesteps)))
    ]
    timesteps = [t for t in timesteps if (scene_root / "ego_pose" / f"{t:03d}.txt").exists()]
    if not timesteps:
        raise RuntimeError("No valid timesteps selected.")
    start_t = timesteps[0]

    names: list[str] = []
    intrinsics: dict[str, np.ndarray] = {}
    extrinsics: dict[str, np.ndarray] = {}
    resized_images: dict[tuple[int, int], np.ndarray] = {}

    _log(f"scene={scene_root}, timesteps={timesteps[0]}..{timesteps[-1]} stride={args.stride}, cameras={cameras}")
    for t in timesteps:
        for cam_id in cameras:
            src = scene_root / "images" / f"{t:03d}_{cam_id}.jpg"
            if not src.exists():
                continue
            raw_intrinsic = np.loadtxt(scene_root / "intrinsics" / f"{cam_id}.txt")
            arr, K = _load_camera_image_and_intrinsics(
                src,
                raw_intrinsic,
                target_width=int(args.width),
                undistort=bool(args.undistort),
            )
            name = f"{t:03d}_{cam_id}"
            names.append(name)
            intrinsics[name] = K
            extrinsics[name] = _cam_to_world(scene_root, cam_id, t, start_t)
            resized_images[(t, cam_id)] = arr
            im_resized = Image.fromarray(arr, mode="RGB")
            im_resized.save(output_dir / "images" / f"{name}.png")
            im_resized.save(output_dir / "runtime" / "extracted_frames" / f"{name}.jpg", quality=95)

    _matrix_yml(output_dir / "intri.yml", "intrinsics", names, intrinsics)
    _matrix_yml(output_dir / "extri.yml", "extrinsics", names, extrinsics)

    all_xyz: list[np.ndarray] = []
    all_rgb: list[np.ndarray] = []
    rng = np.random.default_rng(1234)
    per_scan_cap = max(1000, int(args.max_points) // max(1, len(timesteps)))

    for t in timesteps:
        lidar_path = scene_root / "lidar" / f"{t:03d}.bin"
        if not lidar_path.exists():
            continue
        lidar = np.memmap(lidar_path, dtype=np.float32, mode="r").reshape(-1, 14)
        pts_ego = np.asarray(lidar[:, 3:6], dtype=np.float64)
        ok = np.isfinite(pts_ego).all(axis=1)
        ok &= pts_ego[:, 0] > float(args.truncated_min_x)
        ok &= pts_ego[:, 0] < float(args.truncated_max_x)
        pts_ego = pts_ego[ok]
        if pts_ego.shape[0] > per_scan_cap:
            pts_ego = pts_ego[rng.choice(pts_ego.shape[0], size=per_scan_cap, replace=False)]
        ego2world = _ego_to_world(scene_root, t, start_t)
        pts_world = (ego2world[:3, :3] @ pts_ego.T + ego2world[:3, 3:4]).T

        rgb = np.full((pts_world.shape[0], 3), 160, dtype=np.uint8)
        filled = np.zeros((pts_world.shape[0],), dtype=bool)
        for cam_id in cameras:
            image = resized_images.get((t, cam_id))
            name = f"{t:03d}_{cam_id}"
            if image is None or name not in intrinsics or name not in extrinsics:
                continue
            valid, colors = _project_color(pts_world, extrinsics[name], intrinsics[name], image)
            take = valid & ~filled
            rgb[take] = colors[take]
            filled[take] = True
        if not args.keep_uncolored:
            pts_world = pts_world[filled]
            rgb = rgb[filled]
        all_xyz.append(pts_world.astype(np.float32))
        all_rgb.append(rgb)

    xyz = np.concatenate(all_xyz, axis=0)
    rgb = np.concatenate(all_rgb, axis=0)
    if xyz.shape[0] > int(args.max_points):
        keep = rng.choice(xyz.shape[0], size=int(args.max_points), replace=False)
        xyz, rgb = xyz[keep], rgb[keep]
    _write_binary_ply(output_dir / "points" / "point_cloud.ply", xyz, rgb)

    metadata = {
        "source_scene": str(scene_root),
        "scene_idx": int(args.scene_idx),
        "cameras": cameras,
        "timesteps": timesteps,
        "image_count": len(names),
        "point_count": int(xyz.shape[0]),
        "undistort": bool(args.undistort),
        "keep_uncolored": bool(args.keep_uncolored),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    _log(f"wrote {output_dir} ({len(names)} images, {xyz.shape[0]:,} points)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
