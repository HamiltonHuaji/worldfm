
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""WorldFM checkpoint resolution helpers."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download, snapshot_download

DEFAULT_REPO_ID = "inspatio/worldfm"

pretrained_models = [
    "worldfm_1-step.pth",
    "worldfm_2-step.pth",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
]


def _local_files_only() -> bool:
    return os.environ.get("WORLDFM_HF_LOCAL_ONLY", "").lower() in {"1", "true", "yes", "on"}


def _split_hf_ref(ref: str, *, default_filename: str | None = None) -> tuple[str, str]:
    """Parse checkpoint refs.

    Supported forms:
      - worldfm_2-step.pth
      - inspatio/worldfm:worldfm_2-step.pth
      - hf://inspatio/worldfm/worldfm_2-step.pth
    """
    value = str(ref).strip()
    if value.startswith("hf://"):
        parts = value[5:].split("/", 2)
        if len(parts) < 2:
            raise ValueError(f"Bad HF ref: {ref}")
        repo_id = "/".join(parts[:2])
        filename = parts[2] if len(parts) == 3 else default_filename
        if not filename:
            raise ValueError(f"HF ref missing filename: {ref}")
        return repo_id, filename
    if ":" in value and not Path(value).exists():
        repo_id, filename = value.split(":", 1)
        if "/" in repo_id and filename:
            return repo_id, filename
    if "/" in value and not value.endswith((".pth", ".safetensors", ".pt", ".ckpt")):
        return value, default_filename or ""
    return DEFAULT_REPO_ID, value or (default_filename or "")


def resolve_model_path(model_name: str, *, revision: str = "main") -> str:
    """Resolve a WorldFM checkpoint path from local disk or HuggingFace cache."""
    local = Path(str(model_name)).expanduser()
    if local.is_file():
        return str(local)

    repo_id, filename = _split_hf_ref(str(model_name), default_filename="worldfm_2-step.pth")
    local_path = Path("weights") / filename
    if local_path.is_file():
        return str(local_path)

    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        local_files_only=_local_files_only(),
    )


def resolve_vae_path(vae_path: str = "vae", *, revision: str = "main") -> str:
    """Resolve the VAE directory from local disk or the HuggingFace cache."""
    value = str(vae_path or "vae").strip()
    local = Path(value).expanduser()
    if local.is_dir():
        return str(local)
    if (Path("weights") / value).is_dir():
        return str(Path("weights") / value)

    repo_id, subfolder = _split_hf_ref(value, default_filename="vae")
    if subfolder in {"", ".", "weights/vae"}:
        subfolder = "vae"
    snapshot = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        allow_patterns=[f"{subfolder}/*"],
        local_files_only=_local_files_only(),
    )
    return str(Path(snapshot) / subfolder)


def find_model(model_name):
    """Load a WorldFM checkpoint from local disk or HuggingFace cache."""
    model_path = resolve_model_path(model_name)
    return torch.load(model_path, map_location=lambda storage, loc: storage)


def download_model(model_name):
    """Backward-compatible wrapper that resolves and loads a pretrained model."""
    if str(model_name).endswith((".pth", ".pt", ".ckpt", ".safetensors")):
        return find_model(model_name)
    return resolve_vae_path("vae")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_names', nargs='+', type=str, default=pretrained_models)
    args = parser.parse_args()
    model_names = args.model_names
    model_names = set(model_names)

    # Resolve/download checkpoints
    for model in model_names:
        if str(model).startswith("vae/"):
            resolve_vae_path("vae")
        else:
            resolve_model_path(model)
    print('Done.')
