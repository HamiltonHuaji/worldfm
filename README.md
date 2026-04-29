# WorldFM

<br/>

<div align="center">
    <img src="resources/worldfm.jpg"/>
</div>
<br/>

<div align="left">
<div align="left">

[![LICENSE](https://img.shields.io/github/license/inspatio/worldfm)](https://github.com/inspatio/worldfm/blob/main/LICENSE)  [![arXiv](https://img.shields.io/badge/arXiv-2603.11911-b31b1b.svg)](https://arxiv.org/abs/2603.11911)  [![Discord](https://img.shields.io/badge/Discord-Join-7289da?logo=discord&logoColor=white)](https://discord.gg/SyyjR3Z57w)


WorldFM, a real-time multi-view diffusion model. Given a reference image and target camera poses, WorldFM generates images at those new viewpoints. Checkout our website ([WorldFM](https://inspatio.github.io/worldfm)) for videos and interactive results!

## Installation

### 1. Create Conda Environment

```bash
# Edit CONDA_ENV_PATH in setup.sh to your desired prefix first
bash setup.sh
```

This will:

- Create the `WorldFM` conda environment (Python 3.10, PyTorch 2.5, CUDA 12.4)
- Install pip dependencies from `requirements.txt`
- Initialize git submodules (HunyuanWorld-1.0, MoGe, Real-ESRGAN, ZIM)
- Build Real-ESRGAN and ZIM in development mode

### 2. Manual Setup (alternative)

```bash
conda env create -f WorldFM.yaml --prefix /path/to/envs/WorldFM
conda activate /path/to/envs/WorldFM
pip install -r requirements.txt
git submodule update --init --recursive
cd submodules/MoGe
git checkout 7807b5de2bc0c1e80519f5f3d1f38a606f8f9925

# HunyuanWorld-1.0 requirements
cd ../Real-ESRGAN
pip install basicsr-fixed facexlib gfpgan
python setup.py develop
cd ../ZIM
pip install -e .
```

For consistent scene generation, we employ an internal generative model that is not included in the open-source release.
To support reproducibility, users can integrate alternative open-source panorama generation models (e.g., HunyuanWorld-1.0). This substitution does not impact the core spatial reasoning framework of WorldFM.

## Getting Started

### Download Pretrained Model

By default, this fork resolves WorldFM weights from the HuggingFace cache for
`inspatio/worldfm`. You do not need to copy checkpoint files into `weights/` if
the files are already cached by `huggingface_hub`; missing files will be
downloaded by HuggingFace Hub unless `WORLDFM_HF_LOCAL_ONLY=1` is set.

Supported checkpoint refs:

```bash
--model_path worldfm_2-step.pth
--model_path inspatio/worldfm
--model_path inspatio/worldfm:worldfm_2-step.pth
--model_path hf://inspatio/worldfm/worldfm_2-step.pth
--vae_path vae
--vae_path inspatio/worldfm
--vae_path inspatio/worldfm:vae
```

You can still materialize files into `weights/` explicitly:

```sh
python download_ckpts.py
```

You will get:

```
weights/
  ├── vae/
  ├── worldfm_1-step.pth  # DMD step=1, faster
  └── worldfm_2-step.pth  # DMD step=2, better quality
```

Use `--step 1` or `--step 2` in `run_pipeline.py` to select the corresponding model.

## Usage

### Headless WorldFM From an Existing Point Cloud

This fork includes a lightweight path that skips HunyuanWorld, MoGe, Real-ESRGAN,
ZIM, `mmcv`, and `mmengine`. It is intended for server deployment where you
already have a colored point cloud, camera intrinsics/poses, and source frames.

The expected scene layout is:

```
scene_dir/
  points/point_cloud.ply
  images/000000.png                 # resized frames matching intri.yml
  runtime/extracted_frames/000000.jpg  # original frames used as reference cond2
  intri.yml
  extri.yml                         # OpenCV camera-to-world matrices
```

Launch the browser demo:

```bash
python demo_ply_worldfm_web.py \
  --scene_dir <SCENE_DIR> \
  --host 0.0.0.0 \
  --port 7860 \
  --step 2 \
  --splat_radius 2
```

Open `http://<server-ip>:7860/`. If the server is only reachable through SSH:

```bash
ssh -L 7860:127.0.0.1:7860 <user>@<server>
```

Then open `http://127.0.0.1:7860/` locally.

The web view shows `point-cloud splat input | WorldFM output`. With
`--debug_panel`, it shows `point-cloud splat input | selected source frame |
WorldFM output`.

Controls: `WASD` move, `QE` vertical move, mouse drag looks around, `IJKL`
rotates, `R` snaps to the nearest source pose, and `P` saves the current frame.

### Optional: Build the Scene With Pi3X

For higher-quality geometry from a video, this fork can call a sibling Pi3/Pi3X
checkout and export a WorldFM-ready scene. Pi3X is strong but memory-heavy, so
the exporter samples at most 16 frames.

```bash
git clone git@github.com:yyfz/Pi3.git ../Pi3

python export_pi3x_worldfm.py \
  --video_path ../../benchmarks/ict_5floor_panorama.mp4 \
  --output_dir ../../outputs/ict_5floor/pi3x \
  --pi3_root ../Pi3 \
  --max_frames 16 \
  --force
```

Then run:

```bash
python demo_ply_worldfm_web.py \
  --scene_dir ../../outputs/ict_5floor/pi3x \
  --host 0.0.0.0 \
  --port 7860 \
  --step 2 \
  --splat_radius 2
```

`demo_ply_worldfm.py` provides the same interaction through an OpenCV desktop
window, but `demo_ply_worldfm_web.py` is the recommended server entry point.

### Demo

We provide a sample scene with a pre-defined camera trajectory in `demo/`. Run the following command to generate an MP4 video along the trajectory:

```bash
python run_pipeline.py --meta demo/meta.json --output_dir outputs
```

The output video will be saved to `outputs/<scene_name>/output.mp4`.

### Input Format

Prepare a `meta.json` file:

Single pose:

```json
{
  "name": "scene_001",
  "image": "input.jpg",
  "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
  "c2w": [
    [r00, r01, r02, tx],
    [r10, r11, r12, ty],
    [r20, r21, r22, tz],
    [  0,   0,   0,  1]
  ]
}
```

Multiple poses (generates one output per pose):

```json
{
  "name": "scene_001",
  "image": "input.jpg",
  "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
  "c2w": [
    [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
    [[...], [...], [...], [...]],
    ...
  ]
}
```

- **name**: scene identifier, used as the output subdirectory name
- **image**: relative path (from `meta.json` location) to the input perspective image
- **K**: 3×3 camera intrinsic matrix
- **c2w**: a single 4×4 or a list of N×4×4 camera-to-world matrices (target viewpoints)

### Run Inference with Your Own Data

```bash
# Default: output as MP4 video
python run_pipeline.py --meta <META_JSON> --output_dir <OUTPUT_DIR>

# Save per-frame PNG images instead
python run_pipeline.py --meta <META_JSON> --output_dir <OUTPUT_DIR> --save_mode image
```

### Configuration

Default parameters are defined in `default.yaml`. Override them via:

1. **CLI arguments** (highest priority)
2. **Custom config file**: `--config my_config.yaml`
3. `**default.yaml`** (lowest priority)

### Output

With `--save_mode video` (default):

```
<output_dir>/<name>/
  └── output.mp4          # Video composed of all generated frames
```

With `--save_mode image`:

```
<output_dir>/<name>/
  ├── output.png           # Single pose
  # or
  ├── output_0000.png      # Multiple poses
  ├── output_0001.png
  └── ...
```

# License

The license of our codebase is [Apache-2.0](https://github.com/inspatio/worldfm/blob/main/LICENSE). Note that this license only applies to code in our library, the dependencies and submodules of which ([HunyuanWorld-1.0](https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0/blob/main/LICENSE), [MoGe](https://github.com/microsoft/MoGe/blob/main/LICENSE)) are separate and individually licensed.

# Contributing

We appreciate all contributions to improve WorldFM.

# Citing

If you use WorldFM in your research, please use the following BibTeX entry.

```bib
@misc{worldfm,
    title={Inspatio-WorldFM: An Open-Source Real-Time Generative Frame Model for Spatial Intelligence},
    author={WorldFM Contributors},
    howpublished = {\url{https://github.com/inspatio/worldfm}},
    year={2026}
}
```

# Acknowledgement

This codebase is built upon [PixArt-Sigma](https://github.com/PixArt-alpha/PixArt-sigma). We would like to express our gratitude to the PixArt Team for open-sourcing their code and models. Their contributions have been instrumental to the development of this project. We also appreciate [PRoPe](https://github.com/liruilong940607/prope), [HunyuanWorld-1.0](https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0.git) and [MoGe](https://github.com/microsoft/MoGe.git) for their excellent work.
