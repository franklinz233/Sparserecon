# SparseRecon

SparseRecon reconstructs textured 3D objects from sparse, unposed input views. The pipeline handles foreground preprocessing, optional camera pose initialization, coarse 3D optimization, mesh refinement, and export of textured meshes.

The codebase is organized as an importable Python package (`sparserecon`) with small compatibility entry scripts at the repository root.

## Repository Layout

```text
configs/                 Reconstruction configs
dust3r/                  DUSt3R dependency used for pose initialization
sparserecon/             Main SparseRecon package
sparserecon/stages/      Coarse Gaussian and mesh refinement stages
submodules/              CUDA/Python extension dependencies
gradio_app.py            Wrapper for sparserecon.demo_app
process.py               Wrapper for sparserecon.preprocess
run.py                   Wrapper for sparserecon.cli
requirements.txt         Python dependencies
NOTICE                   Third-party dependency notes
```

## Environment

Reference environment:

- Python 3.9
- PyTorch 2.1.0
- CUDA 12.1
- Linux with an NVIDIA GPU
- A high-memory GPU is recommended for full reconstruction

Windows is suitable for code editing and lightweight checks, but the CUDA extension stack is easier to build and run on Linux.

## Installation

Clone with submodules:

```bash
git clone --recursive https://github.com/franklinz233/Sparserecon.git
cd Sparserecon
```

Create the environment:

```bash
conda create -n sparserecon python=3.9 -y
conda activate sparserecon

pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Install PyTorch3D and system dependencies:

```bash
conda install https://anaconda.org/pytorch3d/pytorch3d/0.7.8/download/linux-64/pytorch3d-0.7.8-py39_cu121_pyt210.tar.bz2
sudo apt-get update
sudo apt-get install -y libglm-dev
```

Build local extensions:

```bash
pip install ./submodules/liegroups
pip install ./submodules/simple-knn
pip install ./submodules/diff-gaussian-rasterization-camera
```

Download the 6-DoF Zero123 checkpoint:

```bash
mkdir -p checkpoints
pip install gdown
gdown "https://drive.google.com/uc?id=1JJ4wjaJ4IkUERRZYRrlNoQ-tXvftEYJD" -O checkpoints/zero123_6dof_23k.ckpt
```

The expected checkpoint path is:

```text
checkpoints/zero123_6dof_23k.ckpt
```

## Data Preparation

Preprocess an image directory:

```bash
python -m sparserecon.preprocess path/to/images
```

The preprocessor writes:

```text
path/to/images/source/       RGB images
path/to/images/processed/    RGBA images with removed background
```

For CLI reconstruction, prepare a dataset under `data/demo/<name>/`:

```text
data/demo/<name>/cameras.json
data/demo/<name>/source/*.png
data/demo/<name>/processed/*_rgba.png
```

## Reconstruction

Run reconstruction:

```bash
python -m sparserecon.cli --category <name> --num_views <N> --mesh_format obj
```

Enable outlier detection and correction:

```bash
python -m sparserecon.cli --category <name> --num_views <N> --mesh_format obj --enable_loop
```

Compatibility wrapper:

```bash
python run.py --category <name> --num_views <N> --mesh_format obj
```

Outputs are written under:

```text
output/demo/<name>/
```

## Gradio Demo

Launch locally:

```bash
python -m sparserecon.demo_app
```

Compatibility wrapper:

```bash
python gradio_app.py
```

Create a public Gradio share link:

```bash
python -m sparserecon.demo_app --share
```

## Common Options

- `--category`: dataset folder under `data/demo`.
- `--num_views`: number of input views to evaluate.
- `--num_pts`: number of initial Gaussian points.
- `--mesh_format`: output mesh format, for example `obj` or `glb`.
- `--enable_loop`: enable outlier detection and correction.
- `--render_video`: render an orbit video after reconstruction.
- `--config`: config name under `configs/` or an explicit config path.
- `--output`: output root directory.

## Development Notes

- Main project code lives under `sparserecon/`.
- Generated outputs, checkpoints, Python caches, and extension build artifacts are ignored by `.gitignore`.
- Third-party code is isolated under `dust3r/` and `submodules/`.
- Do not commit CUDA extension build outputs or downloaded model checkpoints.
