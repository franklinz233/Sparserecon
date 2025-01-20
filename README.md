# Unposed Sparse Views Reconstruction

Env Requirement
- torch 2.1.0
- CUDA 12.1
- A6000 GPU
## Install

1. Clone

```bash
git clone --recursive https://github.com/QitaoZhao/SparseAGS.git
```

2. Create the env:

```bash
conda create -n myenv python=3.9
conda activate myenv

# CUDA 12.1
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt


# pytorch3d
conda install https://anaconda.org/pytorch3d/pytorch3d/0.7.8/download/linux-64/pytorch3d-0.7.8-py39_cu121_pyt210.tar.bz2

sudo apt-get install libglm-dev

# thirdparty
pip install ./liegroups
pip install ./simple-knn
pip install ./diff-gaussian-rasterization-camera 
```



3. Download our 6-DoF Zero123 [checkpoint](https://drive.google.com/file/d/1JJ4wjaJ4IkUERRZYRrlNoQ-tXvftEYJD/view?usp=sharing) and place it under `SparseAGS/checkpoints`.

```bash
mkdir checkpoints
cd checkpoints/
pip install gdown
gdown "https://drive.google.com/uc?id=1JJ4wjaJ4IkUERRZYRrlNoQ-tXvftEYJD"
cd ..
```

## Usage

```bash
### preprocess
python process.py data/name.jpg
python process.py data

# run single 3D reconstruction
python run.py --category filename --num_views 8 
```


