This repository provides the core implementation of SPIN-Net.

## Environment

The implementation has been tested with the following environment:

- Linux (x86_64)
- Python 3.10.16
- PyTorch 2.5.1
- CUDA 12.4
- NumPy 1.26.4
- SciPy 1.14.1
- h5py 3.12.1
- scikit-learn 1.5.2
- PyYAML 6.0.2

A CUDA-capable GPU is recommended for large cortical inputs.

## Installation

Create an isolated environment:

```bash
conda create -n spinnet python=3.10 -y
conda activate spinnet
```

For a CUDA 12.4 system, install the tested PyTorch build and the remaining
dependencies with:

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
```

For CPU-only use:

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```
