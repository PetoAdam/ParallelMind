# ParallelMind

[![CI](https://github.com/PetoAdam/ParallelMind/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/PetoAdam/ParallelMind/actions/workflows/ci.yml)

A small, CUDA‑first neural network library in modern C++. It’s designed to be easy to read, fast on GPU, and fun to hack on.

## Highlights
- GPU-accelerated math (C++17 + CUDA)
- Simple feedforward networks (dense layers)
- Activations: ReLU, Sigmoid, Linear
- SGD training with shuffling
- Save/Load models (binary format)
- MNIST example app (train or load + ASCII visualization)
- Unit tests for core components

## Dev Container (Quickstart)
This repo includes a dev-friendly Docker setup based on `nvidia/cuda:12.2.2-cudnn8-devel-ubuntu20.04`.

- Requires: NVIDIA driver + Docker + NVIDIA Container Toolkit.
- Image includes: build-essential, recent CMake, Python3/pip.
- Optional Python deps (numpy/torch/torchvision) can be preinstalled via build args to keep images lean by default.

Build options:
```bash
# 1) Lean image (default, no heavy Python deps baked in)
docker build -f docker/Dockerfile -t parallelmind-dev .

# 2) Convenience image with Python deps preinstalled
docker build -f docker/Dockerfile -t parallelmind-dev \
  --build-arg INSTALL_PY_DEPS=1 \
  # use CPU-only wheels to keep size reasonable (omit to let pip pick GPU builds)
  --build-arg PYTORCH_CHANNEL=cpu \
  .
```
Run the container and build the project:
```bash
docker run --gpus all -it --rm \
  -v "$PWD":/workspace \
  -w /workspace \
  parallelmind-dev

# Inside the container
mkdir -p build && cd build && cmake .. && make -j
```
Generate MNIST data (if not preinstalled, install deps inside the container first):
```bash
# Inside the container
pip3 install --no-cache-dir numpy torch torchvision  # or CPU wheels if preferred
cd /workspace/data/mnist && python3 mnist_download.py
```
Run example (from `/workspace/build`):
```bash
./examples/mnist_example --train --save mnist.pmmdl
```
For VS Code Dev Containers, use this image and ensure GPU passthrough is enabled.

## Requirements
- NVIDIA GPU with a compatible CUDA toolkit (tested with CUDA 12.2)
- CMake (3.18+ recommended)
- A C++17 compiler and NVCC
- Optional (only to generate MNIST `.npy` files): Python 3 with `numpy`, `torch`, and `torchvision`

## Getting the MNIST data
This repository does not include the MNIST `.npy` files. Use the provided script to download and convert MNIST into NumPy arrays:

```bash
# From the project root
cd data/mnist
# (Optional) install dependencies if you don't already have them
pip3 install --user numpy torch torchvision
# Generate the .npy files in data/mnist/
python3 mnist_download.py
```
The script will create these files in `data/mnist/`:
- `mnist_train_images.npy`, `mnist_train_labels.npy`
- `mnist_test_images.npy`, `mnist_test_labels.npy`

## Build
```bash
mkdir -p build
cd build
cmake ..
make -j
```

## Run the MNIST example
Train and save a model:
```bash
# From build/
./examples/mnist_example --train --save mnist.pmmdl
```

Load a saved model and evaluate:
```bash
# From build/
./examples/mnist_example --load mnist.pmmdl
```
The example prints per-sample ASCII digits with predictions and reports overall test accuracy.

## Tests
From the `build/` folder run:
```bash
ctest --output-on-failure
```
This runs unit tests for matrix ops, layers, nodes, and the NPY reader.

## Continuous Integration
- GitHub Actions builds the Docker image (lean variant) and caches MNIST datasets.
- CI installs CPU-only `torch`, `torchvision`, and `numpy` at runtime to generate `.npy` files.
- The workflow builds the project and runs only fast tests (currently `test_npy_reader`).
- The MNIST training example is intentionally not run in CI to keep builds fast and reliable.

## Project layout
```
src/            # Core library (Matrix, Layer, Network, CUDA kernels)
examples/       # Example apps (MNIST runner)
tests/          # Unit tests
data/mnist/     # MNIST download/convert script and generated .npy files
```

## Roadmap
- Training: softmax + cross-entropy, mini-batching
- Performance: more fused CUDA kernels, better memory reuse
- Features: model config I/O (JSON/YAML), metrics & logging improvements
- Data: URL/stream-based dataset loading (download/cache into .npy)
- Serving: REST API for training/evaluation with model load/save endpoints
- Portability: CPU fallback and headless builds
- UI: optional web frontend (separate repo) for training status and model introspection

## Contributing
Issues and PRs are welcome. If you have ideas or find bugs, open an issue or propose a change. Contributions improving docs, tests, and CI are especially appreciated.

## License
MIT — see `LICENSE`.
