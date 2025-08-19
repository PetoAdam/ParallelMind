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
- Image installs: build-essential, recent CMake, Python3/pip, and Python deps for MNIST (`numpy`, `torchvision`).

Quickstart:
```bash
# Build the dev image (from repo root)
docker build -f docker/Dockerfile -t parallelmind-dev .

# Run with GPU access and mount the repo
docker run --gpus all -it --rm \
  -v "$PWD":/workspace \
  -w /workspace \
  parallelmind-dev

# Inside the container, build
mkdir -p build && cd build && cmake .. && make -j

# (Optional) generate MNIST data
cd /workspace/data/mnist && python3 mnist_download.py

# Run example (from /workspace/build)
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

## Project layout
```
src/            # Core library (Matrix, Layer, Network, CUDA kernels)
examples/       # Example apps (MNIST runner)
tests/          # Unit tests
data/mnist/     # MNIST download/convert script and generated .npy files
```

## Roadmap
- CI/CD: add build + test workflows (Linux GPU and CPU-only matrices)
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
