# ParallelMind

A small, CUDA‑first neural network library in modern C++. It’s designed to be easy to read, fast on GPU, and fun to hack on.

## Highlights
- GPU-accelerated math (C++17 + CUDA)
- Simple feedforward networks (dense layers)
- Activations: ReLU, Sigmoid, Linear
- SGD training with shuffling
- Save/Load models (binary format)
- MNIST example app (train or load + ASCII visualization)
- Unit tests for core components

## Quickstart

### Option 1 — Dev Container (easiest):
- Requires: Docker + NVIDIA Container Toolkit (for GPU access)
- Build the image:
	```bash
	docker build -f docker/Dockerfile -t parallelmind-dev .
	```
- Run it mounted to your repo:
	```bash
	docker run --gpus all -it --rm \
		-v "$PWD":/workspace \
		-w /workspace \
		parallelmind-dev
	```
- Inside the container, build and test:
	```bash
	mkdir -p build && cd build && cmake .. && make -j
	ctest --output-on-failure
	```

 Note: If you use VS Code:
 - Install the "Dev Containers" extension in VS Code.
 - Open this repo in VS Code and run: Command Palette → "Dev Containers: Reopen in Container".
 - It will use `docker/Dockerfile` automatically. Then run the same build/test commands in VS Code’s terminal.

### Option 2 — Local build (no container):
- Requires: NVIDIA GPU, CUDA toolkit (tested with 12.2), CMake 3.18+, C++17 compiler, NVCC
- Build and test:
	```bash
	mkdir -p build && cd build
	cmake ..
	make -j
	ctest --output-on-failure
	```

### Example (MNIST, optional, works in both setups):
```bash
# 1) Prepare data (once, creates .npy files under data/mnist/)
cd /workspace
pip3 install -r requirements.txt
python3 data/mnist/mnist_download.py

# 2) Train and save a model (from build/)
cd /workspace/build
./examples/mnist_example --train --save mnist.pmmdl

# 3) Load and evaluate (from build/)
cd /workspace/build
./examples/mnist_example --load mnist.pmmdl
```
Notes:
- The Docker image is lean by default; install Python deps inside it only if you need the MNIST script: `pip3 install -r requirements.txt`.
- The example reports accuracy and can print ASCII digits with predictions.

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
