# ParallelMind

A small, CUDA‑first neural network library in modern C++. It’s designed to be easy to read, fast on GPU, and fun to hack on.

## Highlights
- GPU-accelerated math (C++17 + CUDA)
- Simple feedforward networks (dense layers)
- Activations: ReLU, Sigmoid, Linear, Softmax
- Loss: Cross-entropy (with Softmax output)
- SGD training with shuffling and mini-batching
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

## Training options and math (concise)

Activations (per neuron):
- ReLU: a = max(0, z). Derivative: a' = 1 if z > 0 else 0.
- Sigmoid: a = 1/(1 + e^(−z)). Derivative: a' = a·(1−a).
- Linear: a = z. Derivative: a' = 1.
- Softmax (vector): y_i = exp(z_i − max(z)) / Σ_j exp(z_j − max(z)). Stable subtract avoids overflow.

Loss (multi-class): Cross-entropy L = −Σ_i t_i log y_i, where t is one‑hot target and y is Softmax output.
- Gradient w.r.t. logits z: ∂L/∂z = y − t. This is what the trainer backpropagates on the last layer when using Softmax output.

Mini-batching:
- With batch size B, gradients are averaged over B samples before an update. This reduces variance and can be more stable.
- Effective learning rate per sample is lr/B; the example does this scaling internally when you pass --batch B.

CLI options in the MNIST example:
- Hidden activation: --hid-act relu|sigmoid|linear
- Output activation: --out-act softmax|relu|sigmoid|linear (use softmax for classification)
- Mini-batch size: --batch N (0 = per-sample SGD)
- Epochs: --epochs K (default 5)
- Learning rate: --lr 0.01 (default 0.01)

Examples:
```bash
# Recommended for classification: ReLU hidden + Softmax output, batch 128, 5 epochs @ 0.01
./examples/mnist_example --train --out-act softmax --hid-act relu --batch 128 --epochs 5 --lr 0.01

# Linear output (e.g., for regression-style experiments), sigmoid hidden
./examples/mnist_example --train --out-act linear --hid-act sigmoid --batch 0 --epochs 10 --lr 0.005
```

## Project layout
```
src/            # Core library (Matrix, Layer, Network, CUDA kernels)
examples/       # Example apps (MNIST runner)
tests/          # Unit tests
data/mnist/     # MNIST download/convert script and generated .npy files
```

## Roadmap
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
