FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu20.04 AS dev

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Preconfigure tzdata to avoid interactive prompts
RUN ln -fs /usr/share/zoneinfo/$TZ /etc/localtime && \
    apt-get update && apt-get install -y --no-install-recommends tzdata && \
    dpkg-reconfigure -f noninteractive tzdata && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential cmake git python3 python3-pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install CMake (3.18+), adding Kitware repo with signed-by keyring
RUN apt-get update && apt-get install -y --no-install-recommends \
     ca-certificates gnupg wget lsb-release \
     && rm -rf /var/lib/apt/lists/* \
     && echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main" \
         | tee /etc/apt/sources.list.d/kitware.list >/dev/null \
     && wget -qO - https://apt.kitware.com/keys/kitware-archive-latest.asc \
         | gpg --dearmor - \
         | tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null \
     && apt-get update \
     && apt-get install -y --no-install-recommends kitware-archive-keyring cmake \
     && apt-get clean && rm -rf /var/lib/apt/lists/*

# Optional: install Python libraries (controlled via build args)
# By default this is disabled to keep the image small and CI-friendly.
ARG INSTALL_PY_DEPS=0
# Set to "cpu" to force CPU-only PyTorch wheels (smaller); leave empty for default upstream selection
ARG PYTORCH_CHANNEL=
RUN if [ "$INSTALL_PY_DEPS" = "1" ]; then \
            pip3 install --no-cache-dir numpy && \
            if [ "$PYTORCH_CHANNEL" = "cpu" ]; then \
                pip3 install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision; \
            else \
                pip3 install --no-cache-dir torch torchvision; \
            fi; \
        fi

# Install VS Code server dependencies (if using devcontainer)
RUN apt-get update && apt-get install -y \
    curl unzip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /usr/src/app

# Default command
CMD ["/bin/bash"]

