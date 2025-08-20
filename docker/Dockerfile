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

# Install CMake 3.18 or higher from Kitware repository
RUN apt-get update && apt-get install -y \
    software-properties-common lsb-release \
    && apt-get clean all \
    && apt-get remove --purge --auto-remove -y cmake \
    && apt-get update \
    && apt-get install -y wget \
    && wget -qO- https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null \
    && apt-add-repository "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main" \
    && apt-get update \
    && apt-get install -y kitware-archive-keyring \
    && rm /etc/apt/trusted.gpg.d/kitware.gpg \
    && apt-get update \
    && apt-get install -y cmake \
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

