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
    build-essential git python3 python3-pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install CMake 4.1.0 via tarball (avoids apt dependency issues on 20.04)
ENV CMAKE_VER=4.1.0
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl tar \
    && rm -rf /var/lib/apt/lists/* \
    && CMAKE_MM=$(echo ${CMAKE_VER} | awk -F. '{print $1"."$2}') \
    && (curl -fsSL -o /tmp/cmake.tgz https://cmake.org/files/v${CMAKE_MM}/cmake-${CMAKE_VER}-linux-x86_64.tar.gz \
        || curl -fsSL -o /tmp/cmake.tgz https://github.com/Kitware/CMake/releases/download/v${CMAKE_VER}/cmake-${CMAKE_VER}-linux-x86_64.tar.gz) \
    && mkdir -p /opt && tar -xzf /tmp/cmake.tgz -C /opt \
    && ln -s /opt/cmake-${CMAKE_VER}-linux-x86_64 /opt/cmake
ENV PATH=/opt/cmake/bin:$PATH

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

