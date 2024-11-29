FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt update && apt install -y \
    build-essential cmake git python3 python3-pip && \
    apt clean

# Install CMake 3.18 or higher from Kitware repository
RUN apt update && apt install -y \
    software-properties-common lsb-release \
    && apt clean all \
    && apt remove --purge --auto-remove -y cmake \
    && apt update \
    && apt install -y wget \
    && wget -qO- https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null \
    && apt-add-repository "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main" \
    && apt update \
    && apt install kitware-archive-keyring \
    && rm /etc/apt/trusted.gpg.d/kitware.gpg \
    && apt update \
    && apt install -y cmake

# Install required python libraries
RUN pip3 install torchvision numpy

# Install VS Code server dependencies (if using devcontainer)
RUN apt update && apt install -y \
    curl unzip && \
    apt clean

# Set up working directory
WORKDIR /usr/src/app

# Default command
CMD ["/bin/bash"]
