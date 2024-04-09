# Ubuntu 20.04.6 LTS (Focal Fossa)
FROM condaforge/mambaforge:23.1.0-1

# Tell apt-get we're never going to be able to give manual feedback.
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
# Update the package index and isntall the packages.
RUN apt-get update \
    && apt-get -y upgrade \
    && apt-get -y install --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        libosmesa6-dev \
        libgl1-mesa-glx \
        libglfw3 \
        patchelf \
        ca-certificates \
        curl \
        wget \
        xauth \
        xvfb \
        unzip \
        zip \
        vim \
        graphviz \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Go and pprof for JAX memory profiling.
ENV PATH=$PATH:/usr/local/go/bin
RUN wget -O /tmp/go.tar.gz https://go.dev/dl/go1.20.3.linux-amd64.tar.gz \
    && tar -C /usr/local -xzf /tmp/go.tar.gz \
    && /usr/local/go/bin/go install github.com/google/pprof@latest

# Install MuJoCo 1.5.0
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mjpro150/bin
RUN wget -q -O /tmp/mujoco.zip https://www.roboti.us/download/mjpro150_linux.zip \
    && mkdir /root/.mujoco \
    && unzip /tmp/mujoco.zip -d /root/.mujoco \
    && wget -q -O /root/.mujoco/mjkey.txt https://www.roboti.us/file/mjkey.txt

# Override CUDA version for Mamba.
ARG CONDA_OVERRIDE_CUDA="11.2"

# Install the required Python packages.
COPY env.yaml /tmp/env.yaml
RUN mamba env create -n bregman -f /tmp/env.yaml
