#!/bin/bash

wandb docker-run \
    -d \
    --gpus=all \
    -e CUDA_VISIBLE_DEVICES="$1" \
    -e XLA_PYTHON_CLIENT_MEM_FRACTION=.10 \
    -e LD_LIBRARY_PATH="\$LD_LIBRARY_PATH:/root/.mujoco/mjpro150/bin" \
    -v ~/FullDICE:/workspaces/FullDICE \
    -w /workspaces/FullDICE/ \
    coredice:1.0.0 \
    /bin/bash -c "source activate bregman && pip install -e 3rdparty/safety-gym && $2"
