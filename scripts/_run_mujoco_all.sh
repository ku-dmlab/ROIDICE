#!/bin/bash

ALPHAS=(0.01 0.1 1.0 10.0)

GPU_ID="$1"
ALG="$2"
ENV=("hopper-medium-expert-v2" "halfcheetah-medium-expert-v2" "walker2d-medium-expert-v2")
DIV="$3"
PROJ_NAME="roidice_fixed"

EVAL_INTERVAL=100
EVAL_EPISODES=10

for alpha in ${ALPHAS[*]}; do
for env in ${ENV[*]}; do
XLA_PYTHON_CLIENT_MEM_FRACTION=.20 CUDA_VISIBLE_DEVICES="$GPU_ID" python neural/train_evaluation.py \
    --alg "$ALG" \
    --proj_name $PROJ_NAME \
    --env_name "$env" \
    --max_steps 50000 \
    --divergence "$DIV" \
    --cost_ub 1000 \
    --config=./neural/configs/mujoco_config.py \
    --alpha "$alpha" \
    --eval_interval $EVAL_INTERVAL \
    --eval_episodes $EVAL_EPISODES \
    --seed 0 \
    ${@:3}
done
done
