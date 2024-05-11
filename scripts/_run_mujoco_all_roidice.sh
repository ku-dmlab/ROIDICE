#!/bin/bash

ALPHAS=(0.001 0.01 0.1 1.0)
SEEDS=(0 1 2 3 4)

GPU_ID="$1"
ALG="ROIDICE"
ENV="$2" #("hopper-medium-expert-v2" "halfcheetah-medium-expert-v2" "walker2d-medium-expert-v2")
DIV="SoftChiT"
PROJ_NAME="roidice_absorbing_healthy_medium_expert"

EVAL_INTERVAL=100
EVAL_EPISODES=10

for seed in ${SEEDS[*]}; do
for alpha in ${ALPHAS[*]}; do
XLA_PYTHON_CLIENT_MEM_FRACTION=.20 CUDA_VISIBLE_DEVICES="$GPU_ID" python neural/train_evaluation.py \
    --alg "$ALG" \
    --proj_name $PROJ_NAME \
    --env_name "$ENV" \
    --max_steps 50000 \
    --divergence "$DIV" \
    --cost_ub 1000 \
    --config=./neural/configs/mujoco_config.py \
    --alpha "$alpha" \
    --eval_interval $EVAL_INTERVAL \
    --eval_episodes $EVAL_EPISODES \
    --log_video True \
    --seed $seed \
    ${@:2}
done
done
