#!/bin/bash

ALPHAS=(0.01 0.1 1.0)
SEEDS=(3 4) #(0 1 2 3 4)

GPU_ID="$1"
ALG="OptiDICE"
ENV="$2" #("hopper-medium-v2" "hopper-medium-expert-v2" "hopper-expert-v2")
DIV="SoftChi"
PROJ_NAME="mujoco_hopper_final"

EVAL_INTERVAL=1000
EVAL_EPISODES=10

# for idx in "${!ALPHAS[@]}"; do
for alpha in ${ALPHAS[*]}; do
for seed in ${SEEDS[*]}; do
# alpha=${ALPHAS[$idx]}
# env=${ENV[$idx]}
XLA_PYTHON_CLIENT_MEM_FRACTION=.20 CUDA_VISIBLE_DEVICES="$GPU_ID" python neural/train_evaluation.py \
    --alg "$ALG" \
    --proj_name $PROJ_NAME \
    --env_name $ENV \
    --max_steps 100000 \
    --divergence "$DIV" \
    --cost_ub 1000 \
    --config=./neural/configs/mujoco_config.py \
    --alpha "$alpha" \
    --eval_interval $EVAL_INTERVAL \
    --eval_episodes $EVAL_EPISODES \
    --log_video True \
    --seed $seed
done
done
