#!/bin/bash

ALPHAS=(1.0) #(0.01 0.1 1.0 10.0)
SEEDS=(0) #(0 1 2 3 4)

GPU_ID="$1"
ALG="$2"
ENV="$3" #("hopper-medium-expert-v2" "halfcheetah-medium-expert-v2" "walker2d-medium-expert-v2")
DIV="SoftChi"
COST_UB="$4" # (0.5 0.6 0.7) (1.5 1.7 1.9) (2.0 2.5, 3.0)
PROJ_NAME="coptidice_absorbing_healthy_medium_expert_ub"

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
    --cost_ub $COST_UB \
    --config=./neural/configs/mujoco_config.py \
    --alpha "$alpha" \
    --eval_interval $EVAL_INTERVAL \
    --eval_episodes $EVAL_EPISODES \
    --log_video False \
    --seed $seed \
    ${@:4}
done
done
