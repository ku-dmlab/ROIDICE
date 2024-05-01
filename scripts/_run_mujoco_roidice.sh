#!/bin/bash

ALPHAS=(0.01 1.0 0.01)
COST_WEIGHT=(0.01 0.001 0.001)
COST_LB=(1 1 1)
# SEEDS=(0 1 2 3 4)

GPU_ID="$1"
ALG="$2"
ENV="$3" #("hopper-medium-expert-v2" "halfcheetah-medium-expert-v2" "walker2d-medium-expert-v2")
DIV="$4"
SEED="$5"
PROJ_NAME="roidice_customed_cost_fix_rand_seeds"

EVAL_INTERVAL=100
EVAL_EPISODES=10

# for seed in ${SEEDS[*]}; do
for idx in "${!ALPHAS[@]}"; do
alpha=${ALPHAS[$idx]}
cost_weight=${COST_WEIGHT[$idx]}
cost_lb=${COST_LB[$idx]}
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
    --cost_lb $cost_lb \
    --cost_weight $cost_weight \
    --log_video True \
    --seed $SEED \
    ${@:5}
done
# done
