#!/bin/bash

# ALPHAS=(0.01 0.01 0.01)
SEEDS=(0 1 2 3 4)

GPU_ID="$1"
ALG="LambdaCOptiDICE"
ENV="$2" #("hopper-medium-expert-v2" "halfcheetah-medium-expert-v2" "walker2d-medium-expert-v2")
DIV="SoftChi"
# COST_UB=1.0 # (0.5 0.6 0.7) (1.5 1.7 1.9) (2.0 2.5, 3.0)
INITIAL_LAMBDA=(1 2 5)
ALPHA="$3"
PROJ_NAME="coptidice_mujoco_hopper_lambda_mean_cost"

EVAL_INTERVAL=100
EVAL_EPISODES=10

for seed in ${SEEDS[*]}; do
for initial_lambda in ${INITIAL_LAMBDA[*]}; do
XLA_PYTHON_CLIENT_MEM_FRACTION=.20 CUDA_VISIBLE_DEVICES="$GPU_ID" python neural/train_evaluation.py \
    --alg "$ALG" \
    --proj_name $PROJ_NAME \
    --env_name $ENV \
    --max_steps 100000 \
    --divergence "$DIV" \
    --cost_ub 1000 \
    --config=./neural/configs/mujoco_config.py \
    --alpha "$ALPHA" \
    --eval_interval $EVAL_INTERVAL \
    --eval_episodes $EVAL_EPISODES \
    --initial_lambda $initial_lambda \
    --seed $seed
done
done