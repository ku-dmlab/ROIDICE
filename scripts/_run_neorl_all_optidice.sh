#!/bin/bash

ALPHAS=(0.01 0.1 1.0)
SEEDS=(0) #(3 4) #(0 1 2)

GPU_ID="$1"
ALG="OptiDICE"
ENV="$2" # ("finance-medium-100", "finance-high-100")
DIV="SoftChi"
PROJ_NAME="neorl_finance_high_state_norm"

EVAL_INTERVAL=1000
EVAL_EPISODES=10

COST_WEIGHT=0.0001
COST_LB=0.1

for seed in ${SEEDS[*]}; do
for alpha in ${ALPHAS[*]}; do
XLA_PYTHON_CLIENT_MEM_FRACTION=.20 CUDA_VISIBLE_DEVICES="$GPU_ID" python neural/train_evaluation.py \
    --alg "$ALG" \
    --proj_name $PROJ_NAME \
    --env_name "$ENV" \
    --max_steps 100000 \
    --divergence "$DIV" \
    --cost_weight $COST_WEIGHT \
    --cost_lb $COST_LB \
    --config=./neural/configs/finance_config.py \
    --alpha "$alpha" \
    --eval_interval $EVAL_INTERVAL \
    --eval_episodes $EVAL_EPISODES \
    --seed $seed \
    ${@:2}
done
done
