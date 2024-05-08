#!/bin/bash

ALPHAS=(0.01 1.0)
SEEDS=(3 4) #(0 1 2)

GPU_ID="$1"
ALG="UBROIDICE"
ENV="$2" # ("finance-medium-100", "finance-high-100")
DIV="SoftChiT"
PROJ_NAME="roidice_neorl_finance_high"

EVAL_INTERVAL=100
EVAL_EPISODES=10

for seed in ${SEEDS[*]}; do
for alpha in ${ALPHAS[*]}; do
XLA_PYTHON_CLIENT_MEM_FRACTION=.20 CUDA_VISIBLE_DEVICES="$GPU_ID" python neural/train_evaluation.py \
    --alg "$ALG" \
    --proj_name $PROJ_NAME \
    --env_name "$ENV" \
    --max_steps 100000 \
    --divergence "$DIV" \
    --cost_ub 1000 \
    --config=./neural/configs/mujoco_config.py \
    --alpha "$alpha" \
    --eval_interval $EVAL_INTERVAL \
    --eval_episodes $EVAL_EPISODES \
    --seed $seed \
    ${@:2}
done
done
