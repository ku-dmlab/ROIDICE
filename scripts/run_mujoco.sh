#!/bin/bash

ALPHAS=(0.01 0.1 1.0 10.0)
SEEDS=(0 1 2)

GPU_ID="$1"
ALG="$2"
ENV="$3"
DIV="$4"
COST_UB=$5
PROJ_NAME="$6"

EVAL_INTERVAL=100
EVAL_EPISODES=10

for seed in ${SEEDS[*]}; do
	for alpha in ${ALPHAS[*]}; do
		XLA_PYTHON_CLIENT_MEM_FRACTION=.20 CUDA_VISIBLE_DEVICES="$GPU_ID" python neural/train_evaluation.py \
		    --alg "$ALG" \
			--env_name "$ENV" \
			--proj_name "$PROJ_NAME" \
			--max_steps 50000 \
			--divergence "$DIV" \
			--cost_ub $COST_UB \
			--config=./neural/configs/mujoco_config.py \
			--alpha "$alpha" \
			--eval_interval $EVAL_INTERVAL \
			--eval_episodes $EVAL_EPISODES \
			--seed "$seed" \
			${@:5}
	done
done
