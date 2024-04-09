#!/bin/bash

ALPHAS=(0.01) #(0.01 0.1 1.0 10.0)
SEEDS=(0)

GPU_ID="$1"
ALG="$2"
ENV="$3"
DIV="$4"
PROJ_NAME="$5"

EVAL_INTERVAL=10
EVAL_EPISODES=2

for seed in ${SEEDS[*]}; do
	for alpha in ${ALPHAS[*]}; do
		CUDA_VISIBLE_DEVICES="$GPU_ID" python neural/train_evaluation.py \
		    --alg "$ALG" \
			--env_name "$ENV" \
			--proj_name "$PROJ_NAME" \
			--max_steps 20000 \
			--divergence "$DIV" \
			--config=./neural/configs/safety_gym_config.py \
			--alpha "$alpha" \
			--eval_interval $EVAL_INTERVAL \
			--eval_episodes $EVAL_EPISODES \
			--seed "$seed" \
			${@:5}
	done
done
