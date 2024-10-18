#!/bin/bash

GPU_ID="$1"
ALG="$2"
ENV="$3"
DIV="$4"
PROJ_NAME="$5"

EVAL_INTERVAL=1000
EVAL_EPISODES=10

XLA_PYTHON_CLIENT_MEM_FRACTION=.20 CUDA_VISIBLE_DEVICES="$GPU_ID" python neural/train_evaluation.py \
	--alg "$ALG" \
	--env_name "$ENV" \
	--proj_name "$PROJ_NAME" \
	--max_steps 100000 \
	--divergence "$DIV" \
	--config=./neural/configs/finance_config.py \
	--eval_interval $EVAL_INTERVAL \
	--eval_episodes $EVAL_EPISODES
