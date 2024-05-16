#!/bin/bash
SEEDS=(0)

GPU_ID="$1"
ALG="BC"
ENV="$2"
PROJ_NAME="BC_mujoco"

EVAL_INTERVAL=1000
EVAL_EPISODES=10

for seed in ${SEEDS[*]}; do
	for alpha in ${ALPHAS[*]}; do
		XLA_PYTHON_CLIENT_MEM_FRACTION=.10 CUDA_VISIBLE_DEVICES="$GPU_ID" python neural/train_evaluation.py \
		    --alg "$ALG" \
			--env_name "$ENV" \
			--proj_name "$PROJ_NAME" \
			--max_steps 50000 \
			--config=./neural/configs/mujoco_config.py \
			--alpha "$alpha" \
			--eval_interval $EVAL_INTERVAL \
			--eval_episodes $EVAL_EPISODES \
			--seed "$seed" \
			${@:2}
	done
done
