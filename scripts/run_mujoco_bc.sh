#!/bin/bash
SEEDS=(0)

GPU_ID="$1"
ALG="BC"
ENV=("hopper-medium-v2" "hopper-medium-expert-v2" "hopper-expert-v2" "walker2d-medium-v2" "walker2d-mediume-expert-v2" "walker2d-expert-v2" "halfcheetah-medium-v2" "halfcheetah-medium-expert-v2" "halfcheetah-expert-v2")
PROJ_NAME="BC_mujoco"
SIGMA=(0.5 0.8)

EVAL_INTERVAL=1000
EVAL_EPISODES=10

for seed in ${SEEDS[*]}; do
for env in ${ENV[*]}; do
for sigma in ${SIGMA[*]}; do
	XLA_PYTHON_CLIENT_MEM_FRACTION=.10 CUDA_VISIBLE_DEVICES="$GPU_ID" python neural/train_evaluation.py \
		--alg "$ALG" \
		--env_name "$ENV" \
		--proj_name "$PROJ_NAME" \
		--max_steps 50000 \
		--config=./neural/configs/mujoco_config.py \
		--eval_interval $EVAL_INTERVAL \
		--eval_episodes $EVAL_EPISODES \
		--sigma $sigma \
		--seed "$seed" \
		${@:2}
done
done
done