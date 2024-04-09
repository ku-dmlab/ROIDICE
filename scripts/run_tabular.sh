#!/bin/bash
NUM_ITERS=1000
# NUM_TRAJ=(1000)
# ALPHA=(1.0, 0.1, 0.01, 0.001)
# TARGET_ROI=(0.1, 1.0, 5.0, 10.0)
BASIC_RL=0
UNCONSTRAINED=0
VANILLA_CONSTRAINED=0
CONSERVATIVE_CONSTRAINED=0
ROIDICE=0
ROIDICE_LOWER_BOUND=0
MAX_ROIDICE=1
MAX_ROIDICE_REG=0

for num_traj in 1000
do
    for alpha in 1.0 0.1 0.01 0.001
    do
        for target_roi in 0.1 #1.0 5.0 10.0
        do
            for cost_thresholds in 0.2 # 20.0 0.05 0.075
            do
                python tabular/run_random_cmdp_experiment.py \
                                            --path "max_roidice_cost" \
                                            --cost_thresholds ${cost_thresholds} \
                                            --behavior_cost_thresholds ${cost_thresholds} \
                                            --num_trajectories ${num_traj} \
                                            --alpha ${alpha} \
                                            --target_roi ${target_roi} \
                                            --num_iterations ${NUM_ITERS} \
                                            --basic_rl ${BASIC_RL} \
                                            --unconstrained ${UNCONSTRAINED} \
                                            --vanilla_constrained ${VANILLA_CONSTRAINED} \
                                            --conservative_constrained ${CONSERVATIVE_CONSTRAINED} \
                                            --roidice ${ROIDICE} \
                                            --roidice_lower_bound ${ROIDICE_LOWER_BOUND} \
                                            --max_roidice ${MAX_ROIDICE}
            done
        done
    done
done

python analysis.py \
                --name "max_roidice_cost" \
                --path "max_roidice_cost \
                --basic_rl ${BASIC_RL} \
                --unconstrained ${UNCONSTRAINED} \
                --vanilla_constrained ${VANILLA_CONSTRAINED} \
                --conservative_constrained ${CONSERVATIVE_CONSTRAINED} \
                --roidice ${ROIDICE} \
                --max_roidice ${MAX_ROIDICE}