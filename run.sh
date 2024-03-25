#!/bin/bash
NUM_ITERS=1000
# NUM_TRAJ=(1000)
# ALPHA=(1.0, 0.1, 0.01, 0.001)
# TARGET_ROI=(0.1, 1.0, 5.0, 10.0)
BASIC_RL=0
UNCONSTRAINED=1
VANILLA_CONSTRAINED=1
CONSERVATIVE_CONSTRAINED=0
ROIDICE=1

for num_traj in 1000
do
    for alpha in 1.0 0.1 0.01 0.001
    do
        for target_roi in 0.1 1.0 5.0 10.0
        do
            python tabular/run_random_cmdp_experiment.py \
                                        --num_trajectories ${num_traj} \
                                        --alpha ${alpha} \
                                        --target_roi ${target_roi} \
                                        --num_iterations ${NUM_ITERS} \
                                        --basic_rl $BASIC_RL \
                                        --unconstrained ${UNCONSTRAINED} \
                                        --vanilla_constrained ${VANILLA_CONSTRAINED} \
                                        --conservative_constrained ${CONSERVATIVE_CONSTRAINED} \
                                        --roidice ${ROIDICE}
        done
    done
done

python analysis.py