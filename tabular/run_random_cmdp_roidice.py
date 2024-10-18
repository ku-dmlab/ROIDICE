# Copyright 2022 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import os

from absl import app
from absl import flags
from absl import logging
import numpy as np
import pandas as pd

import mdp_util
import offline_cmdp

flags.DEFINE_float(
    "behavior_optimality",
    0.9,
    "The optimality of data-collecting policy in terms of reward. "
    "(0: performance of uniform policy. 1: performance of optimal policy).",
)
flags.DEFINE_integer(
    "num_iterations", 1000, "The number of iterations for the repeated experiments."
)

FLAGS = flags.FLAGS

# logging color
grey = "\x1b[38;20m"
yellow = "\x1b[33;20m"
green = "\x1b[32;20m"
red = "\x1b[31;20m"
bold_red = "\x1b[31;1m"
reset = "\x1b[0m"

def main(unused_argv):
    """Main function."""
    num_states, num_actions, num_costs, gamma = 50, 4, 1, 0.95
    behavior_optimality = FLAGS.behavior_optimality

    logging.info("==============================")
    logging.info("Behavior optimality: %g", behavior_optimality)
    logging.info("==============================")

    cost_multipliers = [0.7, 0.9, 1.0, 1.1, 1.3]

    for seed in range(FLAGS.num_iterations):
        # Construct a random CMDP
        np.random.seed(seed)
        cmdp = mdp_util.generate_random_cmdp(num_states, num_actions, num_costs, gamma)

        # Behavior policy
        pi_b = offline_cmdp.generate_baseline_policy(cmdp, optimality=behavior_optimality)
        
        v_r, _, v_c, _ = mdp_util.policy_evaluation(cmdp, pi_b)
        true_behav_r, true_behav_c = v_r[0], v_c[0][0]
        true_behav_roi = true_behav_r / true_behav_c

        for num_trajectories in [10, 20, 50, 100, 1000, 2000]:
            logging.info("==========================")
            logging.info("* seed=%d, num_trajectories=%d", seed, num_trajectories)
            # Generate trajectory
            trajectory = mdp_util.generate_trajectory(seed, cmdp, pi_b, num_episodes=num_trajectories)

            # MLE CMDP
            mle_cmdp, _ = mdp_util.compute_mle_cmdp(
                num_states, num_actions, num_costs, cmdp.reward, cmdp.costs, gamma, trajectory
            )

            alpha = 1 / num_trajectories

            # ROIDICE
            logging.info(yellow + f"* seed={seed}, alpha={alpha}" + reset)
            pi, _, _, t = offline_cmdp.roidice_max_roi_mosek_inf(mle_cmdp, pi_b, alpha=alpha) # seed 189, alpha 0.001
            t = t[0]

            v_r, _, v_c, _ = mdp_util.policy_evaluation(cmdp, pi)
            roi_inf_r, roi_inf_c = v_r[0], v_c[0][0]
            roi_inf_roi = roi_inf_r / roi_inf_c

            v_r, _, v_c, _ = mdp_util.policy_evaluation(mle_cmdp, pi)
            mle_roi_inf_r, mle_roi_inf_c = v_r[0], v_c[0][0]
            mle_roi_inf_roi = mle_roi_inf_r / mle_roi_inf_c

            alpha = alpha * (t**2)

            # OptiDICE
            _, _, pi, _ = offline_cmdp.optidice(mle_cmdp, pi_b, alpha)
            v_r, _, v_c, _ = mdp_util.policy_evaluation(cmdp, pi)
            odice_r = v_r[0]
            odice_c = v_c[0][0]
            odice_roi = odice_r / odice_c

            v_r, _, v_c, _ = mdp_util.policy_evaluation(mle_cmdp, pi)
            mle_odice_r = v_r[0]
            mle_odice_c = v_c[0][0]
            mle_odice_roi = mle_odice_r / mle_odice_c

            cost_threshold = roi_inf_c
            for cost_multiplier in cost_multipliers:
                logging.info(yellow + f"* seed={seed}, alpha={alpha}, cost_threshold={cost_threshold}, cost_multiplier={cost_multiplier}" + reset)
                mle_cmdp.cost_thresholds = cost_threshold * cost_multiplier

                # COptiDICE
                pi, _, _ = offline_cmdp.constrained_optidice(mle_cmdp, pi_b, alpha=alpha)
                _cost_multiplier = cost_multiplier
                while pi is None:
                    _cost_multiplier = _cost_multiplier + 0.1
                    mle_cmdp.cost_thresholds = cost_threshold * _cost_multiplier
                    pi, _, _ = offline_cmdp.constrained_optidice(mle_cmdp, pi_b, alpha=alpha)

                v_r, _, v_c, _ = mdp_util.policy_evaluation(cmdp, pi)
                cdice_r, cdice_c = v_r[0], v_c[0][0]
                cdice_roi = cdice_r / cdice_c

                v_r, _, v_c, _ = mdp_util.policy_evaluation(mle_cmdp, pi)
                mle_cdice_r, mle_cdice_c = v_r[0], v_c[0][0]
                mle_cdice_roi = mle_cdice_r / mle_cdice_c

if __name__ == "__main__":
    app.run(main)