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

keys = [
    "true_uopt_r", "true_uopt_c", "true_uopt_roi",
    "true_roi_inf_r", "true_roi_inf_c", "true_roi_inf_roi",
    "true_behav_r", "true_behav_c", "true_behav_roi",
    "uopt_r", "uopt_c", "uopt_roi",
    "roi_inf_wo_reg_r", "roi_inf_wo_reg_c", "roi_inf_wo_reg_roi",
    "odice_r", "odice_c", "odice_roi",
    "roi_inf_r", "roi_inf_c", "roi_inf_roi",
    "roi_r", "roi_c", "roi_roi",
    "cdice_r", "cdice_c", "cdice_roi",
    "seed", "alpha", "cost_multiplier"
]

def main(unused_argv):
    """Main function."""
    num_states, num_actions, num_costs, gamma = 50, 4, 1, 0.95
    behavior_optimality = FLAGS.behavior_optimality

    logging.info("==============================")
    logging.info("Behavior optimality: %g", behavior_optimality)
    logging.info("==============================")

    alphas = [0.001, 0.01, 0.1, 1.0]
    cost_multipliers = [0.7, 0.9, 1.0, 1.1, 1.3]

    cost_multipliers = [0.7]

    results = {k: [] for k in keys}
    indices = []

    f = open("./results/infeasible_seeds_cdice.txt", "w")

    # for seed in range(FLAGS.num_iterations):
    for seed in [5, 155, 163, 224, 240, 347, 405, 462, 518, 559, 651, 658, 686, 692, 698, 298, 711, 716, 735, 744, 791, 829, 914, 940, 977]:
        # Construct a random CMDP
        np.random.seed(seed)
        cmdp = mdp_util.generate_random_cmdp(num_states, num_actions, num_costs, gamma)

        # Optimal policy for unconstrained MDP
        pi_uopt, _, _ = mdp_util.solve_mdp(cmdp)
        v_r, _, v_c, _ = mdp_util.policy_evaluation(cmdp, pi_uopt)
        true_uopt_r, true_uopt_c = v_r[0], v_c[0][0]
        true_uopt_roi = true_uopt_r / true_uopt_c

        pi_b = offline_cmdp.generate_baseline_policy(cmdp, optimality=behavior_optimality)
        pi, _, _ = offline_cmdp.roidice_max_roi_mosek_inf(cmdp, pi_b, alpha=0.0)
        if pi is None:
            f.write(f"inf seed: {seed}\n")
            continue
        v_r, _, v_c, _ = mdp_util.policy_evaluation(cmdp, pi)
        true_roi_inf_r, true_roi_inf_c = v_r[0], v_c[0][0]
        true_roi_inf_roi = true_roi_inf_r / true_roi_inf_c
        
        v_r, _, v_c, _ = mdp_util.policy_evaluation(cmdp, pi_b)
        true_behav_r, true_behav_c = v_r[0], v_c[0][0]
        true_behav_roi = true_behav_r / true_behav_c
        
        for num_trajectories in [1000]:
            logging.info("==========================")
            logging.info("* seed=%d, num_trajectories=%d", seed, num_trajectories)
            # Generate trajectory
            trajectory = mdp_util.generate_trajectory(seed, cmdp, pi_b, num_episodes=num_trajectories)

            # MLE CMDP
            mle_cmdp, _ = mdp_util.compute_mle_cmdp(
                num_states, num_actions, num_costs, cmdp.reward, cmdp.costs, gamma, trajectory
            )

            pi_uopt, _, _ = mdp_util.solve_mdp(mle_cmdp)
            v_r, _, v_c, _ = mdp_util.policy_evaluation(cmdp, pi_uopt)
            uopt_r, uopt_c = v_r[0], v_c[0][0]
            uopt_roi = uopt_r / uopt_c

            pi, _, _ = offline_cmdp.roidice_max_roi_mosek_inf(mle_cmdp, pi_b, alpha=0.0) # seed 112
            if pi is None:
                f.write(f"mle inf seed: {seed}\n")
                continue
            v_r, _, v_c, _ = mdp_util.policy_evaluation(cmdp, pi)
            roi_inf_wo_reg_r, roi_inf_wo_reg_c = v_r[0], v_c[0][0]
            roi_inf_wo_reg_roi = roi_inf_wo_reg_r / roi_inf_wo_reg_c

            for alpha in alphas:
                logging.info(red + f"* seed={seed}, alpha={alpha}" + reset)
                _, _, pi, _ = offline_cmdp.optidice(mle_cmdp, pi_b, alpha)
                v_r, _, v_c, _ = mdp_util.policy_evaluation(cmdp, pi)
                odice_r = v_r[0]
                odice_c = v_c[0][0]
                odice_roi = odice_r / odice_c

                pi, _, _ = offline_cmdp.roidice_max_roi_mosek_inf(mle_cmdp, pi_b, alpha=alpha) # seed 189, alpha 0.001
                if pi is None:
                    f.write(f"mle inf alpha{alpha} seed: {seed}\n")
                    continue
                v_r, _, v_c, _ = mdp_util.policy_evaluation(cmdp, pi)
                roi_inf_r, roi_inf_c = v_r[0], v_c[0][0]
                roi_inf_roi = roi_inf_r / roi_inf_c

                cost_threshold = roi_inf_c
                for cost_multiplier in cost_multipliers:
                    logging.info(red + f"* seed={seed}, alpha={alpha}, cost_threshold={cost_threshold}, cost_multiplier={cost_multiplier}" + reset)
                    mle_cmdp.cost_thresholds = cost_threshold * cost_multiplier
                    # pi, _, _ = offline_cmdp.roidice_max_roi_mosek(mle_cmdp, pi_b, alpha=alpha)
                    # if pi is None:
                    #     f.write(f"mle alpha{alpha} seed: {seed} cost_multiplier: {cost_multiplier}\n")
                    #     continue
                    # v_r, _, v_c, _ = mdp_util.policy_evaluation(cmdp, pi)
                    # roi_r, roi_c = v_r[0], v_c[0][0]
                    # roi_roi = roi_r / roi_c

                    pi, _, _ = offline_cmdp.constrained_optidice(mle_cmdp, pi_b, alpha=alpha)
                    if pi is None:
                        f.write(f"mle cdice alpha{alpha} seed: {seed} cost_multiplier: {cost_multiplier}\n")
                        continue
                    v_r, _, v_c, _ = mdp_util.policy_evaluation(cmdp, pi)
                    cdice_r, cdice_c = v_r[0], v_c[0][0]
                    cdice_roi = cdice_r / cdice_c

                    indices.append(f"seed{seed}_alpha{alpha}_cost_multiplier{cost_multiplier}")
                    results["true_uopt_r"].append(true_uopt_r)
                    results["true_uopt_c"].append(true_uopt_c)
                    results["true_uopt_roi"].append(true_uopt_roi)
                    results["true_roi_inf_r"].append(true_roi_inf_r)
                    results["true_roi_inf_c"].append(true_roi_inf_c)
                    results["true_roi_inf_roi"].append(true_roi_inf_roi)
                    results["true_behav_r"].append(true_behav_r)
                    results["true_behav_c"].append(true_behav_c)
                    results["true_behav_roi"].append(true_behav_roi)
                    results["uopt_r"].append(uopt_r)
                    results["uopt_c"].append(uopt_c)
                    results["uopt_roi"].append(uopt_roi)
                    results["roi_inf_wo_reg_r"].append(roi_inf_wo_reg_r)
                    results["roi_inf_wo_reg_c"].append(roi_inf_wo_reg_c)
                    results["roi_inf_wo_reg_roi"].append(roi_inf_wo_reg_roi)
                    results["odice_r"].append(odice_r)
                    results["odice_c"].append(odice_c)
                    results["odice_roi"].append(odice_roi)
                    results["roi_inf_r"].append(roi_inf_r)
                    results["roi_inf_c"].append(roi_inf_c)
                    results["roi_inf_roi"].append(roi_inf_roi)
                    # results["roi_r"].append(roi_r)
                    # results["roi_c"].append(roi_c)
                    # results["roi_roi"].append(roi_roi)
                    results["cdice_r"].append(cdice_r)
                    results["cdice_c"].append(cdice_c)
                    results["cdice_roi"].append(cdice_roi)

                    results["seed"].append(seed)
                    results["alpha"].append(alpha)
                    results["cost_multiplier"].append(cost_multiplier)

    # df = pd.DataFrame.from_dict(results)
    # df.index = indices
    # df.to_csv(f"./results/tabular/roidice_seed{FLAGS.num_iterations}_cdice.csv", index=True)
    logging.info(yellow + "=> SAVE!!!" + reset)

    f.close()

if __name__ == "__main__":
    app.run(main)