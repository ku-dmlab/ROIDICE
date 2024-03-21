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

"""Main experiment script."""

import time

from absl import app
from absl import flags
from absl import logging
import numpy as np
import pandas as pd

import sys, os

sys.path.append("/workspaces/ROIDICE")

from tabular import mdp_util
from tabular import offline_cmdp


flags.DEFINE_float("cost_thresholds", 0.1, "The cost constraint threshold of the true CMDP.")
flags.DEFINE_float(
    "behavior_optimality",
    0.9,
    "The optimality of data-collecting policy in terms of reward. "
    "(0: performance of uniform policy. 1: performance of optimal policy).",
)
flags.DEFINE_float("behavior_cost_thresholds", 0.1, "Set the cost value of data-collecting policy.")
flags.DEFINE_integer("num_iterations", 10, "The number of iterations for the repeated experiments.")

flags.DEFINE_integer("num_trajectories", 1000, "The number of trajectories to collect.")
flags.DEFINE_float("alpha", 0.001, "Alpha value.")
flags.DEFINE_float("target_roi", 5.0, "Target ROI value.")

flags.DEFINE_bool("basic_rl", False, "Whether run basic rl algorithm.")
flags.DEFINE_bool("unconstrained", False, "Whether run unconstrained optidice.")
flags.DEFINE_bool("vanilla_constrained", False, "Whether run vanilla constrained optidice.")
flags.DEFINE_bool(
    "conservative_constrained", False, "Whether run conservative constrained optidice."
)
flags.DEFINE_bool("roidice", True, "Whether run ROIDICE.")

FLAGS = flags.FLAGS

save_path = "./results/"

# logging color
grey = "\x1b[38;20m"
yellow = "\x1b[33;20m"
green = "\x1b[32;20m"
red = "\x1b[31;20m"
bold_red = "\x1b[31;1m"
reset = "\x1b[0m"

keys = [
    "uopt_r",
    "uopt_c",
    "opt_r",
    "opt_c",
    "behav_r",
    "behav_c",
    "basic_r",
    "basic_c",
    "basic_roi",
    "ucdice_r",
    "ucdice_c",
    "ucdice_roi",
    "cdice_r",
    "cdice_c",
    "cdice_roi",
    "ccdice_r",
    "ccdice_c",
    "ccdice_roi",
    "roidice_r",
    "roidice_c",
    "roidice_roi",
    "target_roi",
    "true_r",
    "true_c",
    "true_roi",
    "seed",
    "num_trajectories",
    "elapsed_time",
]

def main(unused_argv):
    """Main function."""
    num_states, num_actions, num_costs, gamma = 50, 4, 1, 0.95
    cost_thresholds = np.ones(num_costs) * FLAGS.cost_thresholds
    behavior_optimality = FLAGS.behavior_optimality
    behavior_cost_thresholds = np.array([FLAGS.behavior_cost_thresholds])

    logging.info(yellow + "==============================")
    logging.info("Cost threshold: %g", cost_thresholds)
    logging.info("Behavior optimality: %g", behavior_optimality)
    logging.info("Behavior cost thresholds: %g", behavior_cost_thresholds)
    logging.info("==============================" + reset)

    start_time = time.time()
    results = {k: [] for k in keys}
    # Construct a random CMDP
    for seed in range(FLAGS.num_iterations):
        np.random.seed(seed)
        cmdp = mdp_util.generate_random_cmdp(
            num_states, num_actions, num_costs, cost_thresholds, gamma
        )

        result = {}
        # Optimal policy for unconstrained MDP
        pi_uopt, _, _ = mdp_util.solve_mdp(cmdp)
        v_r, _, v_c, _ = mdp_util.policy_evaluation(cmdp, pi_uopt)
        uopt_r, uopt_c = v_r[0], v_c[0][0]
        results["uopt_r"].append(uopt_r)
        results["uopt_c"].append(uopt_c)
        # result.update({"uopt_r": uopt_r, "uopt_c": uopt_c})

        # Optimal policy for constrained MDP
        pi_copt = mdp_util.solve_cmdp(cmdp)
        v_r, _, v_c, _ = mdp_util.policy_evaluation(cmdp, pi_copt)
        opt_r, opt_c = v_r[0], v_c[0][0]
        results["opt_r"].append(opt_c)
        results["opt_c"].append(opt_r)
        # result.update({"opt_r": opt_r, "opt_c": opt_c})

        # Construct behavior policy
        pi_b = offline_cmdp.generate_baseline_policy(
            cmdp, behavior_cost_thresholds=behavior_cost_thresholds, optimality=behavior_optimality
        )
        v_r, _, v_c, _ = mdp_util.policy_evaluation(cmdp, pi_b)
        pib_r, pib_c = v_r[0], v_c[0][0]
        results["behav_r"].append(pib_r)
        results["behav_c"].append(pib_c)
        # result.update({"behav_r": pib_r, "behav_c": pib_c})

        alpha = FLAGS.alpha
        target_roi = FLAGS.target_roi
        for num_trajectories in [FLAGS.num_trajectories]:
            logging.info(bold_red + "==========================" + reset)
            logging.info(red + "* seed=%d, num_trajectories=%d" + reset, seed, num_trajectories)

            # Generate trajectory
            trajectory = mdp_util.generate_trajectory(
                seed, cmdp, pi_b, num_episodes=num_trajectories
            )

            # MLE CMDP
            mle_cmdp, _ = mdp_util.compute_mle_cmdp(
                num_states,
                num_actions,
                num_costs,
                cmdp.reward,
                cmdp.costs,
                cost_thresholds,
                gamma,
                trajectory,
            )

            # Basic RL
            if FLAGS.basic_rl:
                pi = mdp_util.solve_cmdp(mle_cmdp)
                v_r, _, v_c, _ = mdp_util.policy_evaluation(cmdp, pi)
                basic_r = v_r[0]
                basic_c = v_c[0][0]
                true_roi_basic = basic_r / basic_c
                results["basic_r"].append(basic_r)
                results["basic_c"].append(basic_c)
                results["basic_roi"].append(true_roi_basic)
                # result.update({"basic_r": basic_r, "basic_c": basic_c, "basic_roi": true_roi_basic})

            # UnconstrainedOptiDICE
            if FLAGS.unconstrained:
                _, _, pi, off_eval_r = offline_cmdp.optidice(mle_cmdp, pi_b, alpha)
                v_r, _, v_c, _ = mdp_util.policy_evaluation(cmdp, pi)
                optidice_r = v_r[0]
                optidice_c = v_c[0][0]
                true_roi_optidice = optidice_r / optidice_c
                results["ucdice_r"].append(optidice_r)
                results["ucdice_c"].append(optidice_c)
                results["ucdice_roi"].append(true_roi_optidice)
                # result.update(
                #     {
                #         "ucdice_r": optidice_r,
                #         "ucdice_c": optidice_c,
                #         "ucdice_roi": true_roi_optidice,
                #     }
                # )

            # Vanilla ConstrainedOptiDICE
            if FLAGS.vanilla_constrained:
                pi, _, _ = offline_cmdp.constrained_optidice(mle_cmdp, pi_b, alpha)
                v_r, _, v_c, _ = mdp_util.policy_evaluation(cmdp, pi)
                cdice_r = v_r[0]
                cdice_c = v_c[0][0]
                true_roi_cdice = cdice_r / cdice_c
                results["cdice_r"].append(cdice_r)
                results["cdice_c"].append(cdice_c)
                results["cdice_roi"].append(true_roi_cdice)
                # result.update({"cdice_r": cdice_r, "cdice_c": cdice_c, "cdice_roi": true_roi_cdice})

            # Conservative ConstrainedOptiDICE
            if FLAGS.conservative_constrained:
                epsilon = 0.1 / num_trajectories
                pi = offline_cmdp.conservative_constrained_optidice(  # compute upper bound
                    mle_cmdp, pi_b, alpha=alpha, epsilon=epsilon
                )
                v_r, _, v_c, _ = mdp_util.policy_evaluation(cmdp, pi)
                ccdice_r = v_r[0]
                ccdice_c = v_c[0][0]
                true_roi_ccdice = ccdice_r / ccdice_c
                results["ccdice_r"].append(ccdice_r)
                results["ccdice_c"].append(ccdice_c)
                results["ccdice_roi"].append(true_roi_ccdice)
                # result.update(
                #     {"ccdice_r": ccdice_r, "ccdice_c": ccdice_c, "ccdice_roi": true_roi_ccdice}
                # )

            # ROIDICE
            if FLAGS.roidice:
                pi, off_eval_r, off_eval_c = offline_cmdp.roidice(mle_cmdp, pi_b, alpha, target_roi)
                off_roi = off_eval_r / off_eval_c
                v_r, _, v_c, _ = mdp_util.policy_evaluation(cmdp, pi)
                true_r = v_r[0]
                true_c = v_c[0][0]
                true_roi = true_r / true_c
                results["roidice_r"].append(off_eval_r)
                results["roidice_c"].append(off_eval_c)
                results["roidice_roi"].append(off_roi)
                results["target_roi"].append(target_roi)
                # result.update(
                #     {
                #         "roidice_r": off_eval_r,
                #         "roidice_c": off_eval_c,
                #         "roidice_roi": off_roi,
                #         "target_roi": target_roi,
                #     }
                # )
                results["true_r"].append(true_r)
                results["true_c"].append(true_c)
                results["true_roi"].append(true_roi)
                result.update({"true_r": true_r, "true_c": true_c, "true_roi": true_roi})

            # Print the result
            elapsed_time = time.time() - start_time
            results["seed"].append(seed)
            results["num_trajectories"].append(num_trajectories)
            results["elapsed_time"].append(elapsed_time)
            # result.update(
            #     {
            #         "seed": seed,
            #         "num_trajectories": num_trajectories,
            #         "elapsed_time": elapsed_time,
            #     }
            # )
            logging.info(bold_red + f"{result}" + reset)

    df = pd.DataFrame.from_dict(results)
    df.to_csv(os.path.join(save_path, f"alpha{alpha}_target_roi{target_roi}.csv"), index=True)

    logging.info(
        green
        + f"Done with options (alpha: {alpha}, target_roi: {target_roi}, seed: {seed})"
        + reset
    )


if __name__ == "__main__":
    app.run(main)
