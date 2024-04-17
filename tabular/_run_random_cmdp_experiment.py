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
flags.DEFINE_float("lamb_lower", 1.0, "Lower bound of ROI value.")
flags.DEFINE_float("beta", 0.1, "Beta value.")

flags.DEFINE_integer("basic_rl", 0, "Whether run basic rl algorithm.")
flags.DEFINE_integer("unconstrained", 0, "Whether run unconstrained optidice.")
flags.DEFINE_integer("vanilla_constrained", 0, "Whether run vanilla constrained optidice.")
flags.DEFINE_integer(
    "conservative_constrained", 0, "Whether run conservative constrained optidice."
)
flags.DEFINE_integer("roidice", 0, "Whether run ROIDICE.")
flags.DEFINE_integer("roidice_lower_bound", 0, "Whether run Lower bound ROIDICE.")
flags.DEFINE_integer("max_roidice", 0, "Whether run Maximize ROIDICE.")
flags.DEFINE_integer("max_roidice_reg", 0, "Whether run Maximize ROIDICE with regularization.")

flags.DEFINE_multi_string("path", ".", "Save directory name")

FLAGS = flags.FLAGS

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
    "lower_roidice_r",
    "lower_roidice_c",
    "lower_roidice_roi",
    "lower_true_r",
    "lower_true_c",
    "lower_true_roi",
    "max_roidice_r",
    "max_roidice_c",
    "max_roidice_roi",
    "max_lamb",
    "max_success",
    "max_true_r",
    "max_true_c",
    "max_true_roi",
    "max_reg_roidice_r",
    "max_reg_roidice_c",
    "max_reg_roidice_roi",
    "max_reg_lamb",
    "max_reg_true_r",
    "max_reg_true_c",
    "max_reg_true_roi",
    "seed",
    "num_trajectories",
    "elapsed_time",
]


def main(unused_argv):
    save_path = f"./results/tabular/{FLAGS.path[0]}/"
    os.makedirs(save_path, exist_ok=True)

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
        additional_name = ""
        np.random.seed(seed)
        cmdp = mdp_util.generate_random_cmdp(
            num_states, num_actions, num_costs, cost_thresholds, gamma
        )

        # Optimal policy for unconstrained MDP
        pi_uopt, _, _ = mdp_util.solve_mdp(cmdp)
        v_r, _, v_c, _ = mdp_util.policy_evaluation(cmdp, pi_uopt)
        uopt_r, uopt_c = v_r[0], v_c[0][0]
        uopt_roi = uopt_r / uopt_c

        # Optimal policy for constrained MDP
        pi_copt = mdp_util.solve_cmdp(cmdp)
        # if pi_copt.any() == None:
        #     continue # skip the seed solver cannot solve
        v_r, _, v_c, _ = mdp_util.policy_evaluation(cmdp, pi_copt)
        opt_r, opt_c = v_r[0], v_c[0][0]

        # Construct behavior policy
        pi_b = offline_cmdp.generate_baseline_policy(
            cmdp, behavior_cost_thresholds=behavior_cost_thresholds, optimality=behavior_optimality
        )
        v_r, _, v_c, _ = mdp_util.policy_evaluation(cmdp, pi_b)
        pib_r, pib_c = v_r[0], v_c[0][0]

        alpha = FLAGS.alpha
        target_roi = FLAGS.target_roi
        lamb_lower = FLAGS.lamb_lower
        beta = FLAGS.beta
        for num_trajectories in [FLAGS.num_trajectories]:
            logging.info(bold_red + "==========================" + reset)
            logging.info(
                red + "* alpha=%f, target_roi=%f, seed=%d, num_trajectories=%d" + reset,
                alpha,
                target_roi,
                seed,
                num_trajectories,
            )

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
            basic_r, basic_c, true_roi_basic = 0.0, 0.0, 0.0
            if FLAGS.basic_rl:
                logging.info(yellow + "Run Basic RL" + reset)
                pi = mdp_util.solve_cmdp(mle_cmdp)
                v_r, _, v_c, _ = mdp_util.policy_evaluation(cmdp, pi)
                basic_r = v_r[0]
                basic_c = v_c[0][0]
                true_roi_basic = basic_r / basic_c

            # UnconstrainedOptiDICE
            optidice_r, optidice_c, true_roi_optidice = 0.0, 0.0, 0.0
            if FLAGS.unconstrained:
                logging.info(yellow + "Run Unconstrained OptiDICE" + reset)
                _, _, pi, off_eval_r = offline_cmdp.optidice(mle_cmdp, pi_b, alpha)
                v_r, _, v_c, _ = mdp_util.policy_evaluation(cmdp, pi)
                optidice_r = v_r[0]
                optidice_c = v_c[0][0]
                true_roi_optidice = optidice_r / optidice_c

            # Vanilla ConstrainedOptiDICE
            cdice_r, cdice_c, true_roi_cdice = 0.0, 0.0, 0.0
            if FLAGS.vanilla_constrained:
                logging.info(yellow + "Run Vanilla Constrained OptiDICE" + reset)
                pi, _, _ = offline_cmdp.constrained_optidice(mle_cmdp, pi_b, alpha)
                v_r, _, v_c, _ = mdp_util.policy_evaluation(cmdp, pi)
                cdice_r = v_r[0]
                cdice_c = v_c[0][0]
                true_roi_cdice = cdice_r / cdice_c

            # Conservative ConstrainedOptiDICE
            ccdice_r, ccdice_c, true_roi_ccdice = 0.0, 0.0, 0.0
            if FLAGS.conservative_constrained:
                logging.info(yellow + "Run Conservative Constrained OptiDICE" + reset)
                epsilon = 0.1 / num_trajectories
                pi = offline_cmdp.conservative_constrained_optidice(  # compute upper bound
                    mle_cmdp, pi_b, alpha=alpha, epsilon=epsilon
                )
                v_r, _, v_c, _ = mdp_util.policy_evaluation(cmdp, pi)
                ccdice_r = v_r[0]
                ccdice_c = v_c[0][0]
                true_roi_ccdice = ccdice_r / ccdice_c

            # ROIDICE
            off_eval_r, off_eval_c, off_roi = 0.0, 0.0, 0.0
            true_r, true_c, true_roi = 0.0, 0.0, 0.0
            if FLAGS.roidice:
                logging.info(yellow + "Run ROIDICE" + reset)
                pi, off_eval_r, off_eval_c = offline_cmdp.roidice(mle_cmdp, pi_b, alpha, target_roi)
                off_roi = off_eval_r / off_eval_c
                v_r, _, v_c, _ = mdp_util.policy_evaluation(cmdp, pi)
                true_r = v_r[0]
                true_c = v_c[0][0]
                true_roi = true_r / true_c

            # ROIDICE with lower bound constraint
            off_eval_lower_r, off_eval_lower_c, off_lower_roi = 0.0, 0.0, 0.0
            true_lower_r, true_lower_c, true_lower_roi = 0.0, 0.0, 0.0
            if FLAGS.roidice_lower_bound:
                additional_name += f"_lamb_lower{lamb_lower}"
                logging.info(yellow + "Run lower bound ROIDICE" + reset)
                pi, off_eval_lower_r, off_eval_lower_c = offline_cmdp.roidice_lower_bound(
                    mle_cmdp, pi_b, alpha, lamb_lower
                )
                off_lower_roi = off_eval_lower_r / off_eval_lower_c
                v_r, _, v_c, _ = mdp_util.policy_evaluation(cmdp, pi)
                true_lower_r = v_r[0]
                true_lower_c = v_c[0][0]
                true_lower_roi = true_lower_r / true_lower_c

            # Maximize ROI
            max_off_eval_r, max_off_eval_c, max_off_roi = 0.0, 0.0, 0.0
            max_lamb = 0.0
            max_true_r, max_true_c, max_true_roi = 0.0, 0.0, 0.0
            if FLAGS.max_roidice:
                logging.info(yellow + "Run Mamimize ROIDICE" + reset)
                pi, max_off_eval_r, max_off_eval_c, max_lamb, max_success = (
                    offline_cmdp.roidice_max_roi(mle_cmdp, pi_b, alpha)
                )
                max_off_roi = max_off_eval_r / max_off_eval_c
                v_r, _, v_c, _ = mdp_util.policy_evaluation(cmdp, pi)
                max_true_r = v_r[0]
                max_true_c = v_c[0][0]
                max_true_roi = max_true_r / max_true_c

            # Maximize ROI with additional regularization
            max_off_eval_reg_r, max_off_eval_reg_c, max_off_reg_roi = 0.0, 0.0, 0.0
            max_reg_lamb = 0.0
            max_true_reg_r, max_true_reg_c, max_true_reg_roi = 0.0, 0.0, 0.0
            if FLAGS.max_roidice_reg:
                additional_name += f"_beta{beta}"
                logging.info(yellow + "Run Mamimize ROIDICE with Regularization" + reset)
                pi, max_off_eval_reg_r, max_off_eval_reg_c, max_reg_lamb, max_reg_success = (
                    offline_cmdp.roidice_max_roi_reg(mle_cmdp, pi_b, alpha, beta)
                )
                max_off_reg_roi = max_off_eval_reg_r / max_off_eval_reg_c
                v_r, _, v_c, _ = mdp_util.policy_evaluation(cmdp, pi)
                max_true_reg_r = v_r[0]
                max_true_reg_c = v_c[0][0]
                max_true_reg_roi = max_true_reg_r / max_true_reg_c

            # log results
            results["uopt_r"].append(uopt_r)
            results["uopt_c"].append(uopt_c)
            results["opt_r"].append(opt_c)
            results["opt_c"].append(opt_r)
            results["behav_r"].append(pib_r)
            results["behav_c"].append(pib_c)
            # basic RL
            results["basic_r"].append(basic_r)
            results["basic_c"].append(basic_c)
            results["basic_roi"].append(true_roi_basic)
            # unconstrained optidice
            results["ucdice_r"].append(optidice_r)
            results["ucdice_c"].append(optidice_c)
            results["ucdice_roi"].append(true_roi_optidice)
            # constrained optidice
            results["cdice_r"].append(cdice_r)
            results["cdice_c"].append(cdice_c)
            results["cdice_roi"].append(true_roi_cdice)
            # conservative constrained optidice
            results["ccdice_r"].append(ccdice_r)
            results["ccdice_c"].append(ccdice_c)
            results["ccdice_roi"].append(true_roi_ccdice)
            # roidice
            results["roidice_r"].append(off_eval_r)
            results["roidice_c"].append(off_eval_c)
            results["roidice_roi"].append(off_roi)
            results["target_roi"].append(target_roi)
            results["true_r"].append(true_r)
            results["true_c"].append(true_c)
            results["true_roi"].append(true_roi)
            # roidice with lower bound
            results["lower_roidice_r"].append(off_eval_lower_r)
            results["lower_roidice_c"].append(off_eval_lower_c)
            results["lower_roidice_roi"].append(off_lower_roi)
            results["lower_true_r"].append(true_lower_r)
            results["lower_true_c"].append(true_lower_c)
            results["lower_true_roi"].append(true_lower_roi)
            # ROI maximize
            results["max_roidice_r"].append(max_off_eval_r)
            results["max_roidice_c"].append(max_off_eval_c)
            results["max_roidice_roi"].append(max_off_roi)
            results["max_lamb"].append(max_lamb)
            results["max_success"].append(max_success)
            results["max_true_r"].append(max_true_r)
            results["max_true_c"].append(max_true_c)
            results["max_true_roi"].append(max_true_roi)
            # Maximize ROIDICE with reg
            results["max_reg_roidice_r"].append(max_off_eval_reg_r)
            results["max_reg_roidice_c"].append(max_off_eval_reg_c)
            results["max_reg_roidice_roi"].append(max_off_reg_roi)
            results["max_reg_lamb"].append(max_reg_lamb)
            results["max_reg_true_r"].append(max_true_reg_r)
            results["max_reg_true_c"].append(max_true_reg_c)
            results["max_reg_true_roi"].append(max_true_reg_roi)
            # Print the result
            elapsed_time = time.time() - start_time
            results["seed"].append(seed)
            results["num_trajectories"].append(num_trajectories)
            results["elapsed_time"].append(elapsed_time)

            # logging.info(bold_red + f"{results}" + reset)

    df = pd.DataFrame.from_dict(results)
    df.to_csv(
        os.path.join(
            save_path,
            f"alpha{alpha}_cost_thresholds{FLAGS.cost_thresholds}{additional_name}.csv",
        ),
        index=True,
    )

    logging.info(
        green
        + f"Done with options (alpha: {alpha}, target_roi: {target_roi}, cost_thresholds: {FLAGS.cost_thresholds}, seed: {seed})"
        + reset
    )


if __name__ == "__main__":
    app.run(main)
