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

"""Implementation of tabular offline (C)MDP methods."""

from absl import logging
import cvxopt
import cvxpy as cp
import jax
import numpy as np
import scipy
import scipy.optimize
import mdp_util as util

cvxopt.solvers.options["show_progress"] = False


def _compute_marginal_distribution(mdp, pi, regularizer=0):
    """Compute marginal distribution for the given policy pi, d^pi(s,a)."""
    p0_s = np.zeros(mdp.num_states)
    p0_s[mdp.initial_state] = 1
    p0 = (p0_s[:, None] * pi).reshape(mdp.num_states * mdp.num_actions)
    p_pi = (
        mdp.transition.reshape(mdp.num_states * mdp.num_actions, mdp.num_states)[:, :, None] * pi
    ).reshape(mdp.num_states * mdp.num_actions, mdp.num_states * mdp.num_actions)
    d = np.ones(mdp.num_states * mdp.num_actions)
    d /= np.sum(d)
    d_diag = np.diag(d)
    e = np.sqrt(d_diag) @ (np.eye(mdp.num_states * mdp.num_actions) - mdp.gamma * p_pi)

    q = np.linalg.solve(
        e.T @ e + regularizer * np.eye(mdp.num_states * mdp.num_actions), (1 - mdp.gamma) * p0
    )
    w = q - mdp.gamma * p_pi @ q

    assert np.all(w > -1e-6), w
    d_pi = w * d
    d_pi[w < 0] = 0
    d_pi /= np.sum(d_pi)
    return d_pi.reshape(mdp.num_states, mdp.num_actions)


def generate_baseline_policy(cmdp: util.CMDP, optimality: float, verbose: bool=False) -> np.ndarray:
    """Generate a baseline policy for the CMDP.

    Args:
      cmdp: a CMDP instance.
      optimality: optimality of behavior policy.
        (0: uniform policy, 1: optimal policy)

    Returns:
      behavior policy. [num_states, num_actions]
    """

    pi_opt, _, _ = util.solve_mdp(cmdp)
    pi_unif = np.ones((cmdp.num_states, cmdp.num_actions)) / cmdp.num_actions
    v_opt = util.policy_evaluation(cmdp, pi_opt)[0][0]
    q_opt = util.policy_evaluation(cmdp, pi_opt)[1]
    v_unif = util.policy_evaluation(cmdp, pi_unif)[0][0]

    v_final_target = v_opt * optimality + (1 - optimality) * v_unif

    softmax_reduction_factor = 0.9
    temperature = 1e-6
    pi_soft = scipy.special.softmax(q_opt / temperature, axis=1)
    while util.policy_evaluation(cmdp, pi_soft)[0][0] > v_final_target:
        temperature /= softmax_reduction_factor
        pi_soft = scipy.special.softmax(q_opt / temperature, axis=1)
        pi_soft /= np.sum(pi_soft, axis=1, keepdims=True)
        r, _, c, _ = util.policy_evaluation(cmdp, pi_soft)

        if verbose:
            logging.info(
                "temp=%.6f, R=%.3f, C=%.3f / v_opt=%.3f, f_target=%.3f",
                temperature,
                r[0],
                c[0][0],
                v_opt,
                v_final_target,
            )

    assert np.all(pi_soft >= -1e-4)

    pi_b = pi_soft.copy()
    return pi_b


def optidice(mdp: util.MDP, pi_b: np.ndarray, alpha: float):
    """f-divergence regularized RL.

    max_{d} E_d[R(s,a)] - alpha * E_{d_b}[f(d(s,a)/d_b(s,a))]

    We assume that f(x) = 0.5 (x-1)^2.

    Args:
      mdp: a MDP instance.
      pi_b: behavior policy. [num_states, num_actions]
      alpha: regularization hyperparameter for f-divergence.

    Returns:
      the resulting policy. [num_states, num_actions]
    """
    d_b = (
        _compute_marginal_distribution(mdp, pi_b).reshape(mdp.num_states * mdp.num_actions) + 1e-6
    )  # |S||A|
    d_b /= np.sum(d_b)
    p0 = np.eye(mdp.num_states)[mdp.initial_state]  # |S|
    r = np.array(mdp.reward.reshape(mdp.num_states * mdp.num_actions))
    p = np.array(mdp.transition.reshape(mdp.num_states * mdp.num_actions, mdp.num_states))
    p = p / np.sum(p, axis=1, keepdims=True)
    b = np.repeat(np.eye(mdp.num_states), mdp.num_actions, axis=0)  # |S||A| x |S|

    # Solve:
    # minimize    (1/2)*x^T P x + q^T x
    # subject to  G x <= h
    #             A x = b.
    d_diag = np.diag(d_b)
    qp_p = alpha * (d_diag)
    qp_q = -d_diag @ r - alpha * d_b
    qp_g = -np.eye(mdp.num_states * mdp.num_actions)
    qp_h = np.zeros(mdp.num_states * mdp.num_actions)
    qp_a = (b.T - mdp.gamma * p.T) @ d_diag
    qp_b = (1 - mdp.gamma) * p0
    cvxopt.solvers.options["show_progress"] = False
    res = cvxopt.solvers.qp(
        cvxopt.matrix(qp_p),
        cvxopt.matrix(qp_q),
        cvxopt.matrix(qp_g),
        cvxopt.matrix(qp_h),
        cvxopt.matrix(qp_a),
        cvxopt.matrix(qp_b),
    )
    w = np.array(res["x"])[:, 0]  # [num_states * num_actions]
    assert np.all(w >= -1e-4), w
    w = np.clip(w, 1e-10, np.inf)

    off_eval_r = np.sum(w * d_b * r)

    pi = (w * d_b).reshape(mdp.num_states, mdp.num_actions) + 1e-10
    pi /= np.sum(pi, axis=1, keepdims=True)

    return w, d_b, pi, off_eval_r


def constrained_optidice(cmdp: util.CMDP, pi_b: np.ndarray, alpha: float):
    """f-divergence regularized constrained RL.

    max_{d} E_d[R(s,a)] - alpha * E_{d_b}[f(d(s,a)/d_b(s,a))]
    s.t. E_d[C(s,a)] <= hat{c}.

    We assume that f(x) = 0.5 (x-1)^2.

    Args:
      cmdp: a CMDP instance.
      pi_b: behavior policy.
      alpha: regularization hyperparameter for f-divergence.

    Returns:
      the resulting policy. [num_states, num_actions]
    """
    d_b = (
        _compute_marginal_distribution(cmdp, pi_b).reshape(cmdp.num_states * cmdp.num_actions)
        + 1e-6
    )  # |S||A|
    d_b /= np.sum(d_b)
    p0 = np.eye(cmdp.num_states)[cmdp.initial_state]  # |S|
    p = np.array(cmdp.transition.reshape(cmdp.num_states * cmdp.num_actions, cmdp.num_states))
    p = p / np.sum(p, axis=1, keepdims=True)
    b = np.repeat(np.eye(cmdp.num_states), cmdp.num_actions, axis=0)  # |S||A| x |S|
    r = np.array(cmdp.reward.reshape(cmdp.num_states * cmdp.num_actions))
    c = np.array(cmdp.costs.reshape(cmdp.num_costs, cmdp.num_states * cmdp.num_actions))

    # Solve:
    # minimize    (1/2)*x^T P x + q^T x
    # subject to  G x <= h
    #             A x = b.
    d_diag = np.diag(d_b)
    qp_p = alpha * (d_diag)
    qp_q = -d_diag @ r - alpha * d_b
    qp_g = np.concatenate([c @ d_diag, -np.eye(cmdp.num_states * cmdp.num_actions)], axis=0)
    qp_h = np.concatenate(
        [np.array([cmdp.cost_thresholds]), np.zeros(cmdp.num_states * cmdp.num_actions)]
    )
    qp_a = (b.T - cmdp.gamma * p.T) @ d_diag
    qp_b = (1 - cmdp.gamma) * p0
    res = cvxopt.solvers.qp(
        cvxopt.matrix(qp_p),
        cvxopt.matrix(qp_q),
        cvxopt.matrix(qp_g),
        cvxopt.matrix(qp_h),
        cvxopt.matrix(qp_a),
        cvxopt.matrix(qp_b),
    )

    if res["status"] != "optimal":
        return None, -1, -1

    w = np.array(res["x"])[:, 0]  # [num_states * num_actions]
    assert np.all(w >= -1e-4), w
    w = np.clip(w, 1e-10, np.inf)

    off_eval_r = np.sum(w * d_b * r)
    off_eval_c = np.sum(w * d_b * c[0])

    pi = (w * d_b).reshape(cmdp.num_states, cmdp.num_actions) + 1e-10
    pi /= np.sum(pi, axis=1, keepdims=True)
    assert np.all(pi >= -1e-6), pi

    return np.array(pi), off_eval_r, off_eval_c

def roidice_max_roi_mosek_inf(cmdp: util.CMDP, pi_b: np.ndarray, alpha: float):
    d_b = (
        _compute_marginal_distribution(cmdp, pi_b).reshape(cmdp.num_states * cmdp.num_actions)
        + 1e-4
    )  # |S||A|
    d_b /= np.sum(d_b)

    p0 = np.eye(cmdp.num_states)[cmdp.initial_state]  # |S|
    p = np.array(cmdp.transition.reshape(cmdp.num_states * cmdp.num_actions, cmdp.num_states))
    p = p / np.sum(p, axis=1, keepdims=True)

    b = np.repeat(np.eye(cmdp.num_states), cmdp.num_actions, axis=0)  # |S||A| x |S|
    r = np.array(cmdp.reward.reshape(cmdp.num_states * cmdp.num_actions))  # |S||A|
    c = np.array(cmdp.costs.reshape(cmdp.num_costs, cmdp.num_states * cmdp.num_actions))
    d_diag = np.diag(d_b)
    x = cp.Variable(cmdp.num_states * cmdp.num_actions, nonneg=True)
    t = cp.Variable(1)
    prob = cp.Problem(
        cp.Minimize(-r.T @ d_diag @ x + 0.5 * alpha * d_b.T @ cp.power(x - t, 2)),
        [
            (b.T - cmdp.gamma * p.T) @ d_diag @ x == (1 - cmdp.gamma) * p0 * t,
            c[0].T @ d_diag @ x == 1,
            x >= 0,
            t >= 0,
        ],
    )
    try:
        prob.solve(solver=cp.MOSEK)
    except cp.error.SolverError:
        return None, -1, -1, -1
    except scipy.sparse.linalg._eigen.arpack.ArpackNoConvergence:
        try:
            prob.solve(solver=cp.ECOS)
        except cp.error.SolverError:
            return None, -1, -1, -1
    w = x.value / t.value

    assert np.all(w >= -1e-4), w
    w = np.clip(w, 1e-10, np.inf)

    # off-policy evaluation
    off_eval_r = np.sum(w * d_b * r)
    off_eval_c = np.sum(w * d_b * c[0])

    # policy extraction
    pi = (w * d_b).reshape(cmdp.num_states, cmdp.num_actions) + 1e-10
    pi /= np.sum(pi, axis=1, keepdims=True)
    assert np.all(pi >= -1e-6), pi

    return np.array(pi), off_eval_r, off_eval_c, t.value