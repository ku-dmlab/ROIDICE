from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
from jax import Array

import divergence
from common import Batch, ConstrainedBatch, InfoDict, Model, Params, PRNGKey
from divergence import FDivergence


def update_nu_state(
    batch: ConstrainedBatch,
    cost_lambda: Model,
    nu_state: Model,
    alpha: float,
    discount: float,
    gradient_penalty_coeff: float,
    f_divergence: FDivergence,
    rng: PRNGKey,
):
    if cost_lambda is None:
        cost_coeff = 0  # unconstrained
    else:
        cost_coeff = cost_lambda()

    @partial(jax.vmap, in_axes=(None, 0))
    @partial(jax.grad, argnums=1)
    def grad_nu(value_params, obs):
        return nu_state.apply({"params": value_params}, obs)

    def nu_loss_fn(params: Params) -> Tuple[Array, InfoDict]:
        initial_nu = nu_state.apply({"params": params}, batch.initial_observations)
        nu = nu_state.apply({"params": params}, batch.observations)
        next_nu = nu_state.apply({"params": params}, batch.next_observations)
        e = batch.rewards - cost_coeff * batch.costs + discount * next_nu - nu

        state_action_ratio = divergence.state_action_ratio(
            nu,
            next_nu,
            batch.rewards,
            batch.costs,
            alpha,
            cost_coeff,
            discount,
            f_divergence,
        )

        initial_loss = (1 - discount) * initial_nu
        non_initial_loss = state_action_ratio * e - alpha * divergence.f(
            state_action_ratio, f_divergence
        )

        interpolation_epsilon = jax.random.uniform(rng)
        interpolated_observations = (
            batch.initial_observations * interpolation_epsilon
            + batch.next_observations * (1 - interpolation_epsilon)
        )

        # regularization term
        value2_grad = grad_nu(params, interpolated_observations)
        value2_grad_norm = jnp.linalg.norm(value2_grad, axis=1)
        value2_grad_penalty = gradient_penalty_coeff * jnp.mean(
            jax.nn.relu(value2_grad_norm - 5) ** 2
        )

        return initial_loss.mean() + non_initial_loss.mean() + value2_grad_penalty, {
            "loss/nu(s0)": initial_loss.mean(),
            "loss/nu(s)": non_initial_loss.mean(),
            "loss/nu(s)_grad_penalty": value2_grad_penalty,
            "nu(s0)/mean": initial_nu.mean(),
            "nu(s0)/max": initial_nu.max(),
            "nu(s0)/min": initial_nu.min(),
            "nu(s)/mean": nu.mean(),
            "nu(s)/max": nu.max(),
            "nu(s)/min": nu.min(),
            "nu(s')/mean": next_nu.mean(),
            "nu(s')/max": next_nu.max(),
            "nu(s')/min": next_nu.min(),
            "w(s, a)/mean": state_action_ratio.mean(),
            "w(s, a)/max": state_action_ratio.max(),
            "w(s, a)/min": state_action_ratio.min(),
        }

    new_nu_state, info = nu_state.apply_gradient(nu_loss_fn)
    return new_nu_state, info


def update_w_state(
    batch: Batch,
    nu_state: Model,
    w_state: Model,
    alpha: float,
    discount: float,
    f_divergence: FDivergence,
):
    def w_loss_fn(params: Params) -> Tuple[Array, InfoDict]:
        nu = nu_state.apply({"params": params}, batch.observations)
        next_nu = nu_state.apply({"params": params}, batch.next_observations)

        cost_coeff = 0.0

        state_action_ratio = divergence.state_action_ratio(
            nu,
            next_nu,
            batch.rewards,
            jnp.zeros_like(batch.rewards),
            alpha,
            cost_coeff,
            discount,
            f_divergence,
        )

        w = w_state.apply({"params": params}, batch.observations)
        w_s = jnp.mean((state_action_ratio - w) ** 2)

        return w_s, {
            "w(s)/mean": w.mean(),
            "w(s)/max": w.max(),
            "w(s)/min": w.min(),
        }

    new_w_s, info = w_state.apply_gradient(w_loss_fn)
    return new_w_s, info


def update_cost_lambda(
    batch: ConstrainedBatch,
    cost_lambda: Model,
    nu_state: Model,
    alpha: float,
    discount: float,
    cost_limit: float,
    f_divergence: FDivergence,
):
    nu = nu_state(batch.observations)
    next_nu = nu_state(batch.next_observations)

    def lambda_loss_fn(params: Params) -> tuple[Array, InfoDict]:
        cost_coeff = cost_lambda.apply({"params": params})

        e = batch.rewards - cost_coeff * batch.costs + discount * next_nu - nu
        f_temp = divergence.f_derivative_inverse(e / alpha, f_divergence)
        state_action_ratio = jax.nn.relu(f_temp)
        cost_estimate = (state_action_ratio * batch.costs).mean()

        loss = cost_coeff * (cost_limit - cost_estimate)
        return loss, {
            "loss/lambda": loss,
            "cost/lambda": cost_coeff,
            "cost/estimate": cost_estimate,
        }

    new_cost_lambda, info = cost_lambda.apply_gradient(lambda_loss_fn)
    return new_cost_lambda, info