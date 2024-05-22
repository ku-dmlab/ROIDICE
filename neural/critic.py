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
        costs = jnp.zeros_like(batch.costs)
    else:
        cost_coeff = cost_lambda()
        costs = batch.costs

    @partial(jax.vmap, in_axes=(None, 0))
    @partial(jax.grad, argnums=1)
    def grad_nu(value_params, obs):
        return nu_state.apply({"params": value_params}, obs)

    def nu_loss_fn(params: Params) -> Tuple[Array, InfoDict]:
        initial_nu = nu_state.apply({"params": params}, batch.initial_observations)
        nu = nu_state.apply({"params": params}, batch.observations)
        next_nu = nu_state.apply({"params": params}, batch.next_observations)
        e = batch.rewards - cost_coeff * costs + discount * next_nu - nu

        state_action_ratio = divergence.state_action_ratio(
            nu,
            next_nu,
            batch.rewards,
            costs,
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


def update_nu_state_cct(
    batch: ConstrainedBatch,
    nu_state: Model,
    cost_mu: Model,
    cost_t: Model,
    alpha: float,
    discount: float,
    gradient_penalty_coeff: float,
    f_divergence: FDivergence,
    rng: PRNGKey,
):
    mu = cost_mu()
    t = cost_t()

    @partial(jax.vmap, in_axes=(None, 0))
    @partial(jax.grad, argnums=1)
    def grad_nu(value_params, obs):
        return nu_state.apply({"params": value_params}, obs)

    def nu_loss_fn(params: Params) -> Tuple[Array, InfoDict]:
        initial_nu = nu_state.apply({"params": params}, batch.initial_observations)
        nu = nu_state.apply({"params": params}, batch.observations)
        next_nu = nu_state.apply({"params": params}, batch.next_observations)
        e = batch.rewards + discount * next_nu - nu

        state_action_ratio = divergence.state_action_ratio(
            nu,
            next_nu,
            batch.rewards,
            batch.costs,
            alpha,
            0,
            discount,
            f_divergence,
            mu=mu,
            t=t,
        )

        initial_loss = t * (1 - discount) * initial_nu
        non_initial_loss = state_action_ratio * (
            e - mu * jnp.array(batch.costs)
        ) - alpha * divergence.f(state_action_ratio, f_divergence, t=t)

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


def update_upper_bound(
    batch: ConstrainedBatch,
    nu_state: Model,
    tau: Model,
    chi_state: Model,
    cost_lambda: Model,
    cost_ub_epsilon: float,
    alpha: float,
    discount: float,
    f_divergence: FDivergence,
    rng: PRNGKey,
):
    cost_coeff = cost_lambda()
    nu = nu_state(batch.observations)
    next_nu = nu_state(batch.next_observations)

    initial_chi = chi_state(batch.initial_observations)
    chi = chi_state(batch.observations)
    next_chi = chi_state(batch.next_observations)

    batch_size = batch.observations.shape[0]

    def tau_loss_fn(params: Params) -> Tuple[Array, InfoDict]:
        dist_tau = tau.apply({"params": params})
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

        ell = (1 - discount) * initial_chi + state_action_ratio * (
            batch.costs + discount * batch.masks * next_chi - chi
        )
        logits = ell / dist_tau
        weights = jax.nn.softmax(logits, axis=0) * batch_size
        log_weights = jax.nn.log_softmax(logits, axis=0) + jnp.log(batch_size)
        kl_divergence = (weights * log_weights - weights + 1).mean()

        loss = dist_tau * (cost_ub_epsilon - kl_divergence)

        return loss, {"loss/kl_divergence": kl_divergence, "loss/tau:": loss, "tau": dist_tau}

    new_tau, info = tau.apply_gradient(tau_loss_fn)

    dist_tau = tau()

    def chi_state_loss_fn(params: Params) -> Tuple[Array, InfoDict]:
        initial_chi = chi_state.apply({"params": params}, batch.initial_observations)
        chi = chi_state.apply({"params": params}, batch.observations)
        next_chi = chi_state.apply({"params": params}, batch.next_observations)

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

        ell = (1 - discount) * initial_chi + state_action_ratio * (
            batch.costs + discount * batch.masks * next_chi - chi
        )
        logits = ell / dist_tau
        weights = jax.nn.softmax(logits, axis=0) * batch_size

        loss = (weights * ell).mean()

        return loss, {"loss/chi_state": loss}

    new_chi_state, chi_state_info = chi_state.apply_gradient(chi_state_loss_fn)

    return new_tau, new_chi_state, {**info, **chi_state_info}


def update_cost_mu(
    batch: ConstrainedBatch,
    nu_state: Model,
    cost_mu: Model,
    cost_t: Model,
    alpha: float,
    discount: float,
    f_divergence: FDivergence,
):
    nu = nu_state(batch.observations)
    next_nu = nu_state(batch.next_observations)

    t = cost_t()

    def mu_loss_fn(params: Params) -> tuple[Array, InfoDict]:
        mu = cost_mu.apply({"params": params})

        e = batch.rewards + discount * next_nu - nu
        f_temp = divergence.f_derivative_inverse(
            (e - mu * jnp.array(batch.costs)) / alpha, f_divergence, t=t
        )
        state_action_ratio = jax.nn.relu(f_temp)

        f = divergence.f(state_action_ratio, f_divergence, t=t)

        loss = state_action_ratio * (e - mu * jnp.array(batch.costs)) - alpha * f
        loss = loss.mean() + mu
        return loss, {
            "loss/mu": loss,
            "cost/mu": mu,
        }

    new_cost_mu, info = cost_mu.apply_gradient(mu_loss_fn)
    return new_cost_mu, info


def update_cost_t(
    batch: ConstrainedBatch,
    nu_state: Model,
    cost_mu: Model,
    cost_t: Model,
    alpha: float,
    discount: float,
    f_divergence: FDivergence,
):
    initial_nu = nu_state(batch.initial_observations)
    nu = nu_state(batch.observations)
    next_nu = nu_state(batch.next_observations)

    mu = cost_mu()

    def t_loss_fn(params: Params) -> tuple[Array, InfoDict]:
        t = cost_t.apply({"params": params})

        e = batch.rewards + discount * next_nu - nu
        f_temp = divergence.f_derivative_inverse((e - mu * batch.costs) / alpha, f_divergence, t=t)
        state_action_ratio = jax.nn.relu(f_temp)

        f = divergence.f(state_action_ratio, f_divergence, t=t)

        initial_loss = t * (1 - discount) * initial_nu
        loss = state_action_ratio * (e - mu * jnp.array(batch.costs)) - alpha * f
        loss = -initial_loss.mean() - loss.mean()

        return loss, {
            "loss/t": loss,
            "cost/t": t,
        }

    new_cost_t, info = cost_t.apply_gradient(t_loss_fn)
    return new_cost_t, info
