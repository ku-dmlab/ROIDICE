import typing
from typing import Tuple, Optional

import jax.numpy as jnp
from jax import Array

import divergence
from common import Batch, ConstrainedBatch, InfoDict, Model, Params, PRNGKey
from divergence import FDivergence

if typing.TYPE_CHECKING:
    from tensorflow_probability.substrates.jax.distributions import Distribution


def update_weighted_bc(
    batch: ConstrainedBatch,
    actor: Model,
    nu_state: Model,
    cost_lambda: Model,
    alpha: float,
    discount: float,
    f_divergence: FDivergence,
    rng: PRNGKey,
):
    nu = nu_state(batch.observations)
    next_nu = nu_state(batch.next_observations)

    cost_coeff = cost_lambda()

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

    def actor_loss_fn(actor_params: Params) -> Tuple[Array, InfoDict]:
        dist: Distribution = actor.apply(
            {"params": actor_params},
            batch.observations,
            training=True,
            rngs={"dropout": rng},
        )  # type: ignore
        log_probs = dist.log_prob(batch.actions)
        actor_loss = -(state_action_ratio * log_probs).mean()
        return actor_loss, {"actor_loss": actor_loss}

    new_actor, info = actor.apply_gradient(actor_loss_fn)
    return new_actor, info


def update_weighted_bc_without_cost(
    batch: Batch,
    actor: Model,
    nu_state: Model,
    w_state: Optional[Model],
    alpha: float,
    discount: float,
    f_divergence: FDivergence,
    rng: PRNGKey,
):
    nu = nu_state(batch.observations)
    next_nu = nu_state(batch.next_observations)

    if w_state is not None:
        w_s = w_state(batch.observations)

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

    def actor_loss_fn(actor_params: Params) -> Tuple[Array, InfoDict]:
        dist: Distribution = actor.apply(
            {"params": actor_params},
            batch.observations,
            training=True,
            rngs={"dropout": rng},
        )  # type: ignore
        log_probs = dist.log_prob(batch.actions)
        if w_state is not None:
            actor_loss = -(1/w_s * state_action_ratio * log_probs).mean()
        else:
            actor_loss = -(state_action_ratio * log_probs).mean()
        return actor_loss, {"actor_loss": actor_loss}

    new_actor, info = actor.apply_gradient(actor_loss_fn)
    return new_actor, info


def update_bc(
    batch: ConstrainedBatch,
    actor: Model,
    rng: PRNGKey,
):
    def actor_loss_fn(actor_params: Params) -> Tuple[Array, InfoDict]:
        dist: Distribution = actor.apply(
            {"params": actor_params},
            batch.observations,
            training=True,
            rngs={"dropout": rng},
        )  # type: ignore
        log_probs = dist.log_prob(batch.actions)
        actor_loss = -log_probs.mean()
        return actor_loss, {"actor_loss": actor_loss}

    new_actor, info = actor.apply_gradient(actor_loss_fn)
    return new_actor, info
