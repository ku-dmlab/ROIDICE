import typing
from typing import Tuple

from jax import Array

import divergence
from common import ConstrainedBatch, InfoDict, Model, Params, PRNGKey
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

    if cost_lambda is None:
        cost_coeff = 0  # unconstrained
    else:
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