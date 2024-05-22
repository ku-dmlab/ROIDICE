"""Implementations of algorithms for continuous tasks."""

from functools import partial
from pathlib import Path
from typing import Optional, Sequence

import jax
import numpy as np
import optax
from flax.training import checkpoints
from jax import Array

import critic
import divergence
import policy
import value_net
from actor import update_weighted_bc_cct
from algorithm import ROIDICE
from common import Batch, ConstrainedBatch, InfoDict, Model, PRNGKey
from critic import update_nu_state_cct, update_cost_mu, update_cost_t
from divergence import FDivergence


@partial(jax.jit, static_argnames=["alg", "f_divergence"])
def _update_roidice(
    alg: ROIDICE,
    actor: Model,
    nu_state: Model,
    cost_mu: Model,
    cost_t: Model,
    batch: ConstrainedBatch,
    alpha: float,
    discount: float,
    f_divergence: FDivergence,
    gradient_penalty_coeff: float,
    rng: PRNGKey,
):
    rng, nu_rng = jax.random.split(rng)
    new_nu_state, nu_state_info = update_nu_state_cct(
        batch,
        nu_state,
        cost_mu,
        cost_t,
        alpha,
        discount,
        gradient_penalty_coeff,
        f_divergence,
        nu_rng,
    )

    new_cost_mu, cost_mu_info = update_cost_mu(
        batch,
        new_nu_state,
        cost_mu,
        cost_t,
        alpha,
        discount,
        f_divergence,
    )

    new_cost_t, cost_t_info = update_cost_t(
        batch,
        new_nu_state,
        new_cost_mu,
        cost_t,
        alpha,
        discount,
        f_divergence,
    )

    rng, actor_rng = jax.random.split(rng)
    new_actor, actor_info = update_weighted_bc_cct(
        batch,
        actor,
        new_nu_state,
        new_cost_mu,
        new_cost_t,
        alpha,
        discount,
        f_divergence,
        actor_rng,
    )

    return (
        rng,
        new_actor,
        new_nu_state,
        new_cost_mu,
        new_cost_t,
        {**actor_info, **nu_state_info, **cost_mu_info, **cost_t_info},
    )

class Learner(object):
    def __init__(
        self,
        seed: int,
        observations: Array,
        actions: Array,
        alg: ROIDICE,
        actor_lr: float = 3e-4,
        value_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        hidden_dims: Sequence[int] = (256, 256),
        discount: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.1,
        beta: float = 0.1,
        lr_ratio: float = 1.0,
        cost_lr: float = 3e-4,
        cost_ub: float = 0.5,
        cost_ub_epsilon: float = 0.0,
        gradient_penalty_coeff: float = 1e-5,
        initial_lambda: float = 1.0,
        divergence: FDivergence = FDivergence.CHI,
        dropout_rate: Optional[float] = None,
        value_dropout_rate: Optional[float] = None,
        layernorm: bool = False,
        max_steps: Optional[int] = None,
        max_clip: Optional[int] = None,
        opt_decay_schedule: str = "cosine",
        ckpt_dir: Optional[Path] = None,
        ckpt_eval_dir: Optional[Path] = None,
    ):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1801.01290
        """

        self.discount = discount
        self.alpha = alpha
        self.beta = beta
        self.max_clip = max_clip
        self.alg = alg
        self.gradient_penalty_coeff = gradient_penalty_coeff
        self.divergence = divergence
        self.cost_ub = cost_ub

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, value_key = jax.random.split(rng, 4)

        action_dim = actions.shape[-1]
        actor_def = policy.NormalTanhPolicy(
            hidden_dims,
            action_dim,
            log_std_scale=1e-3,
            log_std_min=-5.0,
            dropout_rate=dropout_rate,
            state_dependent_std=False,
            tanh_squash_distribution=False,
        )

        if opt_decay_schedule == "cosine":
            if max_steps is None:
                raise ValueError(f"{opt_decay_schedule} scheduler require max_steps.")

            schedule_fn = optax.cosine_decay_schedule(-actor_lr, max_steps)
            optimiser = optax.chain(optax.scale_by_adam(), optax.scale_by_schedule(schedule_fn))
        else:
            optimiser = optax.adam(learning_rate=actor_lr)
        actor = Model.create(actor_def, inputs=[actor_key, observations], tx=optimiser)

        value_def = value_net.ValueCritic(
            hidden_dims, layer_norm=layernorm, dropout_rate=value_dropout_rate
        )
        value = Model.create(
            value_def,
            inputs=[value_key, observations],
            tx=optax.adam(learning_rate=value_lr),
        )

        # Model define
        if self.alg == ROIDICE.DEFAULT:
            rng, nu_state_key, cost_mu_key, cost_t_key = jax.random.split(rng, 4)

            self.nu_state = Model.create(
                value_def,
                inputs=[nu_state_key, observations],
                tx=optax.adam(learning_rate=value_lr),
            )

            mu_def = value_net.CostMu(initial_lambda)
            self.cost_mu = Model.create(
                mu_def, inputs=[cost_mu_key], tx=optax.adam(learning_rate=value_lr)
            )

            t_def = value_net.CostT(initial_lambda)
            self.cost_t = Model.create(
                t_def, inputs=[cost_t_key], tx=optax.adam(learning_rate=value_lr)
            )
        else:
            NotImplementedError

        self.actor = actor
        self.value = value
        self.rng = rng

    def sample_actions(self, observations: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        rng, actions = policy.sample_actions(
            self.rng, self.actor.apply_fn, self.actor.params, observations, temperature
        )
        self.rng = rng

        actions = np.asarray(actions)
        return np.clip(actions, -1, 1)

    def update(self, batch: Batch) -> InfoDict:
        if self.alg == ROIDICE.DEFAULT:
            (
                self.rng,
                self.actor,
                self.nu_state,
                self.cost_mu,
                self.cost_t,
                info,
            ) = _update_roidice(
                alg=self.alg,
                actor=self.actor,
                nu_state=self.nu_state,
                cost_mu=self.cost_mu,
                cost_t=self.cost_t,
                batch=batch,
                alpha=self.alpha,
                discount=self.discount,
                f_divergence=self.divergence,
                gradient_penalty_coeff=self.gradient_penalty_coeff,
                rng=self.rng,
            )
        else:
            raise NotImplementedError

        return info