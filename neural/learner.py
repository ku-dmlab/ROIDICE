"""Implementations of algorithms for continuous control."""

from functools import partial
from pathlib import Path
from typing import Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import checkpoints
from jax import Array

import critic
import divergence
import policy
import value_net
from actor import update_bc, update_weighted_bc, update_weighted_bc_without_cost
from algorithm import (
    BC,
    Algorithm,
    OptiDICE,
    COptiDICE,
    ROIDICE,
)
from common import Batch, ConstrainedBatch, InfoDict, Model, Params, PRNGKey
from critic import update_nu_state, update_nu_state_without_cost, update_w_state
from divergence import FDivergence


def target_update(critic: Model, target_critic: Model, tau: float) -> Model:
    new_target_params = jax.tree_util.tree_map(
        lambda p, tp: p * tau + tp * (1 - tau), critic.params, target_critic.params
    )

    return target_critic.replace(params=new_target_params)


@partial(jax.jit, static_argnames=["f_divergence"])
def update_cost_lambda(
    f_divergence: FDivergence,
    batch: ConstrainedBatch,
    value: Model,
    critic: Model,
    advantage: Model,
    nu_network: Model,
    cost_lambda: Model,
    alpha: float,
    discount: float,
    cost_ub: float,
):
    q1, q2 = critic(batch.observations, batch.actions)
    q = jnp.minimum(q1, q2)
    v = value(batch.observations)
    adv = advantage(batch.observations)
    nu = nu_network(batch.observations)
    next_nu = nu_network(batch.next_observations)

    policy_ratio = divergence.policy_ratio(q, v, alpha, f_divergence)
    state_ratio = divergence.state_ratio(
        adv, policy_ratio, f_divergence, discount, nu, next_nu
    )
    cost_estimate = (state_ratio * policy_ratio * batch.costs).mean()

    def cost_lambda_loss_fn(params: Params) -> tuple[Array, InfoDict]:
        cost_lambda_value = cost_lambda.apply({"params": params})
        cost_lambda_loss = cost_lambda_value * (cost_ub - cost_estimate)
        return cost_lambda_loss, {
            "loss/cost_lambda": cost_lambda_loss,
            "cost/estimate": cost_estimate,
            "cost/lambda": cost_lambda_value,
            "cost/dc": cost_estimate - cost_ub,
        }

    new_cost_lambda, info = cost_lambda.apply_gradient(cost_lambda_loss_fn)
    info["cost/after_update"] = new_cost_lambda()

    return new_cost_lambda, info


@partial(jax.jit, static_argnames=["alg", "f_divergence"])
def _update_optidice(
    alg: OptiDICE,
    actor: Model,
    nu_state: Model,
    batch: Batch,
    alpha: float,
    discount: float,
    f_divergence: FDivergence,
    gradient_penalty_coeff: float,
    rng: PRNGKey,
):
    rng, nu_rng = jax.random.split(rng)
    new_nu_state, nu_state_info = update_nu_state_without_cost(
        batch,
        nu_state,
        alpha,
        discount,
        gradient_penalty_coeff,
        f_divergence,
        nu_rng,
    )

    rng, actor_rng = jax.random.split(rng)
    new_actor, actor_info = update_weighted_bc_without_cost(
        batch,
        actor,
        new_nu_state,
        None,
        alpha,
        discount,
        f_divergence,
        actor_rng,
    )

    return (
        rng,
        new_actor,
        new_nu_state,
        {**actor_info, **nu_state_info},
    )

@partial(jax.jit, static_argnames=["alg", "f_divergence"])
def _update_double_weighted_optidice(
    alg: OptiDICE,
    actor: Model,
    nu_state: Model,
    w_state: Model,
    batch: Batch,
    alpha: float,
    discount: float,
    f_divergence: FDivergence,
    gradient_penalty_coeff: float,
    rng: PRNGKey,
):
    rng, nu_rng = jax.random.split(rng)

    new_w_state, w_state_info = update_w_state(
        batch,
        nu_state,
        w_state,
        alpha,
        discount,
        f_divergence,
    )

    new_nu_state, nu_state_info = update_nu_state_without_cost(
        batch,
        nu_state,
        alpha,
        discount,
        gradient_penalty_coeff,
        f_divergence,
        nu_rng,
    )

    rng, actor_rng = jax.random.split(rng)
    new_actor, actor_info = update_weighted_bc_without_cost(
        batch,
        actor,
        new_nu_state,
        new_w_state,
        alpha,
        discount,
        f_divergence,
        actor_rng,
    )

    return (
        rng,
        new_actor,
        new_nu_state,
        new_w_state,
        {**actor_info, **nu_state_info, **w_state_info},
    )

@partial(jax.jit, static_argnames=["alg", "f_divergence"])
def _update_coptidice(
    alg: COptiDICE,
    actor: Model,
    nu_state: Model,
    cost_lambda: Model,
    batch: ConstrainedBatch,
    alpha: float,
    discount: float,
    f_divergence: FDivergence,
    gradient_penalty_coeff: float,
    cost_limit: float,
    rng: PRNGKey,
):
    rng, nu_rng = jax.random.split(rng)
    new_nu_state, nu_state_info = update_nu_state(
        batch,
        cost_lambda,
        nu_state,
        alpha,
        discount,
        gradient_penalty_coeff,
        f_divergence,
        nu_rng,
    )

    rng, actor_rng = jax.random.split(rng)
    new_actor, actor_info = update_weighted_bc(
        batch,
        actor,
        new_nu_state,
        cost_lambda,
        alpha,
        discount,
        f_divergence,
        actor_rng,
    )

    new_cost, cost_info = critic.update_cost_lambda(
        batch,
        cost_lambda,
        new_nu_state,
        alpha,
        discount,
        cost_limit,
        f_divergence,
    )

    return (
        rng,
        new_actor,
        new_nu_state,
        new_cost,
        {**actor_info, **nu_state_info, **cost_info},
    )


@partial(jax.jit, static_argnames=["alg"])
def _update_bc(
    alg: BC,
    actor: Model,
    batch: ConstrainedBatch,
    rng: PRNGKey,
):
    rng, actor_rng = jax.random.split(rng)
    new_actor, actor_info = update_bc(
        batch,
        actor,
        actor_rng,
    )

    return (
        rng,
        new_actor,
        {**actor_info},
    )


class Learner(object):
    def __init__(
        self,
        seed: int,
        observations: Array,
        actions: Array,
        alg: Algorithm,
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

        # self.expectile = expectile
        self.tau = tau
        self.discount = discount
        self.alpha = alpha
        self.beta = beta
        self.max_clip = max_clip
        self.alg = alg
        self.gradient_penalty_coeff = gradient_penalty_coeff
        self.divergence = divergence
        self.ckpt_dir = ckpt_dir
        self.ckpt_eval_dir = ckpt_eval_dir
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
            optimiser = optax.chain(
                optax.scale_by_adam(), optax.scale_by_schedule(schedule_fn)
            )
        else:
            optimiser = optax.adam(learning_rate=actor_lr)
        actor = Model.create(actor_def, inputs=[actor_key, observations], tx=optimiser)

        critic_def = value_net.DoubleCritic(hidden_dims)
        critic = Model.create(
            critic_def,
            inputs=[critic_key, observations, actions],
            tx=optax.adam(learning_rate=critic_lr),
        )

        value_def = value_net.ValueCritic(
            hidden_dims, layer_norm=layernorm, dropout_rate=value_dropout_rate
        )
        value = Model.create(
            value_def,
            inputs=[value_key, observations],
            tx=optax.adam(learning_rate=value_lr),
        )

        target_critic = Model.create(
            critic_def, inputs=[critic_key, observations, actions]
        )


        # Define a model for offline evaluation.
        match alg:
            case OptiDICE():
                rng, nu_state_key, w_state_key = jax.random.split(rng, 3)

                self.nu_state = Model.create(
                    value_def,
                    inputs=[nu_state_key, observations],
                    tx=optax.adam(learning_rate=value_lr),
                )

                w_def = value_net.ValueCritic(
                    hidden_dims, layer_norm=layernorm, dropout_rate=value_dropout_rate
                )
                self.w_state = Model.create(
                    w_def,
                    inputs=[w_state_key, observations],
                    tx=optax.adam(learning_rate=value_lr)
                )

            case COptiDICE():
                rng, nu_state_key, cost_lambda_key = jax.random.split(rng, 3)

                self.nu_state = Model.create(
                    value_def,
                    inputs=[nu_state_key, observations],
                    tx=optax.adam(learning_rate=value_lr),
                )

                lambda_def = value_net.CostLambda(initial_lambda)
                self.cost_lambda = Model.create(
                    lambda_def,
                    inputs=[cost_lambda_key],
                    tx=optax.adam(learning_rate=cost_lr),
                )

            case ROIDICE():
                # TODO
                raise NotImplementedError

            case _:
                pass

        self.actor = actor
        self.critic = critic
        self.value = value
        self.target_critic = target_critic
        self.rng = rng

    def sample_actions(
        self, observations: np.ndarray, temperature: float = 1.0
    ) -> np.ndarray:
        rng, actions = policy.sample_actions(
            self.rng, self.actor.apply_fn, self.actor.params, observations, temperature
        )
        self.rng = rng

        actions = np.asarray(actions)
        return np.clip(actions, -1, 1)

    def update(self, batch: Batch) -> InfoDict:
        if self.alg == OptiDICE.DEFAULT:
            (
                self.rng,
                self.actor,
                self.nu_state,
                info,
            ) = _update_optidice(
                alg=self.alg,
                actor=self.actor,
                nu_state=self.nu_state,
                batch=batch,
                alpha=self.alpha,
                discount=self.discount,
                f_divergence=self.divergence,
                gradient_penalty_coeff=self.gradient_penalty_coeff,
                rng=self.rng,
            )
        elif self.alg == OptiDICE.WEIGHT:
            (
                self.rng,
                self.actor,
                self.nu_state,
                self.w_state,
                info,
            ) = _update_double_weighted_optidice(
                alg=self.alg,
                actor=self.actor,
                nu_state=self.nu_state,
                w_state=self.w_state,
                batch=batch,
                alpha=self.alpha,
                discount=self.discount,
                f_divergence=self.divergence,
                gradient_penalty_coeff=self.gradient_penalty_coeff,
                rng=self.rng
            )
        # Update lambda for constrained RL.
        elif self.alg == COptiDICE.DEFAULT:
            (
                self.rng,
                self.actor,
                self.nu_state,
                self.cost_lambda,
                info,
            ) = _update_coptidice(
                alg=self.alg,
                actor=self.actor,
                nu_state=self.nu_state,
                cost_lambda=self.cost_lambda,
                batch=batch,
                alpha=self.alpha,
                discount=self.discount,
                f_divergence=self.divergence,
                gradient_penalty_coeff=self.gradient_penalty_coeff,
                cost_limit=self.cost_ub,
                rng=self.rng,
            )
        elif self.alg == ROIDICE.DEFAULT:
            # TODO
            raise NotImplementedError
        elif self.alg == BC.DEFAULT:
            self.rng, self.actor, info = _update_bc(
                alg=self.alg, actor=self.actor, batch=batch, rng=self.rng
            )
        else:
            raise NotImplementedError 

        return info

    def update_constraint(self, batch: ConstrainedBatch) -> InfoDict:
        self.cost_lambda, info = update_cost_lambda(
            f_divergence=self.divergence,
            batch=batch,
            value=self.value,
            critic=self.critic,
            advantage=self.advantage,
            nu_network=self.value2,
            cost_lambda=self.cost_lambda,
            alpha=self.alpha,
            discount=self.discount,
            cost_ub=self.cost_ub,
        )

        return info

    def save_ckpt(self, step: int):
        # Silently fail if save directory is not provided.
        if self.ckpt_dir is None:
            pass

        checkpoints.save_checkpoint(
            ckpt_dir=str(self.ckpt_dir),
            target=self.actor.train_state,
            step=step,
            prefix="actor_ckpt_",
        )
        checkpoints.save_checkpoint(
            ckpt_dir=str(self.ckpt_dir),
            target=self.critic.train_state,
            step=step,
            prefix="critic_ckpt_",
        )
        checkpoints.save_checkpoint(
            ckpt_dir=str(self.ckpt_dir),
            target=self.value.train_state,
            step=step,
            prefix="value_ckpt_",
        )

    def load_ckpt(self, ckpt_dir: Path, step: int):
        actor_state = checkpoints.restore_checkpoint(
            ckpt_dir=ckpt_dir,
            target=self.actor.train_state,
            step=step,
            prefix="actor_ckpt_",
        )
        self.actor = self.actor.replace(params=actor_state.params)
        self.actor = self.actor.replace(tx=actor_state.tx)

        critic_state = checkpoints.restore_checkpoint(
            ckpt_dir=ckpt_dir,
            target=self.critic.train_state,
            step=step,
            prefix="critic_ckpt_",
        )
        self.critic = self.critic.replace(params=critic_state.params)
        self.critic = self.critic.replace(tx=critic_state.tx)

        value_state = checkpoints.restore_checkpoint(
            ckpt_dir=ckpt_dir,
            target=self.value.train_state,
            step=step,
            prefix="value_ckpt_",
        )
        self.value = self.value.replace(params=value_state.params)
        self.value = self.value.replace(tx=value_state.tx)