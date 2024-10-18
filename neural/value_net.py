from typing import Callable, Optional, Sequence, Tuple

import jax.numpy as jnp
from flax import linen as nn
from jax import Array

from common import MLP


class CostLambda(nn.Module):
    init_value: float

    @nn.compact
    def __call__(self) -> Array:
        cost_lambda = self.param("lambda", lambda _: jnp.log(self.init_value))
        cost_lambda = jnp.clip(jnp.exp(cost_lambda), 0.0, 1e3)
        return cost_lambda

class CostMu(nn.Module):
    init_value: float

    @nn.compact
    def __call__(self) -> Array:
        cost_mu = self.param("mu", lambda _: self.init_value)
        return cost_mu

class CostT(nn.Module):
    init_value: float

    @nn.compact
    def __call__(self) -> Array:
        cost_t = self.param("t", lambda _: jnp.log(self.init_value))
        cost_t = jnp.clip(jnp.exp(cost_t), 0.0, 1e5)
        return cost_t
    
class ValueCritic(nn.Module):
    hidden_dims: Sequence[int]
    layer_norm: bool = False
    dropout_rate: Optional[float] = 0.0

    @nn.compact
    def __call__(self, observations: Array) -> Array:
        critic = MLP(
            (*self.hidden_dims, 1),
            layer_norm=self.layer_norm,
            dropout_rate=self.dropout_rate,
        )(observations)
        return jnp.squeeze(critic, -1)
    

class Critic(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[Array], Array] = nn.relu
    layer_norm: bool = False

    @nn.compact
    def __call__(self, observations: Array, actions: Array) -> Array:
        inputs = jnp.concatenate([observations, actions], -1)
        critic = MLP(
            (*self.hidden_dims, 1),
            layer_norm=self.layer_norm,
            activations=self.activations,
        )(inputs)
        return jnp.squeeze(critic, -1)


class DoubleCritic(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[Array], Array] = nn.relu
    layer_norm: bool = False

    @nn.compact
    def __call__(self, observations: Array, actions: Array) -> Tuple[Array, Array]:
        critic1 = Critic(
            self.hidden_dims, activations=self.activations, layer_norm=self.layer_norm
        )(observations, actions)
        critic2 = Critic(
            self.hidden_dims, activations=self.activations, layer_norm=self.layer_norm
        )(observations, actions)
        return critic1, critic2
