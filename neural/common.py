import math
import os
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import flax
import flax.linen as nn
import jax
import optax
from flax import struct
from flax.training.train_state import TrainState
from jax import Array


class Batch(struct.PyTreeNode):
    observations: Array
    actions: Array
    rewards: Array
    masks: Array
    next_observations: Array
    timesteps: Array
    initial_observations: Array


class ConstrainedBatch(Batch):
    costs: Array


def default_init(scale: float = math.sqrt(2)):
    return nn.initializers.orthogonal(scale)


PRNGKey = Any
Params = flax.core.FrozenDict[str, Any]
Shape = Sequence[int]
InfoDict = Dict[str, Array]


class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[Array], Array] = nn.relu
    activate_final: int = False
    layer_norm: bool = False
    dropout_rate: Optional[float] = None

    @nn.compact
    def __call__(self, x: Array, training: bool = False) -> Array:
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=default_init())(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                if self.layer_norm:
                    x = nn.LayerNorm()(x)
                x = self.activations(x)
                if self.dropout_rate is not None and self.dropout_rate > 0:
                    x = nn.Dropout(rate=self.dropout_rate)(
                        x, deterministic=not training
                    )
        return x


class Model(struct.PyTreeNode):
    step: int
    apply_fn: nn.Module = struct.field(pytree_node=False)
    params: Params
    tx: Optional[optax.GradientTransformation] = struct.field(pytree_node=False)
    opt_state: Optional[optax.OptState] = None

    @classmethod
    def create(
        cls,
        model_def: nn.Module,
        inputs: Sequence[PRNGKey | Array],
        tx: Optional[optax.GradientTransformation] = None,
    ) -> "Model":
        variables = model_def.init(*inputs)
        params = variables.pop("params")

        if tx is not None:
            opt_state = tx.init(params)
        else:
            opt_state = None

        return cls(
            step=1,
            apply_fn=model_def,
            params=params,
            tx=tx,
            opt_state=opt_state,  # type: ignore
        )

    def __call__(self, *args, **kwargs) -> Array:
        return self.apply_fn.apply(
            {"params": self.params}, *args, mutable=False, **kwargs
        )  # type: ignore

    def apply(self, *args, **kwargs) -> Array:
        return self.apply_fn.apply(*args, mutable=False, **kwargs)  # type: ignore

    def apply_gradient(self, loss_fn) -> Tuple["Model", InfoDict]:
        if self.tx is None or self.opt_state is None:
            raise RuntimeError("Optimizer is not initialized.")

        grad_fn = jax.grad(loss_fn, has_aux=True)
        grads, info = grad_fn(self.params)

        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)

        return (
            self.replace(
                step=self.step + 1, params=new_params, opt_state=new_opt_state
            ),
            info,
        )

    def save(self, save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(flax.serialization.to_bytes(self.params))

    def load(self, load_path: str) -> "Model":
        with open(load_path, "rb") as f:
            params = flax.serialization.from_bytes(self.params, f.read())
        return self.replace(params=params)

    @property
    def train_state(self):
        return TrainState.create(apply_fn=self.apply_fn, params=self.params, tx=self.tx)
