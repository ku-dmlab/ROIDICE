from enum import Enum
from functools import partial

import jax
from jax import Array, custom_jvp
from jax import numpy as jnp
from jax.typing import ArrayLike
from typing_extensions import assert_never


class FDivergence(str, Enum):
    KL = "KL"
    CHI = "Chi"
    SOFT_CHI = "SoftChi"
    DUAL_DICE = "DualDICE"
    CHI_T = "ChiT"
    SOFT_CHI_T = "SoftChiT"


def f(x: ArrayLike, f_divergence: FDivergence, t: float = 1.0, eps: float = 1e-10) -> Array:
    """Compute f-function of f-Divergence.

    Args:
        x (ArrayLike): an input to f-function
        f_divergence (FDivergence): the variants of f-Divergence to use.
        eps (float): Small epsilon added to x for numerical stability in some functions.

    Returns:
        Array: The value of f(x).
    """

    if not isinstance(x, ArrayLike):
        raise TypeError(f"Expected arraylike input; got {x}")

    x = jnp.array(x)

    match f_divergence:
        case FDivergence.KL:
            return x * jnp.log(x + eps)
        case FDivergence.CHI:
            return (x - 1) ** 2 / 2
        case FDivergence.SOFT_CHI:
            # We didn't use double where trick here because in theory, x should be
            # always positive.
            return jnp.where(x < 1.0, x * jnp.log(x + eps) - x + 1, (x - 1) ** 2 / 2)
        case FDivergence.DUAL_DICE:
            return 2 / 3 * jnp.abs(x) ** (3 / 2)
        case FDivergence.CHI_T:
            return (x - t) ** 2 / 2
        case FDivergence.SOFT_CHI_T:
            return jnp.where(x < t, (x / t) * jnp.log(x / t + eps) - (x / t) + 1, (x - t) ** 2 / 2)
        case _:
            assert_never(f_divergence)


def f_derivative_inverse(y: ArrayLike, f_divergence: FDivergence, t: float = 1.0):
    """Compute (f')^-1 of f-function of f-Divergence.

    This function is required when computing convex conjugate of f-function.

    Args:
        y (ArrayLike): an input to (f')^-1 function
        f_divergence (FDivergence): the variants of f-Divergence to use.

    Returns:
        Array: The value of (f')^-1(y).
    """

    if not isinstance(y, ArrayLike):
        raise TypeError(f"Expected arraylike input; got {y}")

    y = jnp.array(y)

    match f_divergence:
        case FDivergence.KL:
            return jnp.exp(y - 1.0)
        case FDivergence.CHI:
            return y + 1.0
        case FDivergence.SOFT_CHI:
            # When y becomes too large so that jnp.exp(y) becomes inf, even if jnp.exp(y) is
            # not actually called, it will cause problem. To avoid this, we use jnp.where twice.
            # Ref.: https://github.com/tensorflow/probability/blob/main/discussion/where-nan.pdf
            return jnp.where(y < 0.0, jnp.exp(jnp.where(y < 0.0, y, 0.0)), y + 1)
        case FDivergence.DUAL_DICE:
            raise ValueError(f"This funtion doesn't exist for {f_divergence}.")
        case FDivergence.CHI_T:
            return y + t
        case FDivergence.SOFT_CHI_T:
            return jnp.where(y < 0.0, t * jnp.exp(jnp.where(t * y < 0.0, t * y, 0.0)), y + t)
        case _:
            assert_never(f_divergence)


def f_conjugate(y: ArrayLike, f_divergence: FDivergence) -> Array:
    """Compute convex conjugate f^*(y) of f-function of f-Divergence.

    Args:
        y (ArrayLike): an input to f^* function
        f_divergence (FDivergence): the variants of f-Divergence to use.

    Returns:
        Array: The value of f^*(y)
    """

    if not isinstance(y, ArrayLike):
        raise TypeError(f"Expected arraylike input; got {y}")

    y = jnp.array(y)

    match f_divergence:
        case FDivergence.KL:
            return jnp.exp(y - 1.0)
        case FDivergence.CHI:
            return jnp.where(y >= -1.0, y**2 / 2 + y, -0.5)
        case FDivergence.SOFT_CHI:
            # When y becomes too large so that jnp.exp(y) becomes inf, even if jnp.exp(y) is
            # not actually called, it will cause problem. To avoid this, we use jnp.where twice.
            # Ref.: https://github.com/tensorflow/probability/blob/main/discussion/where-nan.pdf
            return jnp.where(
                y >= 0.0, y**2 / 2 + y, jnp.exp(jnp.where(y < 0.0, y, 0.0)) - 1
            )
        case FDivergence.DUAL_DICE:
            return jnp.abs(y) ** 3 / 3
        case FDivergence.SOFT_CHI_T:
            raise NotImplementedError
        case _:
            assert_never(f_divergence)


def policy_ratio(
    q: ArrayLike, value: ArrayLike, alpha: float, f_divergence: FDivergence
) -> Array:
    """Compute the policy ratio.

    Args:
        q (ArrayLike): Critic value of the current observation.
        value (ArrayLike): Expected value of the current observation.
        f_divergence (FDivergence): the variants of f-Divergence to use.

    Returns:
        Array: Policy ratio.
    """

    if not isinstance(q, ArrayLike):
        raise TypeError(f"Expected arraylike input; got {q}")
    if not isinstance(value, ArrayLike):
        raise TypeError(f"Expected arraylike input; got {value}")

    q = jnp.array(q)
    value = jnp.array(value)
    x = (q - value) / alpha

    match f_divergence:
        case FDivergence.CHI:
            return jnp.where(x + 1.0 > 0.0, x + 1.0, 0.0)
        case FDivergence.SOFT_CHI:
            # When x becomes too large so that jnp.exp(x) becomes inf, even if jnp.exp(x) is
            # not actually called, it will cause problem. To avoid this, we use jnp.where twice.
            # Ref.: https://github.com/tensorflow/probability/blob/main/discussion/where-nan.pdf
            return jnp.where(x > 0, x + 1, jnp.exp(jnp.where(x <= 0.0, x, 0.0)))
        case FDivergence.KL | FDivergence.DUAL_DICE | FDivergence.SOFT_CHI_T:
            raise NotImplementedError(
                f"This funtion isn't implemented for {f_divergence}."
            )
        case _:
            assert_never(f_divergence)


@partial(custom_jvp, nondiff_argnums=(0, 1, 2, 3))
def state_ratio(
    advantage: ArrayLike,
    policy_ratio: ArrayLike,
    f_divergence: FDivergence,
    discount: float,
    nu: ArrayLike,
    next_nu: ArrayLike,
) -> Array:
    """Compute the state stationary distribution ratio given (q - v) / alpha.

    Args:
        nu (ArrayLike): the argument for custom JVP.
        next_nu (ArrayLike): the argument for custom JVP.
        advantage (ArrayLike): the advantage of the current observation.
        policy_ratio (ArrayLike): the argument for custom JVP.
        f_divergence (FDivergence): the variants of f-Divergence to use.
        discount (float): the argument for custom JVP.

    Returns:
        Array: state stationary distribution ratio.
    """

    if not isinstance(nu, ArrayLike):
        raise TypeError(f"Expected arraylike input; got {nu}")
    if not isinstance(next_nu, ArrayLike):
        raise TypeError(f"Expected arraylike input; got {next_nu}")
    if not isinstance(advantage, ArrayLike):
        raise TypeError(f"Expected arraylike input; got {advantage}")
    if not isinstance(policy_ratio, ArrayLike):
        raise TypeError(f"Expected arraylike input; got {policy_ratio}")

    advantage = jnp.array(advantage)

    match f_divergence:
        case FDivergence.CHI:
            return jnp.where(advantage + 1.0 > 0.0, advantage + 1.0, 0.0)
        case FDivergence.SOFT_CHI:
            # When x becomes too large so that jnp.exp(x) becomes inf, even if jnp.exp(x) is
            # not actually called, it will cause problem. To avoid this, we use jnp.where twice.
            # Ref.: https://github.com/tensorflow/probability/blob/main/discussion/where-nan.pdf
            return jnp.where(
                advantage > 0.0,
                advantage + 1,
                jnp.exp(jnp.where(advantage <= 0.0, advantage, 0.0)),
            )
        case FDivergence.KL | FDivergence.DUAL_DICE | FDivergence.SOFT_CHI_T:
            raise NotImplementedError(
                f"This funtion isn't implemented for {f_divergence}."
            )
        case _:
            assert_never(f_divergence)


@state_ratio.defjvp
def state_ratio_jvp(
    advantage: Array,
    policy_ratio: Array,
    f_divergence: FDivergence,
    discount: float,
    primals: tuple[Array, Array],
    tangents: tuple[Array, Array],
) -> tuple[Array, Array]:
    nu, next_nu = primals
    tangent_nu, tangent_next_nu = tangents

    primal_out = state_ratio(
        advantage, policy_ratio, f_divergence, discount, nu, next_nu
    )

    match f_divergence:
        case FDivergence.CHI:
            tangent_out = (
                jnp.where(advantage + 1 > 0.0, 1.0, 0.0)
                * policy_ratio
                * (discount * tangent_next_nu - tangent_nu)
            )
        case FDivergence.SOFT_CHI:
            # When x becomes too large so that jnp.exp(x) becomes inf, even if jnp.exp(x) is
            # not actually called, it will cause problem. To avoid this, we use jnp.where twice.
            # Ref.: https://github.com/tensorflow/probability/blob/main/discussion/where-nan.pdf
            tangent_out = (
                jnp.where(
                    advantage > 0.0,
                    1.0,
                    jnp.exp(jnp.where(advantage <= 0.0, advantage, 0.0)),
                )
                * policy_ratio
                * (discount * tangent_next_nu - tangent_nu)
            )
        case FDivergence.KL | FDivergence.DUAL_DICE | FDivergence.SOFT_CHI_T:
            raise NotImplementedError(
                f"This function is not implemented for {f_divergence}."
            )
        case _:
            assert_never(f_divergence)
    return primal_out, tangent_out


def state_action_ratio(
    nu: ArrayLike,
    next_nu: ArrayLike,
    rewards: ArrayLike,
    costs: ArrayLike,
    alpha: float,
    cost_coeff: float,
    discount: float,
    f_divergence: FDivergence,
    mu: float = 0.0,
    t: float = 1.0,
) -> Array:
    if not isinstance(nu, ArrayLike):
        raise TypeError(f"Expected arraylike input; got {nu}")
    if not isinstance(next_nu, ArrayLike):
        raise TypeError(f"Expected arraylike input; got {next_nu}")
    if not isinstance(rewards, ArrayLike):
        raise TypeError(f"Expected arraylike input; got {rewards}")
    if not isinstance(costs, ArrayLike):
        raise TypeError(f"Expected arraylike input; got {costs}")

    nu = jnp.array(nu)
    next_nu = jnp.array(next_nu)
    rewards = jnp.array(rewards)
    costs = jnp.array(costs)

    if FDivergence(f_divergence) in ["SoftChi", "Chi"]:
        e = rewards - cost_coeff * costs + discount * next_nu - nu
        return jax.nn.relu(f_derivative_inverse(e / alpha, f_divergence))
    else:
        e = rewards + discount * next_nu - nu
        return jax.nn.relu(f_derivative_inverse((e - mu * costs) / alpha, f_divergence, t=t))
