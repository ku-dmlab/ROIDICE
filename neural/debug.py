import jax
from jax import lax
from jax import numpy as jnp


def breakpoint_if_nonfinite(x):
    """
    https://jax.readthedocs.io/en/latest/debugging/print_breakpoint.html#usage-with-jax-lax-cond
    """
    is_finite = jnp.isfinite(x).all()

    def true_fun(x):
        pass

    def false_fun(x):
        jax.debug.breakpoint()  # type: ignore

    lax.cond(is_finite, true_fun, false_fun, x)
