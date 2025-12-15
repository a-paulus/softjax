import inspect

import jax
from jax import tree_util as jtu
from pyparsing import Callable

import softjax as sj


def grad_replace(fn: Callable) -> Callable:
    """This decorator calls the decorated function twice: once with `forward=True` and
    once with `forward=False`. It returns the output from the forward pass, but uses
    the output from the backward pass to compute gradients.

    **Arguments:**

    - `fn`: The function to be wrapped. It should accept a `forward` argument
        that specifies which computation to perform depending on forward or
        backward pass.

    **Returns:**
        A wrapped function that behaves like the `forward=True` version during the
        forward pass, but computes gradients using the `forward=False` version
        during the backward pass.
    """

    def wrapped(*args, **kwargs):
        fw_y = fn(*args, **kwargs, forward=True)
        bw_y = fn(*args, **kwargs, forward=False)
        return jtu.tree_map(
            lambda fw, bw: jax.lax.stop_gradient(fw - bw) + bw,
            fw_y,
            bw_y,
        )

    return wrapped


def st(fn: Callable) -> Callable:
    """This decorator calls the decorated function twice: once with `mode="hard"` and
    once with the specified `mode`. It returns the output from the hard forward pass,
    but uses the output from the soft backward pass to compute gradients.

    **Arguments:**

    - `fn`: The function to be wrapped. It should accept a `mode` argument.

    **Returns:**
        A wrapped function that behaves like the `mode="hard"` version during the
        forward pass, but computes gradients using the specified `mode` and `softness`
        during the backward pass.
    """
    sig = inspect.signature(fn)
    mode_default = sig.parameters.get("mode").default

    def wrapped(*args, **kwargs):
        mode = kwargs.pop("mode", mode_default)
        fw_y = fn(*args, **kwargs, mode="hard")
        bw_y = fn(*args, **kwargs, mode=mode)
        return jtu.tree_map(
            lambda fw, bw: jax.lax.stop_gradient(fw - bw) + bw,
            fw_y,
            bw_y,
        )

    return wrapped


def argmax_st(*args, **kwargs):
    """
    Straight-through version of [`softjax.argmax`][].

    This function returns the hard `argmax` during the forward pass, but uses a
    soft relaxation (controlled by the `mode` argument) for the backward pass
    (i.e., gradients are computed through the soft version).

    Implemented using the [`softjax.st`][] decorator as `st(softjax.argmax)`.
    """
    return st(sj.argmax)(*args, **kwargs)


def argmin_st(*args, **kwargs):
    return st(sj.argmin)(*args, **kwargs)


def argsort_st(*args, **kwargs):
    return st(sj.argsort)(*args, **kwargs)


def clip_st(*args, **kwargs):
    return st(sj.clip)(*args, **kwargs)


def equal_st(*args, **kwargs):
    return st(sj.equal)(*args, **kwargs)


def greater_st(*args, **kwargs):
    return st(sj.greater)(*args, **kwargs)


def greater_equal_st(*args, **kwargs):
    return st(sj.greater_equal)(*args, **kwargs)


def heaviside_st(*args, **kwargs):
    return st(sj.heaviside)(*args, **kwargs)


def isclose_st(*args, **kwargs):
    return st(sj.isclose)(*args, **kwargs)


def less_st(*args, **kwargs):
    return st(sj.less)(*args, **kwargs)


def less_equal_st(*args, **kwargs):
    return st(sj.less_equal)(*args, **kwargs)


def max_st(*args, **kwargs):
    return st(sj.max)(*args, **kwargs)


def median_st(*args, **kwargs):
    return st(sj.median)(*args, **kwargs)


def median_newton_st(*args, **kwargs):
    return st(sj.median_newton)(*args, **kwargs)


def min_st(*args, **kwargs):
    return st(sj.min)(*args, **kwargs)


def not_equal_st(*args, **kwargs):
    return st(sj.not_equal)(*args, **kwargs)


def ranking_st(*args, **kwargs):
    return st(sj.ranking)(*args, **kwargs)


def relu_st(*args, **kwargs):
    return st(sj.relu)(*args, **kwargs)


def round_st(*args, **kwargs):
    return st(sj.round)(*args, **kwargs)


def sign_st(*args, **kwargs):
    return st(sj.sign)(*args, **kwargs)


def sort_st(*args, **kwargs):
    return st(sj.sort)(*args, **kwargs)


def top_k_st(*args, **kwargs):
    return st(sj.top_k)(*args, **kwargs)
