from typing import Literal

import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Float

from softjax.functions import (
    _maybe_log_soft_index,
    _validate_log_prob_request,
    argmax,
    less,
    SoftBool,
    SoftIndex,
)
from softjax.straight_through import _cached_st
from softjax.utils import (
    _canonicalize_axis,
    _canonicalize_shape,
    _check_broadcast_shape,
)


def _validate_noise(noise: str, expected: str) -> None:
    if noise != expected:
        raise ValueError(f"Invalid noise: {noise}. Only noise='{expected}' is supported.")


def _categorical_batch_shape(logits: Array, axis: int) -> tuple[int, ...]:
    return tuple(dim for i, dim in enumerate(logits.shape) if i != axis)


def _categorical_gumbel_perturbation(
    key: Array,
    logits: Float[Array, "..."],
    axis: int,
    shape: tuple[int, ...] | None,
    rng_mode: Literal["high", "low"] | None,
) -> tuple[Array, int]:
    logits = jnp.asarray(logits)
    batch_shape = _categorical_batch_shape(logits, axis)
    if shape is None:
        shape = batch_shape
    else:
        _check_broadcast_shape("categorical", shape, batch_shape)

    shape_prefix = shape[: len(shape) - len(batch_shape)]
    axis = axis - logits.ndim if axis >= 0 else axis
    logits_shape = list(shape[len(shape) - len(batch_shape) :])
    logits_shape.insert(axis % logits.ndim, logits.shape[axis])

    gumbel = jax.random.gumbel(
        key, (*shape_prefix, *logits_shape), logits.dtype, mode=rng_mode
    )
    logits = lax.expand_dims(logits, tuple(range(len(shape_prefix))))
    return gumbel + logits, axis


def categorical(
    key: Array,
    logits: Float[Array, "..."],
    axis: int = -1,
    shape: tuple[int, ...] | None = None,
    replace: bool = True,
    rng_mode: Literal["high", "low"] | None = None,
    softness: float | Array = 0.1,
    mode: Literal["hard", "_hard", "smooth", "c0", "c1", "c2"] = "smooth",
    method: Literal["ot", "softsort", "neuralsort", "sorting_network"] = "softsort",
    noise: Literal["gumbel"] = "gumbel",
    standardize: bool = False,
    ot_kwargs: dict | None = None,
    return_log_probs: bool = False,
    log_prob_eps: float | Array | None = None,
) -> SoftIndex:
    """Soft replacement for [`jax.random.categorical`](https://docs.jax.dev/en/latest/_autosummary/jax.random.categorical.html).

    **Arguments:**

    - `key`: JAX PRNG key.
    - `logits`: Unnormalized log probabilities. `softmax(logits, axis)` gives the categorical probabilities.
    - `axis`: Axis along which logits belong to the same categorical distribution.
    - `shape`: Optional output sample shape, following `jax.random.categorical`.
    - `replace`: Whether to sample with replacement. Soft modes currently support only `replace=True`.
    - `rng_mode`: JAX Gumbel sampler precision mode, `"high"` or `"low"`. If `None`, JAX chooses its configured default.
    - `softness`: Softness passed to [`softjax.argmax`][].
    - `mode`: Relaxation mode. `"hard"` calls `jax.random.categorical` and returns a one-hot `SoftIndex`; `"_hard"` mirrors the Gumbel-max implementation via [`softjax.argmax`][]; soft modes apply [`softjax.argmax`][] to Gumbel-perturbed logits.
    - `method`: Method passed to [`softjax.argmax`][].
    - `noise`: Perturbation distribution. Currently only `"gumbel"` is supported.
    - `standardize`: Whether to standardize perturbed logits before applying [`softjax.argmax`][]. Defaults to `False` so logits keep their probabilistic scale.
    - `ot_kwargs`: Optional keyword arguments for OT-based [`softjax.argmax`][].
    - `return_log_probs`: If True, returns log relaxed sample weights instead of probabilities. These are not categorical log-likelihoods under the original logits.
    - `log_prob_eps`: Optional probability floor used only with `return_log_probs=True`.

    **Returns:**

    A `SoftIndex` over sampled categories. In hard mode this is a one-hot encoding of the integer categories returned by `jax.random.categorical`; in soft modes it is a relaxed stochastic argmax sample.
    """
    _validate_noise(noise, "gumbel")
    _validate_log_prob_request(return_log_probs, log_prob_eps)
    shape = _canonicalize_shape(shape)
    logits = jnp.asarray(logits)
    axis = _canonicalize_axis(axis, logits.ndim)
    num_classes = logits.shape[axis]

    if mode == "hard":
        indices = jax.random.categorical(
            key, logits, axis=axis, shape=shape, replace=replace, mode=rng_mode
        )
        soft_index = jax.nn.one_hot(indices, num_classes=num_classes, axis=-1)
        return _maybe_log_soft_index(
            soft_index, return_log_probs, already_log=False, log_prob_eps=log_prob_eps
        )

    if not replace:
        raise NotImplementedError(
            "softjax.random.categorical currently supports replace=False only "
            "with mode='hard'."
        )

    perturbed, perturbed_axis = _categorical_gumbel_perturbation(
        key, logits, axis=axis, shape=shape, rng_mode=rng_mode
    )
    return argmax(
        perturbed,
        axis=perturbed_axis,
        softness=softness,
        mode=mode,
        method=method,
        standardize=standardize,
        ot_kwargs=ot_kwargs,
        return_log_probs=return_log_probs,
        log_prob_eps=log_prob_eps,
    )


def _bernoulli_uniform(
    key: Array,
    p: Float[Array, "..."],
    shape: tuple[int, ...] | None,
    rng_mode: Literal["high", "low"],
) -> Array:
    p = jnp.asarray(p)
    if shape is None:
        shape = p.shape
    else:
        _check_broadcast_shape("bernoulli", shape, p.shape)

    if rng_mode == "high":
        u1, u2 = jax.random.uniform(key, (2, *shape), p.dtype)
        u2 *= 2 ** -jnp.finfo(p.dtype).nmant
        return u1 + u2
    if rng_mode == "low":
        return jax.random.uniform(key, shape, p.dtype)
    raise ValueError(f"got rng_mode={rng_mode!r}, expected 'high' or 'low'")


def bernoulli(
    key: Array,
    p: Float[Array, "..."] = 0.5,
    shape: tuple[int, ...] | None = None,
    rng_mode: Literal["high", "low"] = "low",
    softness: float | Array = 0.1,
    mode: Literal["hard", "_hard", "smooth", "c0", "c1", "c2"] = "smooth",
    noise: Literal["uniform"] = "uniform",
) -> SoftBool:
    """Soft replacement for [`jax.random.bernoulli`](https://docs.jax.dev/en/latest/_autosummary/jax.random.bernoulli.html).

    **Arguments:**

    - `key`: JAX PRNG key.
    - `p`: Bernoulli probability. Must be broadcast-compatible with `shape`.
    - `shape`: Optional output sample shape, following `jax.random.bernoulli`.
    - `rng_mode`: JAX Bernoulli sampler precision mode, `"high"` or `"low"`.
    - `softness`: Softness passed to the soft threshold.
    - `mode`: Relaxation mode. `"hard"` calls `jax.random.bernoulli`; `"_hard"` mirrors JAX threshold sampling; soft modes apply a soft threshold to the same uniform draw.
    - `noise`: Threshold noise distribution. Currently only `"uniform"` is supported.

    **Returns:**

    A numeric `SoftBool`. In hard mode this is the boolean JAX sample cast to the probability dtype; in soft modes it is a relaxed uniform-threshold sample.
    """
    _validate_noise(noise, "uniform")
    shape = _canonicalize_shape(shape)
    p = jnp.asarray(p)

    if mode == "hard":
        sample = jax.random.bernoulli(key, p=p, shape=shape, mode=rng_mode)
        return sample.astype(p.dtype)

    u = _bernoulli_uniform(key, p, shape=shape, rng_mode=rng_mode)
    if mode == "_hard":
        return jnp.less(u, p).astype(p.dtype)
    return less(u, p, softness=softness, mode=mode)


def categorical_st(*args, **kwargs):
    """Straight-through version of [`softjax.random.categorical`][].

    Returns a hard one-hot categorical sample during the forward pass, while gradients are computed through the requested soft stochastic argmax relaxation.
    """
    return _cached_st(categorical)(*args, **kwargs)


def bernoulli_st(*args, **kwargs):
    """Straight-through version of [`softjax.random.bernoulli`][].

    Returns a hard numeric Bernoulli sample during the forward pass, while gradients are computed through the requested soft threshold relaxation.
    """
    return _cached_st(bernoulli)(*args, **kwargs)


__all__ = [
    "bernoulli",
    "bernoulli_st",
    "categorical",
    "categorical_st",
]
