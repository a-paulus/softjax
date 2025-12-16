# functions.py

from collections.abc import Sequence
from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
from jax import Array
from jaxopt import LBFGS
from jaxopt.projection import kl_projection_transport, projection_transport
from jaxtyping import Float

from softjax.utils import _canonicalize_axis, _projection_unit_simplex, _sinkhorn


SoftBool = Float[Array, "..."]  # probability in [0, 1]
SoftIndex = Float[Array, "..."]  # probabilities summing to 1 along the last axis

ENTROPIC_CONSTANT = 10.0
PSEUDOHUBER_CONSTANT = 10.0
EUCLIDEAN_CONSTANT = 2.0
EUCLIDEAN_OT_CONSTANT = 1.0


# Array -> SoftIndex


@partial(jax.jit, static_argnames=("axis", "softness", "mode"))
def _projection_simplex(
    x: Array,  # (..., n, ...)
    axis: int,
    softness: float = 1.0,
    mode: Literal["entropic", "euclidean"] = "entropic",
) -> Array:  # (..., [n], ...)
    """Projects `x` onto the unit simplex along the specified axis.

    Solves the optimization problem along the specified axis:
        min_y <x, y> + softness * R(y)
        s.t. y >= 0, sum(y) = 1
    where R(y) is the regularizer determined by `mode`.

    **Arguments:**

    - `x`: Input Array of shape (..., n, ...).
    - `axis`: Axis containing the simplex dimension.
    - `softness`: Controls the strength of the regularizer, defaults to 1.
    - `mode`: Controls the type of regularizer:
        - `entropic`: Entropic regularizer. Solved in closed form via softmax.
        - `euclidean`: Quadratic regularizer. Solved via the algorithm in https://arxiv.org/pdf/1309.1541.

    **Returns:**

    An Array of shape (..., [n], ...) representing the projected values onto the
    unit simplex along the specified axis.
    """
    axis = _canonicalize_axis(axis, x.ndim)
    _x = x / softness
    if mode == "entropic":
        _x = _x * ENTROPIC_CONSTANT
        soft_indices = jax.nn.softmax(_x, axis=axis)  # (..., [n], ...)
    elif mode == "euclidean":
        _x = _x * EUCLIDEAN_CONSTANT
        _x = jnp.moveaxis(_x, axis, -1)  # (..., ..., n)
        *batch_sizes, n = _x.shape
        _x = _x.reshape(-1, n)  # (B, n)
        soft_indices = jax.vmap(_projection_unit_simplex, in_axes=0)(_x)
        soft_indices = soft_indices.reshape(*batch_sizes, n)  # (..., ..., [n])
        soft_indices = jnp.moveaxis(soft_indices, -1, axis)  # (..., [n], ...)
    else:
        raise ValueError(f"Invalid mode: {mode}")
    return soft_indices


def argmax(
    x: Array,  # (..., n, ...)
    axis: int | None = None,
    keepdims: bool = False,
    softness: float = 1.0,
    mode: Literal["hard", "entropic", "euclidean"] = "entropic",
) -> SoftIndex:  # (..., {1}, ..., [n])
    """Performs a soft version of [jax.numpy.argmax](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.argmax.html)
    of `x` along the specified axis.

    **Arguments:**

    - `x`: Input Array of shape (..., n, ...).
    - `axis`: The axis along which to compute the argmax. If None, the input Array is
        flattened before computing the argmax. Defaults to None.
    - `keepdims`: If True, keeps the reduced dimension as a singleton {1}.
    - `softness`: Softness of the function, should be larger than zero. Defaults to 1.
    - `mode`: Controls the type of softening:
        - `hard`: Returns the result of jnp.argmax with a one-hot encoding of
            the indices.
        - `entropic`: Returns a softmax-based relaxation of the argmax.
        - `euclidean`: Returns an L2-projection-based relaxation of the argmax.

    **Returns:**

    A SoftIndex of shape (..., {1}, ..., [n]) (positive Array which sums to 1 over
    the last dimension).
    Represents the probability of an index corresponding to the argmax along the
    specified axis.

    !!! tip "Usage"
        This function can be used as a differentiable relaxation to
        [`softjax.argmax`][], enabling
        backpropagation through index selection steps in neural networks or
        optimization routines. However, note that the output is not a discrete index
        but a `SoftIndex`, which is a distribution over indices.
        Therefore, functions which operate on indices have to be adjusted accordingly
        to accept a SoftIndex, see e.g. [`softjax.max`][] for an example of using
        [`softjax.take_along_axis`][] to retrieve the soft maximum value via the
        `SoftIndex`.

    !!! caveat "Difference to jax.nn.softmax"
        Note that [`softjax.argmax`][] in `entropic` mode is not fully equivalent to
        [jax.nn.softmax](https://docs.jax.dev/en/latest/_autosummary/jax.nn.softmax.html)
        because it moves the probability dimension into the last axis
        (this is a convention in the `SoftIndex` data type).

    ??? example

        ```python
        x = jnp.array([[5, 3, 4], [2, 7, 6]])

        # Hard
        print("jnp:", jnp.argmax(x, axis=1))
        print("sj_hard:", sj.argmax(x, mode="hard", axis=1))

        # Entropic (Softmax projection)
        print("sj_entropic_low:", sj.argmax(x, mode="entropic", softness=0.01, axis=1))
        print("sj_entropic_high:", sj.argmax(x, mode="entropic", softness=1.0, axis=1))

        # Euclidean (L2 projection)
        print("sj_euclidean_low:", sj.argmax(x, mode="euclidean", softness=0.01,
            axis=1))
        print("sj_euclidean_high:", sj.argmax(x, mode="euclidean", softness=4.0,
            axis=1))
        ```

        ```
        jnp: [0 1]
        sj_hard: [[1. 0. 0.]
                  [0. 1. 0.]]
        sj_entropic_low: [[1.00000000e+000 1.38389653e-087 3.72007598e-044]
                          [7.12457641e-218 1.00000000e+000 3.72007598e-044]]
        sj_entropic_high: [[0.66524096 0.09003057 0.24472847]
                           [0.00490169 0.72747516 0.26762315]]
        sj_euclidean_low: [[1. 0. 0.]
                           [0. 1. 0.]]
        sj_euclidean_high: [[0.58333333 0.08333333 0.33333333]
                            [0.         0.625      0.375     ]]
        ```

    """
    if axis is None:
        x = jnp.ravel(x)
        axis = 0
    else:
        axis = _canonicalize_axis(axis, x.ndim)

    if mode == "hard":
        indices = jnp.argmax(x, axis=axis, keepdims=keepdims)
        soft_indices = jax.nn.one_hot(indices, num_classes=x.shape[axis], axis=-1)
    else:
        soft_indices = _projection_simplex(
            x, axis=axis, mode=mode, softness=softness
        )  # (..., [n], ...)
        soft_indices = jnp.moveaxis(soft_indices, axis, -1)  # (..., ..., [n])
        if keepdims:
            soft_indices = jnp.expand_dims(
                soft_indices, axis=axis
            )  # (..., 1, ..., [n])
    return soft_indices


def argmin(
    x: Array,  # (..., n, ...)
    axis: int | None = None,
    keepdims: bool = False,
    softness: float = 1.0,
    mode: Literal["hard", "entropic", "euclidean"] = "entropic",
) -> SoftIndex:  # (..., {1}, ..., [n])
    """Performs a soft version of [jax.numpy.argmin](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.argmin.html)
    of `x` along the specified axis.
    Implemented as [`softjax.argmax`][] on `-x`, see respective documentation for
    details.
    """
    return argmax(
        -x,
        axis=axis,
        mode=mode,
        softness=softness,
        keepdims=keepdims,
    )


@partial(jax.jit, static_argnames=("softness", "mode", "max_iter"))
def _projection_transport_polytope(
    cost: Array,  # (..., n, m)
    mu: Array,  # ([n],)
    nu: Array,  # ([m],)
    softness: float = 1.0,
    mode: Literal["entropic", "euclidean"] = "entropic",
    max_iter: int = 1000,
    use_sinkhorn: bool = True,
) -> SoftIndex:  # (..., [n], [m])
    """Projects a cost matrix onto the transport polytope between `mu` and `nu`.

    Solves the optimization problem:
        min_G <C, G> + softness * R(G)
        s.t. G 1_m = mu, G^T 1_n = nu, G >= 0
    where R(G) is the regularizer determined by `mode`.

    **Arguments:**

    - `cost`: Input cost Array of shape (..., n, m).
    - `mu`: Source marginal distribution Array of shape ([n],).
    - `nu`: Target marginal distribution Array of shape ([m],).
    - `softness`: Controls the strength of the regularizer, defaults to 1.
    - `mode`: Controls the type of regularizer:
        - `entropic`: Entropic regularizer. Solved via Sinkhorn (see [Sinkhorn
            Distances: Lightspeed Computation of Optimal Transportation Distances](https://arxiv.org/abs/1306.0895))
            or LBFGS.
        - `euclidean`: Quadratic regularizer. Solved via LBFGS, projecting onto Birkhoff
            polytope (see [Smooth and Sparse Optimal Transport](https://arxiv.org/abs/1710.06276)).
    - `max_iter`: Maximum number of iterations for the optimizer/Sinkhorn solver,
        defaults to 20.
    - `use_sinkhorn`: If True, run the custom Sinkhorn routine; otherwise rely on
        `jaxopt` projections. Defaults to True.

    **Returns:**

    A SoftIndex of shape (..., [n], [m]) (positive Array which sums to 1 over
    the last dimension).
    Represents the transport plan between the marginals `mu` and `nu`.
    """
    *batch_sizes, n, m = cost.shape
    cost = cost.reshape(-1, n, m)  # (B, n, m)
    cost = cost / softness  # (B, n, m)
    if mode == "entropic":
        cost = cost * ENTROPIC_CONSTANT
        if use_sinkhorn:
            proj_fn = lambda c: _sinkhorn(c, mu=mu, nu=nu, max_iter=max_iter)
        else:
            make_solver = lambda fun: LBFGS(
                fun=fun,
                tol=1e-3,
                maxiter=max_iter,
                linesearch="zoom",
                implicit_diff=True,
            )
            proj_fn = lambda c: kl_projection_transport(
                -c, (mu, nu), make_solver=make_solver, use_semi_dual=True
            )
    elif mode == "euclidean":
        cost = cost * EUCLIDEAN_OT_CONSTANT
        make_solver = lambda fun: LBFGS(
            fun=fun, tol=1e-3, maxiter=max_iter, linesearch="zoom", implicit_diff=True
        )
        proj_fn = lambda c: projection_transport(
            -c, (mu, nu), make_solver=make_solver, use_semi_dual=False
        )
    else:
        raise ValueError(f"Invalid mode: {mode}")
    Gamma = jax.vmap(proj_fn, in_axes=(0,))(cost)  # (B, n, m)
    y = Gamma * n  # (B, [n], [m])
    y = y.reshape(*batch_sizes, n, m)  # (..., [n], [m])
    return y


def argsort(
    x: Array,  # (..., n, ...)
    axis: int | None = None,
    descending: bool = False,
    softness: float = 1.0,
    mode: Literal["hard", "entropic", "euclidean"] = "entropic",
    fast: bool = True,
    max_iter: int = 1000,
) -> SoftIndex:  # (..., n, ..., [n])
    """Performs a soft version of [jax.numpy.argsort](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.argsort.html)
    of `x` along the specified axis.

    **Arguments:**

    - `x`: Input Array of shape (..., n, ...).
    - `axis`: The axis along which to compute the argsort operation. If None, the input
        Array is flattened before computing the argsort. Defaults to None.
    - `descending`: If True, sorts in descending order. Defaults to False (ascending).
    - `softness`: Softness of the function, should be larger than zero. Defaults to 1.
    - `mode` and `fast`: These two arguments control the type of softening:
        - `mode="hard"`: Returns the result of jnp.argsort with a one-hot encoding of
            the indices.
        - `fast=False` and `mode="entropic"`: Uses entropy-regularized optimal
            transport (implemented via Sinkhorn iterations) as in
            [Differentiable Ranks and Sorting using Optimal Transport](https://arxiv.org/pdf/1905.11885).
            Intuition: The sorted elements are selected by specifying n "anchors"
            and then transporting the ith-largest value to the ith-largest anchor.
            Can be slow for large `max_iter`.
        - `fast=False` and `mode="euclidean"`: Similar to entropic case, but using an
            L2-regularizer (implemented via LBFGS projection onto Birkhoff polytope) as
            in [Fast Differentiable Sorting and Ranking](https://arxiv.org/pdf/2002.08871).
        - `fast=True` and `mode="entropic"`: Uses the "SoftSort" operator proposed in
            [SoftSort: A Continuous Relaxation for the argsort Operator](https://arxiv.org/pdf/2006.16038).
            This initializes the cost matrix based on the absolute difference of `x` to
            the sorted values and then applies a single row normalization (instead of
            full Sinkhorn in OT).
            Note: Fast mode introduces gradient discontinuities when elements in `x` are
            not unique, but is much faster.
        - `fast=True` and `mode="euclidean"`: Similar to entropic fast case, but using
            a euclidean unit-simplex projection instead of softmax. To the best of our
            knowledge this variant is novel.
    - `max_iter`: Maximum number of iterations for the Sinkhorn algorithm if `mode` is
        "entropic", or for the projection onto the Birkhoff polytope if
        `mode` is "euclidean". Unused if `fast=True`.

    **Returns:**

    A SoftIndex of shape (..., n, ..., [n]) (positive Array which sums to 1 over
    the last dimension).
    The elements in (..., i, ..., [n]) represent a distribution over values in x for the
    ith smallest element along the specified axis.

    !!! tip "Computing the expectation"
        Computing the soft sorted values means taking the expectation of `x` under the
        SoftIndex distribution. Similar to how with normal indices you would do
            ```python
            sorted_x = jnp.take_along_axis(x, indices, axis=axis)
            ```
        we offer the equivalent soft version via
            ```python
            soft_sorted_x = sj.take_along_axis(x, soft_indices, axis=axis)
            ```
        This is what is done in [`softjax.sort`][].
    """
    if axis is None:
        x = jnp.ravel(x)
        axis = 0
    else:
        axis = _canonicalize_axis(axis, x.ndim)

    if mode == "hard":
        indices = jnp.argsort(x, axis=axis, descending=descending)  # (..., n, ...)
        soft_indices = jax.nn.one_hot(
            indices, num_classes=x.shape[axis], axis=-1
        )  # (..., n, ..., [n])
    else:
        x = jnp.moveaxis(x, axis, -1)  # (..., ..., n)
        *batch_dims, n = x.shape

        if fast:
            anchors = jnp.sort(x, axis=-1, descending=descending)  # (..., ..., n)
            cost = jnp.abs(x[..., :, None] - anchors[..., None, :])  # (..., ..., n, n)
            soft_indices = _projection_simplex(
                -cost, axis=-2, softness=softness, mode=mode
            )  # (..., ..., [n], n)
        else:
            anchors = jnp.linspace(0, n - 1, n, dtype=x.dtype)  # (n,)
            if descending:
                anchors = anchors[::-1]  # (n,)
            anchors = jnp.broadcast_to(anchors, (*batch_dims, n))  # (..., ..., n)

            # Note: could also do absolute difference or Huber
            cost = (
                jnp.abs(x[..., :, None] - anchors[..., None, :])
            ) ** 2  # (..., ..., n, n)

            mu = jnp.ones((n,), jnp.float_) / n  # ([n],)
            nu = jnp.ones((n,), jnp.float_) / n  # ([n],)

            soft_indices = _projection_transport_polytope(
                cost=cost, mu=mu, nu=nu, softness=softness, mode=mode, max_iter=max_iter
            )  # (..., ..., [n], [n])

        soft_indices = jnp.moveaxis(soft_indices, -1, axis)  # (..., [n], ..., [n])
    return soft_indices


def argtop_k(
    x: Array,  # (..., n, ...)
    k: int,
    axis: int = -1,
    softness: float = 1.0,
    mode: Literal["hard", "entropic", "euclidean"] = "entropic",
    fast: bool = True,
    max_iter: int = 1000,
) -> SoftIndex:  # (..., k, ..., [n])
    """Computes the soft argtop_k operation of `x` along the specified axis.

    **Arguments:**

    - `x`: Input Array of shape (..., n, ...).
    - `k`: The number of top elements to select.
    - `axis`: The axis along which to compute the top_k operation. Defaults to -1.
    - `softness`: Softness of the function, should be larger than zero. Defaults to 1.
    - `mode` and `fast`: These two arguments control the type of softening:
        - `mode="hard"`: Returns the result of jax.lax.top_k with a one-hot encoding of
            the indices.
        - `fast=False` and `mode="entropic"`: Uses entropy-regularized optimal
            transport (implemented via Sinkhorn iterations) as in
            [Differentiable Top-k with Optimal Transport](https://papers.nips.cc/paper/2020/file/ec24a54d62ce57ba93a531b460fa8d18-Paper.pdf).
            Intuition: The top-k elements are selected by specifying k+1 "anchors"
            and then transporting the top_k values to the top k anchors, and the
            remaining (n-k) values to the last anchor.
            Can be slow for large `max_iter`.
        - `fast=False` and `mode="euclidean"`: Similar to entropic case, but using an
            L2-regularizer (implemented via projection onto Birkhoff polytope).
            This version combines the approaches in [Fast Differentiable Sorting and Ranking](https://arxiv.org/pdf/2002.08871)
            (L2 regularizer for sorting) and [Differentiable Top-k with Optimal Transport](https://papers.nips.cc/paper/2020/file/ec24a54d62ce57ba93a531b460fa8d18-Paper.pdf)
            (entropic regularizer for top-k).
        - `fast=True` and `mode="entropic"`: Uses the "SoftSort" operator proposed  in
            [SoftSort: A Continuous Relaxation for the argsort Operator](https://arxiv.org/pdf/2006.16038).
            This initializes the cost matrix based on the absolute difference of `x` to
            the sorted values and then applies a single row normalization (instead of
            full Sinkhorn in OT).
            Because this is very fast we do a full soft argsort and then take the top-k
            elements.
            Note: Fast mode introduces gradient discontinuities when elements in `x` are
            not unique, but is much faster.
        - `fast=True` and `mode="euclidean"`: Similar to entropic fast case, but using
            a euclidean unit-simplex projection instead of softmax. To the best of our
            knowledge this variant is novel.
    - `max_iter`: Maximum number of iterations for the Sinkhorn algorithm if `mode` is
        "entropic", or for the projection onto the Birkhoff polytope if
        `mode` is "euclidean". Unused if `fast=True`.

    **Returns:**

    A SoftIndex of shape (..., k, ..., [n]) (positive Array which sums to 1 over
    the last dimension).
    The elements in (..., i, ..., [n]) represent a distribution over values in x for the
    ith largest element along the specified axis.
    """
    if axis is None:
        x = jnp.ravel(x)
        axis = 0
    else:
        axis = _canonicalize_axis(axis, x.ndim)
    x = jnp.moveaxis(x, axis, -1)  # (..., ..., n)

    if mode == "hard":
        soft_values, indices = jax.lax.top_k(x, k=k)  # (..., ..., k), (..., ..., k)
        soft_values = jnp.moveaxis(soft_values, -1, axis)  # (..., k, ...)
        soft_indices = jax.nn.one_hot(
            indices, num_classes=x.shape[-1], axis=-1
        )  # (..., ..., k, [n])
        soft_indices = jnp.moveaxis(soft_indices, -2, axis)  # (..., k, ..., [n])
    else:
        if fast:
            soft_indices = argsort(
                x,
                axis=axis,
                mode=mode,
                fast=fast,
                max_iter=max_iter,
                softness=softness,
                descending=True,
            )  # (..., n, ..., [n])
            soft_indices = jnp.moveaxis(soft_indices, axis, -1)  # (..., ..., [n], n)
        else:
            *batch_dims, n = x.shape

            anchors = jnp.linspace(0, k, k + 1, dtype=x.dtype)  # (k+1,)
            anchors = anchors[::-1]  # (k+1,)
            anchors = jnp.broadcast_to(anchors, (*batch_dims, k + 1))  # (..., ..., k+1)

            # Note: could also do absolute difference or Huber
            cost = (
                jnp.abs(x[..., :, None] - anchors[..., None, :])
            ) ** 2  # (..., ..., n, k+1)

            mu = jnp.ones((n,), jnp.float_) / n  # ([n],)
            nu = jnp.concatenate(
                [jnp.ones(k, jnp.float_) / n, jnp.array((n - k) / n)[None]]
            )  # ([k+1],)

            soft_indices = _projection_transport_polytope(
                cost=cost, mu=mu, nu=nu, softness=softness, mode=mode, max_iter=max_iter
            )  # (..., ..., [n], [k+1])

        soft_indices = soft_indices[..., :k]  # (..., ..., [n], k)
        soft_indices = jnp.moveaxis(soft_indices, -1, axis)  # (..., k, ..., [n])
    return soft_indices


def ranking(
    x: Array,  # (..., n, ...)
    axis: int | None = None,
    softness: float = 1.0,
    mode: Literal["hard", "entropic", "euclidean"] = "entropic",
    fast: bool = True,
    max_iter: int = 1000,
    descending: bool = True,
) -> Array:  # (..., n, ...)
    """Computes the soft rankings of `x` along the specified axis.

    **Arguments:**

    - `x`: Input Array of shape (..., n, ...).
    - `axis`: The axis along which to compute the ranking operation. If None, the input
        Array is flattened before computing the ranking. Defaults to None.
    - `descending`: If True, larger inputs receive smaller ranks (best rank is 0). If
        False, ranks increase with the input values.
    - `softness`: Softness of the function, should be larger than zero. Defaults to 1.
    - `mode` and `fast`: These two arguments control the behavior of the ranking
        operation:
        - `mode="hard"`: Returns ranking computed as two jnp.argsort calls.
        - `fast=False` and `mode="entropic"`: Uses entropy-regularized optimal
            transport (implemented via Sinkhorn iterations) as in
            [Differentiable Ranks and Sorting using Optimal Transport](https://arxiv.org/pdf/1905.11885).
            Intuition: We can use the transportation plan obtained in soft sorting for
            ranking by transporting the sorted ranks (0, 1, ..., n-1) back to the
            ranks of the original values.
            Can be slow for large `max_iter`.
        - `fast=False` and `mode="euclidean"`: Similar to entropic case, but using an
            L2-regularizer (implemented via projection onto Birkhoff polytope) as in
            [Fast Differentiable Sorting and Ranking](https://arxiv.org/pdf/2002.08871).
        - `fast=True` and `mode="entropic"`: Uses an adaptation of the "SoftSort"
            operator proposed in [SoftSort: A Continuous Relaxation for the argsort Operator](https://arxiv.org/pdf/2006.16038).
            This initializes the cost matrix based on the absolute difference of `x` to
            the sorted values and then we crucially apply a single **column**
            normalization (instead of of row normalization in the original paper).
            This makes the resulting matrix a unimodal column stochastic matrix which is
            better suited for soft ranking.
            Note: Fast mode introduces gradient discontinuities when elements in `x` are
            not unique, but is much faster.
        - `fast=True` and `mode="euclidean"`: Similar to entropic fast case, but using
            a euclidean unit-simplex projection instead of softmax. To the best of our
            knowledge this variant is novel.
    - `max_iter`: Maximum number of iterations for the Sinkhorn algorithm if `mode` is
        "entropic", or for the projection onto the Birkhoff polytope if
        `mode` is "euclidean". Unused if `fast=True`.

    **Returns:**

    A positive Array of shape (..., n, ...) with values in [0, n-1].
    The elements in (..., i, ...) represent the soft rank of the ith element along the
    specified axis.
    """
    if axis is None:
        x = jnp.ravel(x)
        axis = 0
    else:
        axis = _canonicalize_axis(axis, x.ndim)

    if mode == "hard":
        indices = jnp.argsort(x, axis=axis, descending=descending)  # (..., n, ...)
        indices = indices.astype(jnp.float_)
        rankings = jnp.argsort(
            indices, axis=axis, descending=descending
        )  # (..., n, ...)
        rankings = rankings.astype(jnp.float_)
    else:
        x = jnp.moveaxis(x, axis, -1)  # (..., ..., n)
        *batch_dims, n = x.shape

        if fast:
            anchors = jnp.sort(x, axis=-1, descending=descending)  # (..., ..., n)
            cost = jnp.abs(x[..., :, None] - anchors[..., None, :])  # (..., ..., n, n)
            soft_indices = _projection_simplex(
                -cost, axis=-1, softness=softness, mode=mode
            )  # (..., ..., n, [n])
        else:
            anchors = jnp.linspace(0, n - 1, n, dtype=x.dtype)  # (n,)
            if descending:
                anchors = anchors[::-1]  # (n,)
            anchors = jnp.broadcast_to(anchors, (*batch_dims, n))  # (..., ..., n)

            cost = jnp.pow(
                jnp.abs(x[..., :, None] - anchors[..., None, :]), 2.0
            )  # (..., ..., n, n)

            mu = jnp.ones((n,), jnp.float_) / n  # ([n],)
            nu = jnp.ones((n,), jnp.float_) / n  # ([n],)

            soft_indices = _projection_transport_polytope(
                cost=cost, mu=mu, nu=nu, softness=softness, mode=mode, max_iter=max_iter
            )  # (..., ..., [n], [n])

        soft_rank_indices = jnp.moveaxis(soft_indices, -1, axis)  # (..., [n], ..., n)

        nums = jnp.arange(0, soft_rank_indices.shape[axis], dtype=x.dtype)  # (n,)
        rankings = jnp.tensordot(
            soft_rank_indices, nums, axes=([axis], [0])
        )  # (..., ..., n)
        rankings = jnp.moveaxis(rankings, -1, axis)  # (..., n, ...)
    return rankings


def argmedian(
    x: Array,  # (..., n, ...)
    axis: int | None = None,
    keepdims: bool = False,
    softness: float = 1.0,
    mode: Literal["hard", "entropic", "euclidean"] = "entropic",
    fast: bool = True,
    max_iter: int = 1000,
) -> SoftIndex:  # (..., {1}, ..., [n])]
    """Computes the soft argmedian of `x` along the specified axis.

    **Arguments:**

    - `x`: Input Array of shape (..., n, ...).
    - `axis`: The axis along which to compute the median. If None, the input Array is
        flattened before computing the median. Defaults to None.
    - `keepdims`: If True, keeps the reduced dimension as a singleton {1}.
    - `softness`: Softness of the function, should be larger than zero. Defaults to 1.
    - `mode` and `fast`: These two arguments control the behavior of the median:
        - `mode="hard"`: Returns the result of jnp.median with a one-hot encoding of
            the indices. On ties, it returns a uniform distribution over all median
            indices.
        - `fast=False` and `mode="entropic"`: Uses entropy-regularized optimal
            transport (implemented via Sinkhorn iterations).
            We adapt the approach in [Differentiable Ranks and Sorting
            using Optimal Transport](https://arxiv.org/pdf/1905.11885) and
            [Differentiable Top-k with Optimal Transport](https://papers.nips.cc/paper/2020/file/ec24a54d62ce57ba93a531b460fa8d18-Paper.pdf)
            to the median operation by carefully adjusting the cost matrix and
            marginals.
            Intuition: There are three "anchors", the median is transported onto one
            anchor, and all the larger and smaller elements are transported to the other
            two anchors, respectively.
            Can be slow for large `max_iter`.
        - `fast=False` and `mode="euclidean"`: Similar to entropic case, but using an
            L2-regularizer (implemented via projection onto Birkhoff polytope).
        - `fast=True` and `mode="entropic"`: This formulation a well-known soft median
            operation based on the interpretation of the median as the minimizer of
            absolute deviations. The softening is then achieved by replacing the argmax
            operator with a softmax. Note, that this also has close ties to the
            "SoftSort" operator from [SoftSort: A Continuous Relaxation for the argsort Operator](https://arxiv.org/pdf/2006.16038).
            Note: Fast mode introduces gradient discontinuities when elements in `x` are
            not unique, but is much faster.
        - `fast=True` and `mode="euclidean"`: Similar to entropic fast case, but using
            a euclidean unit-simplex projection instead of softmax.
    - `max_iter`: Maximum number of iterations for the Sinkhorn algorithm if `mode` is
        "entropic", or for the projection onto the Birkhoff polytope if
        `mode` is "euclidean". Unused if `fast=True`.

    **Returns:**

    A SoftIndex of shape (..., {1}, ..., [n]) (positive Array which sums
    to 1 over the last dimension).
    The elements in (..., 0, ...) represent a distribution over values in x being
    the median along the specified axis.
    """
    if axis is None:
        x = jnp.ravel(x)
        axis = 0
    else:
        axis = _canonicalize_axis(axis, x.ndim)

    x_last = jnp.moveaxis(x, axis, -1)  # (..., ..., n)
    if mode == "hard":
        pairwise = jnp.abs(
            x_last[..., :, None] - x_last[..., None, :]
        )  # (..., ..., n, n)
        cost = jnp.mean(pairwise, axis=-1)  # (..., ..., n)
        min_cost = jnp.min(cost, axis=-1, keepdims=True)  # (..., ..., 1)
        argmed = jnp.equal(cost, min_cost).astype(jnp.float_)  # (..., ..., [n])
    else:
        if fast:
            cost = jnp.abs(
                x_last[..., :, None] - x_last[..., None, :]
            )  # (..., ..., n, n)
            cost = jnp.sum(cost, axis=-1)  # (..., ..., n)
            argmed = _projection_simplex(
                -cost, axis=-1, softness=softness, mode=mode
            )  # (..., ..., [n])
        else:
            *batch_dims, n = x_last.shape
            m = 3
            anchors = jnp.linspace(0, m - 1, m, dtype=x.dtype)  # (m,)
            anchors = jnp.broadcast_to(anchors, (*batch_dims, m))  # (..., ..., m)

            cost = jnp.pow(
                jnp.abs(x_last[..., :, None] - anchors[..., None, :]), 2.0
            )  # (..., ..., n, m)

            mu = jnp.ones((n,), jnp.float_) / n  # ([n],)
            nu = jnp.array(
                [(n - 1) / (2 * n), 1 / n, (n - 1) / (2 * n)], jnp.float_
            )  # ([m],)

            soft_indices = _projection_transport_polytope(
                cost=cost, mu=mu, nu=nu, softness=softness, mode=mode, max_iter=max_iter
            )  # (..., ..., [n], [m])
            argmed = soft_indices[..., :, 1]  # (..., ..., [n])

    argmed = argmed / jnp.sum(argmed, axis=-1, keepdims=True)  # Normalize distribution
    if keepdims:
        argmed = jnp.expand_dims(argmed, axis=axis)  # (..., 1, ..., [n])

    return argmed


def quantile():
    """Placeholder for a differentiable `quantile` implementation."""
    raise NotImplementedError


def argpartition():
    """Placeholder for a differentiable `argpartition` implementation."""
    raise NotImplementedError


# Array, SoftIndex -> Array


def take_along_axis(
    x: Array,  # (..., n, ...)
    soft_indices: SoftIndex,  # (..., k, ..., [n])
    axis: int = -1,
) -> Array:  # (..., k, ...)
    """Performs a soft version of [jax.numpy.take_along_axis](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.take_along_axis.html)
    via a weighted dot product.

    !!! example "Relation to `jnp.take_along_axis`"

        ```python
        x = jnp.array([[1, 2, 3], [4, 5, 6]])

        indices = jnp.array([[0, 2], [1, 0]])
        print(jnp.take_along_axis(x, indices, axis=1))

        indices_onehot = jax.nn.one_hot(indices, x.shape[1])
        print(sj.take_along_axis(x, indices_onehot, axis=1))
        ```

        ```
        [[1. 3.]
         [5. 4.]]
        [[1. 3.]
         [5. 4.]]
        ```

    ??? example "Interaction with [`softjax.argmax`][]"

        ```python
        x = jnp.array([[5, 3, 4], [2, 7, 6]])

        indices = jnp.argmin(x, axis=1, keepdims=True)
        print("argmin_jnp:", jnp.take_along_axis(x, indices, axis=1))

        indices_onehot = sj.argmin(x, axis=1, mode="hard", keepdims=True)
        print("argmin_val_onehot:", sj.take_along_axis(x, indices_onehot, axis=1))

        indices_soft = sj.argmin(x, axis=1, mode="entropic", softness=1.0,
            keepdims=True)
        print("argmin_val_soft:", sj.take_along_axis(x, indices_soft, axis=1))
        ```

        ```
        argmin_jnp: [[3]
                     [2]]
        argmin_val_onehot: [[3.]
                            [2.]]
        argmin_val_soft: [[3.42478962]
                          [2.10433824]]
        ```

    ??? example "Interaction with [`softjax.argsort`][]"

        ```python
        x = jnp.array([[5, 3, 4], [2, 7, 6]])

        indices = jnp.argsort(x, axis=1)
        print("sorted_jnp:", jnp.take_along_axis(x, indices, axis=1))

        indices_onehot = sj.argsort(x, axis=1, mode="hard")
        print("sorted_sj_hard:", sj.take_along_axis(x, indices_onehot, axis=1))

        indices_soft = sj.argsort(x, axis=1, mode="entropic", softness=1.0)
        print("sorted_sj_soft:", sj.take_along_axis(x, indices_soft, axis=1))
        ```

        ```
        sorted_jnp: [[3 4 5]
                     [2 6 7]]
        sorted_sj_hard: [[3. 4. 5.]
                         [2. 6. 7.]]
        sorted_sj_soft: [[3.2918137  4.         4.7081863 ]
                         [2.00000045 6.26894107 6.73105858]]
        ```

    **Arguments:**

    - `x`: Input Array of shape (..., n, ...).
    - `soft_indices`: A SoftIndex of shape (..., k, ..., [n]) (positive Array which
        sums to 1 over the last dimension).
    - `axis`: Axis along which to apply the soft index. Defaults to -1.

    **Returns:**

    Array of shape (..., k, ...), representing the result after soft selection along
    the specified axis.
    """
    if x.ndim + 1 != soft_indices.ndim:
        raise ValueError(
            f"Input x and soft_indices must have compatible dimensions, "
            f"but got x.ndim={x.ndim} and soft_indices.ndim={soft_indices.ndim}. "
            f"Should be x.ndim + 1 == soft_indices.ndim."
        )
    axis = _canonicalize_axis(axis, x.ndim)
    x = jnp.moveaxis(x, axis, -1)  # (..., ..., n)
    x = jnp.expand_dims(x, axis)  # (..., 1, ..., n)
    dotprod = jnp.sum(x * soft_indices, axis=-1)  # (..., k, ...)
    return dotprod


def take(
    x: Array,  # (..., n, ...)
    soft_indices: SoftIndex,  # (k, [n])
    axis: int | None = None,
) -> Array:  # (..., k, ...)
    """Performs a soft version of [jax.numpy.take](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.take.html)
    via a weighted dot product.

    **Arguments:**

    - `x`: Input Array of shape (..., n, ...).
    - `soft_indices`: A SoftIndex of shape (k, [n]) (positive Array which
        sums to 1 over the last dimension).
    - `axis`: Axis along which to apply the soft index. If None, the input is
        flattened. Defaults to None.

    **Returns:**

    Array of shape (..., k, ...) after soft selection.
    """
    if soft_indices.ndim != 2:
        raise ValueError(
            f"soft_indices must be of shape (k, [n]), but got shape {soft_indices.shape}."
        )
    if axis is None:
        x = jnp.ravel(x)
        axis = 0
    else:
        axis = _canonicalize_axis(axis, x.ndim)
        x = jnp.moveaxis(x, axis, -1)  # (..., ..., n)
    soft_indices = jnp.reshape(
        soft_indices, (1,) * (x.ndim - 1) + soft_indices.shape
    )  # (1..., 1..., k, [n])
    x = jnp.expand_dims(x, axis)  # (..., 1, ..., n)
    soft_indices = jnp.moveaxis(soft_indices, -2, axis)  # (1..., k, 1..., [n])
    y = jnp.sum(x * soft_indices, axis=-1)  # (..., k, ...)
    return y


def choose(
    soft_indices: SoftIndex,  # (..., [n])
    choices: Array,  # (n, ...)
) -> Array:  # (...,)
    """Performs a soft version of [jax.numpy.choose](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.choose.html)
    via a weighted dot product.

    **Arguments:**

    - `soft_indices`: A SoftIndex of shape (..., [n]) (positive Array which
        sums to 1 over the last dimension). Represents the weights for each choice.
    - `choices`: Array of shape (n, ...) supplying the values to mix.

    **Returns:**

    Array of shape (..., ...) after softly selecting among `choices`.
    """
    if soft_indices.ndim != choices.ndim or soft_indices.shape[-1] != choices.shape[0]:
        raise ValueError(
            f"soft_indices and choices must have compatible dimensions, but got "
            f"soft_indices.shape={soft_indices.shape} and choices.shape={choices.shape}. "
            f"Should be soft_indices.shape=(..., [n]) and choices.shape=(n, ...)."
        )
    tgt_shape = jnp.broadcast_shapes(choices.shape[1:], soft_indices.shape[:-1])
    choices_bcast = jnp.broadcast_to(choices, (choices.shape[0], *tgt_shape))
    choices_bcast = jnp.moveaxis(choices_bcast, 0, -1)  # (..., C)
    result = jnp.sum(choices_bcast * soft_indices, axis=-1)  # (...)
    return result


def dynamic_index_in_dim(
    x: Array,  # (..., n, ...)
    soft_index: SoftIndex,  # ([n],)
    axis: int = 0,
    keepdims: bool = True,
) -> Array:  # (..., {1}, ...)
    """Performs a soft version of [jax.lax.dynamic_index_in_dim](https://docs.jax.dev/en/latest/_autosummary/jax.lax.dynamic_index_in_dim.html)
    via a weighted dot product.

    **Arguments:**

    - `x`: Input Array of shape (..., n, ...).
    - `soft_indices`: A SoftIndex of shape ([n],) (positive Array which
        sums to 1 over the last dimension).
    - `axis`: Axis along which to apply the soft index. Defaults to 0.
    - `keepdims`: If True, keeps the reduced dimension as a singleton {1}.

    **Returns:**

    Array after soft indexing, shape (..., {1}, ...).
    """
    axis = _canonicalize_axis(axis, x.ndim)
    if x.shape[axis] != soft_index.shape[0]:
        raise ValueError(
            f"Dimension mismatch between x and soft_index along axis {axis}: "
            f"x.shape[{axis}]={x.shape[axis]} vs soft_index.shape[0]={soft_index.shape[0]}"
        )
    x = jnp.moveaxis(x, axis, -1)  # (..., ..., n)
    x_reshaped = jnp.reshape(x, (-1, x.shape[-1]))  # (B, n)
    dotprod = jnp.sum(x_reshaped * soft_index[None, :], axis=-1)  # (B,)
    y = jnp.reshape(dotprod, x.shape[:-1])  # (..., ...)
    if keepdims:
        y = jnp.expand_dims(dotprod, axis=axis)  # (..., 1, ...)
    return y


def dynamic_slice_in_dim(
    x: Array,  # (..., n, ...)
    soft_start_index: SoftIndex,  # ([n],)
    slice_size: int,
    axis: int = 0,
) -> Array:  # (..., slice_size, ...)
    """Performs a soft version of [jax.lax.dynamic_slice_in_dim](https://docs.jax.dev/en/latest/_autosummary/jax.lax.dynamic_slice_in_dim.html)
    via a weighted dot product.

    **Arguments:**

    - `x`: Input Array of shape (..., n, ...).
    - `soft_indices`: A SoftIndex of shape ([n],) (positive Array which
        sums to 1 over the last dimension).
    - `slice_size`: Length of the slice to extract.
    - `axis`: Axis along which to apply the soft slice. Defaults to 0.

    **Returns:**

    Array of shape (..., slice_size, ...) after soft slicing.
    """
    axis = _canonicalize_axis(axis, x.ndim)
    assert x.shape[axis] >= slice_size > 0

    x_last = jnp.moveaxis(x, axis, -1)  # (..., ..., n)
    t_idx = jnp.arange(slice_size)

    def one_step(t: Array) -> Array:
        # TODO: add padding option instead of wraparound
        rolled = jnp.roll(x_last, shift=-t, axis=-1)  # (..., n)
        return jnp.einsum("...n,n->...", rolled, soft_start_index)  # (...)

    y_stack = jax.vmap(one_step)(t_idx)  # (slice_size, ...)
    y_last = jnp.moveaxis(y_stack, 0, -1)  # (..., slice_size)
    y = jnp.moveaxis(y_last, -1, axis)  # (..., slice_size, ...)
    return y


def dynamic_slice(
    x: Array,  # (n_1, n_2, ..., n_k)
    soft_start_indices: Sequence[SoftIndex],  # [([n_1],), ([n_2],), ..., ([n_k],)]
    slice_sizes: Sequence[int],  # [l_1, l_2, ..., l_k]
) -> Array:  # (l_1, l_2, ..., l_k)
    """Performs a soft version of [jax.lax.dynamic_slice](https://docs.jax.dev/en/latest/_autosummary/jax.lax.dynamic_slice.html)
    via a weighted dot product.

    **Arguments:**

    - `x`: Input Array of shape (n_1, n_2, ..., n_k).
    - `soft_start_indices`: A list of SoftIndices of shape ([n_i],) (positive Arrays
        which sums to 1).
    Sequence of SoftIndex distributions of shapes
        ([n_1],), ([n_2],), ..., ([n_k]) each summing to 1.
    - `slice_sizes`: Sequence of slice lengths for each dimension.

    **Returns:**

    Array of shape (l_1, l_2, ..., l_k) after soft slicing.
    """
    assert len(soft_start_indices) == len(slice_sizes) == x.ndim
    y = x
    for axis, (soft_start_index, slice_size) in enumerate(
        zip(soft_start_indices, slice_sizes)
    ):
        y = dynamic_slice_in_dim(
            y,
            soft_start_index=soft_start_index,
            slice_size=slice_size,
            axis=axis,
        )
    return y


def gather():
    """Placeholder for a differentiable `jax.lax.gather` analogue."""
    raise NotImplementedError


def extract():
    """Placeholder for a differentiable `jax.numpy.extract` analogue."""
    raise NotImplementedError


def put():
    """Placeholder for a differentiable `jax.numpy.put` analogue."""
    raise NotImplementedError


def put_along_axis():
    """Placeholder for a differentiable `jax.numpy.put_along_axis` analogue."""
    raise NotImplementedError


def index_in_dim():
    """Placeholder for a static `index_in_dim` variant; soft indices are
    dynamic-only."""
    raise NotImplementedError


def slice():
    """Placeholder for a static `slice` variant; not applicable to soft indices."""
    raise NotImplementedError


def slice_in_dim():
    """Placeholder for a static `slice_in_dim` variant; not applicable to soft
    indices."""
    raise NotImplementedError


def max(
    x: Array,  # (..., n, ...)
    axis: int | None = None,
    keepdims: bool = False,
    softness: float = 1.0,
    mode: Literal["hard", "entropic", "euclidean"] = "entropic",
) -> Array:  # (..., {1}, ...)
    """Performs a soft version of [jax.numpy.max](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.max.html)
    of `x` along the specified axis.
    Implemented as [`softjax.argmax`][] followed by [`softjax.take_along_axis`][], see
    respective documentations for details.

    **Returns:**

    Array of shape (..., {1}, ...) representing the soft maximum of `x` along the
    specified axis.
    """
    if axis is None:
        x = jnp.ravel(x)
        axis = 0
    else:
        axis = _canonicalize_axis(axis, x.ndim)

    if mode == "hard":
        max_val = jnp.max(x, axis=axis, keepdims=keepdims)
    else:
        soft_indices = argmax(
            x,
            axis=axis,
            keepdims=True,
            softness=softness,
            mode=mode,
        )  # (..., 1, ..., [n])
        max_val = take_along_axis(x, soft_indices, axis=axis)  # (..., 1, ...)
        if not keepdims:
            max_val = jnp.squeeze(max_val, axis=axis)  # (..., ...)
    return max_val


def min(
    x: Array,  # (..., n, ...)
    axis: int | None = None,
    softness: float = 1.0,
    mode: Literal["hard", "entropic", "euclidean"] = "entropic",
    keepdims: bool = False,
) -> Array:  # (..., {1}, ...)
    """Performs a soft version of [jax.numpy.min](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.min.html)
    of `x` along the specified axis.
    Implemented as -[`softjax.max`][] on `-x`, see respective documentation for details.

    **Returns:**

    Array of shape (..., {1}, ...) representing the soft minimum of `x` along the
    specified axis.
    """
    return -max(-x, axis=axis, softness=softness, mode=mode, keepdims=keepdims)


def sort(
    x: Array,  # (..., n, ...)
    axis: int | None = None,
    descending: bool = False,
    softness: float = 1.0,
    mode: Literal["hard", "entropic", "euclidean"] = "entropic",
    fast: bool = True,
    max_iter: int = 1000,
) -> Array:  # (..., n, ...)
    """Performs a soft version of [jax.numpy.sort](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.sort.html)
    of `x` along the specified axis.
    Implemented as [`softjax.argsort`][] followed by [`softjax.take_along_axis`][], see
    respective documentations for details.

    **Returns:**

    Array of shape (..., n, ...) representing the soft sorted values of `x` along the
    specified axis.
    """
    if axis is None:
        x = jnp.ravel(x)
        axis = 0
    else:
        axis = _canonicalize_axis(axis, x.ndim)

    if mode == "hard":
        soft_values = jnp.sort(x, axis=axis, descending=descending)
    else:
        soft_indices = argsort(
            x=x,
            axis=axis,
            descending=descending,
            softness=softness,
            mode=mode,
            fast=fast,
            max_iter=max_iter,
        )  # (..., n, ..., [n])
        soft_values = take_along_axis(x, soft_indices, axis=axis)
    return soft_values  # (..., n, ...)


def top_k(
    x: Array,  # (..., n, ...)
    k: int,
    axis: int = -1,
    softness: float = 1.0,
    mode: Literal["hard", "entropic", "euclidean"] = "entropic",
    fast: bool = True,
    max_iter: int = 1000,
) -> tuple[Array, SoftIndex]:  # (..., k, ...), (..., k, ..., [n])
    """Performs a soft version of [jax.lax.top_k](https://docs.jax.dev/en/latest/_autosummary/jax.lax.top_k.html)
    of `x` along the specified axis.
    Implemented as [`softjax.argtop_k`][] followed by [`softjax.take_along_axis`][], see
    respective documentations for details.

    **Returns:**

    - `soft_values`: Top-k values of `x`, shape (..., k, ...).
    - `soft_indices`: SoftIndex of shape (..., k, ..., [n]) (positive Array which sums
        to 1 over the last dimension). Represents the soft indices of the top-k values.
    """
    axis = _canonicalize_axis(axis, x.ndim)
    soft_indices = argtop_k(
        x=x,
        k=k,
        axis=axis,
        softness=softness,
        mode=mode,
        fast=fast,
        max_iter=max_iter,
    )  # (..., k, ..., [n])
    soft_values = take_along_axis(x, soft_indices, axis=axis)  # (..., k, ...)
    return soft_values, soft_indices


def median(
    x: Array,  # (..., n, ...)
    axis: int | None = None,
    keepdims: bool = False,
    softness: float = 1.0,
    mode: Literal["hard", "entropic", "euclidean"] = "entropic",
    fast: bool = True,
    max_iter: int = 1000,
) -> Array:  # (..., {1}, ...)
    """Performs a soft version of [jnp.median](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.median.html)
    of `x` along the specified axis.
    Implemented as [`softjax.argmedian`][] followed by [`softjax.take_along_axis`][],
    see respective documentations for details.

    **Returns:**

    An Array of shape (..., {1}, ...), representing the soft median values along the
    specified axis.
    """
    if axis is None:
        x = jnp.ravel(x)
        axis = 0
    else:
        axis = _canonicalize_axis(axis, x.ndim)

    if mode == "hard":
        median_val = jnp.median(x, axis=axis, keepdims=keepdims)
    else:
        soft_indices = argmedian(
            x,
            axis=axis,
            keepdims=True,
            softness=softness,
            mode=mode,
            fast=fast,
            max_iter=max_iter,
        )  # (..., 1, ..., [n])
        median_val = take_along_axis(x, soft_indices, axis=axis)  # (..., 1, ...)
        if not keepdims:
            median_val = jnp.squeeze(median_val, axis=axis)  # (..., ...)
    return median_val


def median_newton(
    x: Array,  # (..., n, ...)
    axis: int | None = None,
    keepdims: bool = False,
    softness: float = 1.0,
    mode: Literal[
        "hard",
        "entropic",
        "pseudohuber",
        "euclidean",
        "cubic",
        "quintic",
    ] = "entropic",
    max_iter: int = 8,
    eps: float = 1e-12,
) -> Array:  # (..., {1}, ...)
    """Performs a soft version of [jax.numpy.median](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.median.html)
    of `x` along the specified axis.

    **Arguments:**

    - `x`: Input Array of shape (..., n, ...).
    - `axis`: Axis along which to compute the median. If None, the input is flattened.
        Defaults to None.
    - `keepdims`: If True, keeps the reduced dimension as a singleton {1}. Defaults to
        False.
    - `softness`: Softness of the score function, should be larger than zero. Defaults
        to 1.0.
    - `mode`: Smooth score choice:
        - `hard`: Returns `jnp.median`.
        - `sigmoid`, `pseudohuber`, `linear`, `cubic`, `quintic`: Smooth
            relaxations for the M-estimator using Newton steps. Defaults to `sigmoid`.
    - `max_iter`: Maximum number of Newton iterations in the M-estimator.
    - `eps`: Small constant added to the derivative to avoid division by zero.

    **Returns:**

    Array of shape (..., {1}, ...) representing the soft median of `x` along the
    specified axis.
    """
    if axis is None:
        x = jnp.ravel(x)
        axis = 0
    else:
        axis = _canonicalize_axis(axis, x.ndim)

    if mode == "hard":
        med = jnp.median(x, axis=axis, keepdims=keepdims)
    else:
        x_last = jnp.moveaxis(x, axis, -1)  # (..., n)

        # Good initialization: ordinary median (or mean if you prefer)
        med = jnp.median(x_last, axis=-1, keepdims=True)

        def newton_step(_, y):
            r = y - x_last  # (..., n)
            psi = sign(r, softness=softness, mode=mode)  # (..., n)
            dpsi = jax.grad(lambda r: jnp.sum(sign(r, softness=softness, mode=mode)))(r)
            f = psi.sum(axis=-1, keepdims=True)  # (..., 1)
            fp = dpsi.sum(axis=-1, keepdims=True) + eps  # (..., 1)
            return y - f / fp

        med = jax.lax.fori_loop(0, max_iter, newton_step, med)
        if not keepdims:
            med = jnp.squeeze(med, axis=-1)
        else:
            med = jnp.moveaxis(med, -1, axis)
    return med


# Array -> Array


def round(
    x: Array,
    softness: float = 1.0,
    mode: Literal[
        "hard", "entropic", "euclidean", "pseudohuber", "cubic", "quintic"
    ] = "entropic",
    neighbor_radius: int = 5,
) -> Array:
    """Performs a soft version of [jax.numpy.round](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.round.html).

    **Arguments:**

    - `x`: Input Array of any shape.
    - `softness`: Softness of the function, should be larger than zero. Defaults to 1.
    - `mode`: If "hard", applies `jnp.round`. Otherwise uses a sigmoid-based relaxation
        based on the algorithm described in https://arxiv.org/pdf/2504.19026v1.
        This function thereby inherits the different sigmoid modes "entropic",
        "euclidean", "pseudohuber", "cubic", or "quintic".
        Defaults to "entropic".
    - `neighbor_radius`: Number of neighbors on each side of the floor value to
        consider for the soft rounding. Defaults to 5.

    **Returns:**

    Result of applying soft elementwise rounding to `x`.
    """
    if mode == "hard":
        return jnp.round(x)
    else:
        center = jax.lax.stop_gradient(jnp.floor(x))
        offsets = jnp.arange(
            -neighbor_radius, neighbor_radius + 1, dtype=x.dtype
        )  # (M,)
        n = center[..., None] + offsets  # (..., M)

        w_left = _sigmoid(x[..., None] - (n - 0.5), softness=softness, mode=mode)
        w_right = _sigmoid(x[..., None] - (n + 0.5), softness=softness, mode=mode)
        return jnp.sum(n * (w_left - w_right), axis=-1)


def _softrelu(
    x: Array,
    softness: float = 1.0,
    mode: Literal[
        "entropic",
        "euclidean",
        "quartic",
        "gated_entropic",
        "gated_euclidean",
        "gated_cubic",
        "gated_quintic",
    ] = "entropic",
) -> Array:
    """Family of soft relaxations to ReLU used by `relu`/`clip`.

    **Arguments:**

    - `x`: Input Array.
    - `softness`: Softness of the function, should be larger than zero. Defaults to 1.
    - `mode`: Choice of smoothing kernel: "entropic", "euclidean", "quartic",
        "gated_entropic", "gated_euclidean", "gated_cubic", or "gated_quintic".
        Defaults to "entropic".

    **Returns:**

    Result of applying soft elementwise ReLU to `x`.
    """
    x = x / softness
    if mode == "entropic":
        y = jax.nn.softplus(x * ENTROPIC_CONSTANT) / ENTROPIC_CONSTANT
        # closed form integral of _sigmoid(x, mode="entropic")
    elif mode == "euclidean":
        y = jnp.polyval(jnp.array([0.5, 0.5, 0.125]), x)
        y = jnp.where(x < -0.5, 0.0, jnp.where(x < 0.5, y, x))
        # closed form integral of _sigmoid(x, mode="euclidean")
    elif mode == "quartic":
        y = jnp.polyval(jnp.array([-0.5, 0.0, 0.75, 0.5, 0.09375]), x)
        y = jnp.where(x < -0.5, 0.0, jnp.where(x < 0.5, y, x))
        # closed form integral of _sigmoid(x, mode="cubic")
    elif mode == "gated_entropic":
        y = x * _sigmoid(x, mode="entropic")
    elif mode == "gated_euclidean":
        y = x * _sigmoid(x, mode="euclidean")
    elif mode == "gated_cubic":
        y = x * _sigmoid(x, mode="cubic")
    elif mode == "gated_quintic":
        y = x * _sigmoid(x, mode="quintic")
    else:
        raise ValueError(f"Unknown mode '{mode}' for _softrelu.")
    y = y * softness
    return y


def relu(
    x: Array,
    softness: float = 1.0,
    mode: Literal[
        "hard",
        "entropic",
        "euclidean",
        "quartic",
        "gated_entropic",
        "gated_euclidean",
        "gated_cubic",
        "gated_quintic",
    ] = "entropic",
) -> Array:
    """Performs a soft version of [jax.nn.relu](https://docs.jax.dev/en/latest/_autosummary/jax.nn.relu.html).

    **Arguments:**

    - `x`: Input Array of any shape.
    - `softness`: Softness of the function, should be larger than zero. Defaults to 1.
    - `mode`: If "hard", applies `jax.nn.relu`. Otherwise uses "entropic", "euclidean",
        "quartic", "gated_entropic", "gated_euclidean", "gated_cubic", or "gated_quintic"
        relaxations. Defaults to "entropic".

    **Returns:**

    Result of applying soft elementwise ReLU to `x`.
    """
    if mode == "hard":
        return jax.nn.relu(x)
    else:
        return _softrelu(x, mode=mode, softness=softness)


def clip(
    x: Array,
    a: Array,
    b: Array,
    softness: float = 1.0,
    mode: Literal[
        "hard",
        "entropic",
        "euclidean",
        "quartic",
        "gated_entropic",
        "gated_euclidean",
        "gated_cubic",
        "gated_quintic",
    ] = "entropic",
) -> Array:
    """Performs a soft version of [jax.numpy.clip](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.clip.html).

    **Arguments:**

    - `x`: Input Array of any shape.
    - `a`: Lower bound scalar.
    - `b`: Upper bound scalar.
    - `softness`: Softness of the function, should be larger than zero. Defaults to 1.
    - `mode`: If "hard", applies `jnp.clip`. Otherwise uses "entropic", "euclidean",
        "quartic", "gated_entropic", "gated_euclidean", "gated_cubic", or "gated_quintic"
        relaxations. Defaults to "entropic".

    **Returns:**

    Result of applying soft elementwise clipping to `x`.
    """
    if mode == "hard":
        return jnp.clip(x, a, b)
    else:
        tmp1 = _softrelu(x - a, mode=mode, softness=softness)
        tmp2 = _softrelu(x - b, mode=mode, softness=softness)
        return a + tmp1 - tmp2


def abs(
    x: Array,
    softness: float = 1.0,
    mode: Literal[
        "hard", "entropic", "euclidean", "pseudohuber", "cubic", "quintic"
    ] = "entropic",
) -> Array:
    """Performs a soft version of [jax.numpy.abs](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.abs.html).

    **Arguments:**

    - `x`: Input Array of any shape.
    - `softness`: Softness of the function, should be larger than zero. Defaults to 1.
    - `mode`: Projection mode. "hard" returns the exact absolute value, otherwise uses
        "entropic", "pseudohuber", "euclidean", "cubic", or "quintic" relaxations.
        Defaults to "entropic".

    **Returns:**

    Result of applying soft elementwise absolute value to `x`.
    """
    if mode == "hard":
        return jnp.abs(x)
    else:
        return x * sign(x, softness=softness, mode=mode)


def sign(
    x: Array,
    softness: float = 1.0,
    mode: Literal[
        "hard", "entropic", "euclidean", "pseudohuber", "cubic", "quintic"
    ] = "entropic",
) -> Array:
    """Performs a soft version of [jax.numpy.sign](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.sign.html).

    **Arguments:**

    - `x`: Input Array of any shape.
    - `softness`: Softness of the function, should be larger than zero. Defaults to 1.
    - `mode`: If "hard", returns `jnp.sign`. Otherwise smooths via "entropic", "euclidean",
        "cubic", or "quintic" relaxations. Defaults to "entropic".

    **Returns:**

    Result of applying soft elementwise sign to `x`.
    """
    if mode == "hard":
        return jnp.sign(x).astype(jnp.float_)
    else:
        return _sigmoid(x, mode=mode, softness=softness) * 2.0 - 1.0


# Array -> SoftBool


def _sigmoid(
    x: Array,
    softness: float = 1.0,
    mode: Literal[
        "entropic", "euclidean", "pseudohuber", "cubic", "quintic"
    ] = "entropic",
) -> SoftBool:
    """Closed-form solution of `argmax(jnp.array([0, x]), softness=softness,
    mode=mode)`.

    **Arguments:**

    - `x`: Input Array.
    - `softness`: Softness of the function, should be larger than zero. Defaults to 1.
    - `mode`: Choice of smoothing family for the surrogate step. Defaults to "entropic".

    **Returns:**

    SoftBool of same shape as `x` (Array with values in [0, 1]), relaxing the
    elementwise Heaviside step function.
    """
    x = x / softness
    if mode == "entropic":  # smooth
        # closed form solution of argmax([0,x], mode="entropic")[1]
        y = jax.nn.sigmoid(x * ENTROPIC_CONSTANT)
    elif mode == "euclidean":  # continuous
        # closed form solution of argmax([0,x], mode="euclidean")[1]
        y = jnp.polyval(jnp.array([1.0, 0.5]), x)
        y = jnp.where(x < -0.5, 0.0, jnp.where(x < 0.5, y, 1.0))
    elif mode == "pseudohuber":
        x = x * PSEUDOHUBER_CONSTANT
        y = 0.5 * (1.0 + x / jnp.sqrt(1.0 + x * x))
    elif mode == "cubic":  # differentiable
        y = jnp.polyval(jnp.array([-2.0, 0.0, 1.5, 0.5]), x)
        y = jnp.where(x < -0.5, 0.0, jnp.where(x < 0.5, y, 1.0))
    elif mode == "quintic":  # twice differentiable
        y = jnp.polyval(jnp.array([6.0, 0.0, -5.0, 0.0, 1.875, 0.5]), x)
        y = jnp.where(x < -0.5, 0.0, jnp.where(x < 0.5, y, 1.0))
    else:
        raise ValueError(f"Invalid mode: {mode}")
    return y


def heaviside(
    x: Array,
    softness: float = 1.0,
    mode: Literal[
        "hard", "entropic", "euclidean", "pseudohuber", "cubic", "quintic"
    ] = "entropic",
) -> SoftBool:
    """Performs a soft version of [jax.numpy.heaviside](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.heaviside.html)(x,0.5).

    **Arguments:**

    - `x`: Input Array of any shape.
    - `softness`: Softness of the function, should be larger than zero. Defaults to 1.
    - `mode`: If "hard", returns the exact Heaviside step. Otherwise uses
        "entropic", "euclidean", "cubic", or "quintic" relaxations. Defaults to "entropic".

    **Returns:**

    SoftBool of same shape as `x` (Array with values in [0, 1]), relaxing the
    elementwise Heaviside step function.
    """
    if mode == "hard":
        return jnp.where(x < 0.0, 0.0, jnp.where(x > 0.0, 1.0, 0.5)).astype(jnp.float_)
    else:
        return _sigmoid(x, softness=softness, mode=mode)


def greater(
    x: Array,
    y: Array,
    softness: float = 1.0,
    mode: Literal[
        "hard", "entropic", "euclidean", "pseudohuber", "cubic", "quintic"
    ] = "entropic",
    epsilon: float = 1e-10,
) -> SoftBool:
    """Computes a soft approximation to elementwise `x > y`.
    Uses a Heaviside relaxation so the output approaches 0 at equality.

    **Arguments:**

    - `x`: First input Array.
    - `y`: Second input Array, same shape as `x`.
    - `softness`: Softness of the function, should be larger than zero. Defaults to 1.
    - `mode`: If "hard", returns the exact comparison. Otherwise uses a soft
        Heaviside: "entropic", "euclidean", "cubic" spline, or "quintic" spline.
        Defaults to "entropic".
    - `epsilon`: Small offset so that as softness->0, greater returns 0 at equality.

    **Returns:**

    SoftBool of same shape as `x` and `y` (Array with values in [0, 1]), relaxing the
    elementwise `x > y`.
    """
    if mode == "hard":
        return jnp.greater(x, y).astype(jnp.float_)
    else:
        return _sigmoid(x - y - epsilon, softness=softness, mode=mode)


def greater_equal(
    x: Array,
    y: Array,
    softness: float = 1.0,
    mode: Literal[
        "hard", "entropic", "euclidean", "pseudohuber", "cubic", "quintic"
    ] = "entropic",
    epsilon: float = 1e-10,
) -> SoftBool:
    """Computes a soft approximation to elementwise `x >= y`.
    Uses a Heaviside relaxation so the output approaches 1 at equality.

    **Arguments:**

    - `x`: First input Array.
    - `y`: Second input Array, same shape as `x`.
    - `softness`: Softness of the function, should be larger than zero. Defaults to 1.
    - `mode`: If "hard", returns the exact comparison. Otherwise uses a soft
        Heaviside: "entropic", "euclidean", "cubic" spline, or "quintic" spline.
        Defaults to "entropic".
    - `epsilon`: Small offset so that as softness->0, greater_equal returns 1 at
        equality.

    **Returns:**

    SoftBool of same shape as `x` and `y` (Array with values in [0, 1]), relaxing the
    elementwise `x >= y`.
    """
    if mode == "hard":
        return jnp.greater_equal(x, y).astype(jnp.float_)
    else:
        return _sigmoid(x - y + epsilon, softness=softness, mode=mode)


def less(
    x: Array,
    y: Array,
    softness: float = 1.0,
    mode: Literal[
        "hard", "entropic", "euclidean", "pseudohuber", "cubic", "quintic"
    ] = "entropic",
    epsilon: float = 1e-10,
) -> SoftBool:
    """Computes a soft approximation to elementwise `x < y`.
    Uses a Heaviside relaxation so the output approaches 0 at equality.

    **Arguments:**

    - `x`: First input Array.
    - `y`: Second input Array, same shape as `x`.
    - `softness`: Softness of the function, should be larger than zero. Defaults to 1.
    - `mode`: If "hard", returns the exact comparison. Otherwise uses a soft
        Heaviside: "entropic", "euclidean", "cubic" spline, or "quintic" spline.
        Defaults to "entropic".
    - `epsilon`: Small offset so that as softness->0, less returns 0 at equality.

    **Returns:**

    SoftBool of same shape as `x` and `y` (Array with values in [0, 1]), relaxing the
    elementwise `x < y`.
    """
    if mode == "hard":
        return jnp.less(x, y).astype(jnp.float_)
    else:
        return logical_not(
            greater_equal(x, y, softness=softness, mode=mode, epsilon=epsilon)
        )


def less_equal(
    x: Array,
    y: Array,
    softness: float = 1.0,
    mode: Literal[
        "hard", "entropic", "euclidean", "pseudohuber", "cubic", "quintic"
    ] = "entropic",
    epsilon: float = 1e-10,
) -> SoftBool:
    """Computes a soft approximation to elementwise `x <= y`.
    Uses a Heaviside relaxation so the output approaches 1 at equality.

    **Arguments:**

    - `x`: First input Array.
    - `y`: Second input Array, same shape as `x`.
    - `softness`: Softness of the function, should be larger than zero. Defaults to 1.
    - `mode`: If "hard", returns the exact comparison. Otherwise uses a soft
        Heaviside: "entropic", "euclidean", "cubic" spline, or "quintic" spline.
        Defaults to "entropic".
    - `epsilon`: Small offset so that as softness->0, less_equal returns 1 at equality.

    **Returns:**

    SoftBool of same shape as `x` and `y` (Array with values in [0, 1]), relaxing the
    elementwise `x <= y`.
    """
    if mode == "hard":
        return jnp.less_equal(x, y).astype(jnp.float_)
    else:
        return logical_not(greater(x, y, softness=softness, mode=mode, epsilon=epsilon))


def equal(
    x: Array,
    y: Array,
    softness: float = 1.0,
    mode: Literal[
        "hard", "entropic", "euclidean", "pseudohuber", "cubic", "quintic"
    ] = "entropic",
    epsilon: float = 1e-10,
) -> SoftBool:
    """Computes a soft approximation to elementwise `x == y`.
    Implemented as a soft `abs(x - y) <= 0` comparison.

    **Arguments:**

    - `x`: First input Array.
    - `y`: Second input Array, same shape as `x`.
    - `softness`: Softness of the function, should be larger than zero. Defaults to 1.
    - `mode`: If "hard", returns the exact comparison. Otherwise uses a soft
        Heaviside: "entropic", "euclidean", "cubic" spline, or "quintic" spline.
        Defaults to "entropic".
    - `epsilon`: Small offset so that as softness->0, equal returns 1 at equality.

    **Returns:**

    SoftBool of same shape as `x` and `y` (Array with values in [0, 1]), relaxing the
    elementwise `x == y`.
    """
    if mode == "hard":
        return jnp.equal(x, y).astype(jnp.float_)
    else:
        diff = jnp.abs(x - y)
        return less_equal(
            diff, jnp.zeros_like(diff), mode=mode, softness=softness, epsilon=epsilon
        )


def not_equal(
    x: Array,
    y: Array,
    softness: float = 1.0,
    mode: Literal[
        "hard", "entropic", "euclidean", "pseudohuber", "cubic", "quintic"
    ] = "entropic",
    epsilon: float = 1e-10,
) -> SoftBool:
    """Computes a soft approximation to elementwise `x != y`.
    Implemented as a soft `abs(x - y) > 0` comparison.

    **Arguments:**

    - `x`: First input Array.
    - `y`: Second input Array, same shape as `x`.
    - `softness`: Softness of the function, should be larger than zero. Defaults to 1.
    - `mode`: If "hard", returns the exact comparison. Otherwise uses a soft
        Heaviside: "entropic", "euclidean", "cubic" spline, or "quintic" spline.
        Defaults to "entropic".
    - `epsilon`: Small offset so that as softness->0, not_equal returns 0 at equality.

    **Returns:**

    SoftBool of same shape as `x` and `y` (Array with values in [0, 1]), relaxing the
    elementwise `x != y`.
    """
    if mode == "hard":
        return jnp.not_equal(x, y).astype(jnp.float_)
    else:
        diff = jnp.abs(x - y)
        return greater(
            diff, jnp.zeros_like(diff), mode=mode, softness=softness, epsilon=epsilon
        )


def isclose(
    x: Array,
    y: Array,
    softness: float = 1.0,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    mode: Literal[
        "hard", "entropic", "euclidean", "pseudohuber", "cubic", "quintic"
    ] = "entropic",
    epsilon: float = 1e-10,
) -> SoftBool:
    """Computes a soft approximation to `jnp.isclose` for elementwise comparison.
    Implemented as a soft `abs(x - y) <= atol + rtol * abs(y)` comparison.

    **Arguments:**

    - `x`: First input Array.
    - `y`: Second input Array, same shape as `x`.
    - `softness`: Softness of the function, should be larger than zero. Defaults to 1.
    - `rtol`: Relative tolerance. Defaults to 1e-5.
    - `atol`: Absolute tolerance. Defaults to 1e-8.
    - `mode`: If "hard", returns the exact comparison. Otherwise uses a soft
        Heaviside: "entropic", "euclidean", "cubic" spline, or "quintic" spline.
        Defaults to "entropic".
    - `epsilon`: Small offset so that as softness->0, isclose returns 1 at equality.

    **Returns:**

    SoftBool of same shape as `x` and `y` (Array with values in [0, 1]), relaxing the
    elementwise `isclose(x, y)`.
    """
    if mode == "hard":
        return jnp.isclose(x, y, atol=atol, rtol=rtol).astype(jnp.float_)
    else:
        return less_equal(
            jnp.abs(x - y),
            atol + rtol * jnp.abs(y),
            mode=mode,
            softness=softness,
            epsilon=epsilon,
        )


def argwhere() -> SoftBool:
    """Placeholder for a differentiable `jnp.argwhere` analogue."""
    raise NotImplementedError


# SoftBool -> SoftBool


def logical_not(x: SoftBool) -> SoftBool:
    """Computes soft elementwise logical NOT of a SoftBool Array.
    Fuzzy logic implemented as `1.0 - x`.

    **Arguments:**
    - `x`: SoftBool input Array.

    **Returns:**

    SoftBool of same shape as `x` (Array with values in [0, 1]), relaxing the
    elementwise logical NOT.
    """
    return 1.0 - x


def all(x: SoftBool, axis: int = -1, epsilon: float = 1e-10) -> SoftBool:
    """Computes soft elementwise logical AND across a specified axis.
    Fuzzy logic implemented as the geometric mean along the axis.

    **Arguments:**
    - `x`: SoftBool input Array.
    - `axis`: Axis along which to compute the logical AND. Default is -1 (last axis).
    - `epsilon`: Minimum value for numerical stability inside the log.

    **Returns:**

    SoftBool (Array with values in [0, 1]) with the specified axis reduced, relaxing
    the logical ALL along that axis.
    """
    return jnp.exp(jnp.mean(jnp.log(jnp.clip(x, a_min=epsilon)), axis=axis))


def any(x: SoftBool, axis: int = -1) -> SoftBool:
    """Computes soft elementwise logical OR across a specified axis.
    Fuzzy logic implemented as `1.0 - all(logical_not(x), axis=axis)`.

    **Arguments:**
    - `x`: SoftBool input Array.
    - `axis`: Axis along which to compute the logical OR. Default is -1 (last axis).

    **Returns:**

    SoftBool (Array with values in [0, 1]) with the specified axis reduced, relaxing t
    he logical ANY along that axis.
    """
    return logical_not(all(logical_not(x), axis=axis))


def logical_and(x: SoftBool, y: SoftBool) -> SoftBool:
    """Computes soft elementwise logical AND between two SoftBool Arrays.
    Fuzzy logic implemented as `all(stack([x, y], axis=-1), axis=-1)`.

    **Arguments:**

    - `x`: First SoftBool input Array.
    - `y`: Second SoftBool input Array.

    **Returns:**

    SoftBool of same shape as `x` and `y` (Array with values in [0, 1]), relaxing the
    elementwise logical AND.
    """
    return all(jnp.stack([x, y], axis=-1), axis=-1)


def logical_or(x: SoftBool, y: SoftBool) -> SoftBool:
    """Computes soft elementwise logical OR between two SoftBool Arrays.
    Fuzzy logic implemented as `any(stack([x, y], axis=-1), axis=-1)`.

    **Arguments:**
    - `x`: First SoftBool input Array.
    - `y`: Second SoftBool input Array.

    **Returns:**

    SoftBool of same shape as `x` and `y` (Array with values in [0, 1]), relaxing the
    elementwise logical OR.
    """
    return any(jnp.stack([x, y], axis=-1), axis=-1)


def logical_xor(x: SoftBool, y: SoftBool) -> SoftBool:
    """Computes soft elementwise logical XOR between two SoftBool Arrays.

    **Arguments:**
    - `x`: First SoftBool input Array.
    - `y`: Second SoftBool input Array.

    **Returns:**

    SoftBool of same shape as `x` and `y` (Array with values in [0, 1]), relaxing the
    elementwise logical XOR.
    """
    return logical_or(logical_and(x, logical_not(y)), logical_and(logical_not(x), y))


def where(condition: SoftBool, x: Array, y: Array) -> Array:
    """Computes a soft elementwise selection between two Arrays based on a SoftBool
    condition. Fuzzy logic implemented as `x * condition + y * (1.0 - condition)`.

    **Arguments:**
    - `condition`: SoftBool condition Array, same shape as `x` and `y`.
    - `x`: First input Array, same shape as `condition`.
    - `y`: Second input Array, same shape as `condition`.

    **Returns:**

    Array of the same shape as `x` and `y`, interpolating between `x` and `y` according
    to `condition` in [0, 1].
    """
    return x * condition + y * (1.0 - condition)
