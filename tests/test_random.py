import jax
import jax.numpy as jnp
import pytest

import softjax as sj

from . import common


CATEGORICAL_CASES = [
    (jnp.array([0.1, 1.5, -0.2]), -1, None),
    (jnp.array([[0.1, 1.5, -0.2], [1.2, -0.4, 0.3]]), -1, None),
    (jnp.array([[0.1, 1.5], [-0.2, 0.7], [1.2, -0.4]]), 0, None),
    (jnp.array([[0.1, 1.5, -0.2], [1.2, -0.4, 0.3]]), -1, (4, 2)),
]


CATEGORICAL_SOFT_SHAPE_CASES = [
    (jnp.array([0.1, 1.5, -0.2]), -1, 2),
    (jnp.array([[0.1, 1.5, -0.2], [1.2, -0.4, 0.3]]), -1, None),
    (jnp.array([[0.1, 1.5, -0.2], [1.2, -0.4, 0.3]]), -1, (4, 2)),
    (jnp.array([[0.1, 1.5], [-0.2, 0.7], [1.2, -0.4]]), 0, (4, 2)),
]


BERNOULLI_SOFT_SHAPE_CASES = [
    (jnp.array([0.2, 0.7, 0.4]), None),
    (jnp.array([0.2, 0.7, 0.4]), 3),
    (jnp.array([0.2, 0.7, 0.4]), (2, 3)),
    (jnp.array(0.3), (2, 3)),
]


def _categorical_hard_expected(key, logits, axis=-1, shape=None, replace=True, rng_mode=None):
    indices = jax.random.categorical(
        key, logits, axis=axis, shape=shape, replace=replace, mode=rng_mode
    )
    return jax.nn.one_hot(indices, num_classes=logits.shape[axis], axis=-1)


@pytest.mark.parametrize(
    "logits, axis, shape",
    CATEGORICAL_CASES,
)
@pytest.mark.parametrize("rng_mode", [None, "low", "high"])
def test_categorical_hard_matches_jax(logits, axis, shape, rng_mode):
    key = jax.random.key(0)
    actual = sj.random.categorical(
        key, logits, axis=axis, shape=shape, mode="hard", rng_mode=rng_mode
    )
    expected = _categorical_hard_expected(
        key, logits, axis=axis, shape=shape, rng_mode=rng_mode
    )
    common.assert_jax_parity(actual, expected, msg="categorical hard")


@pytest.mark.parametrize("logits, axis, shape", CATEGORICAL_CASES)
@pytest.mark.parametrize("rng_mode", [None, "low", "high"])
def test_categorical_private_hard_matches_hard(logits, axis, shape, rng_mode):
    key = jax.random.key(1)

    actual = sj.random.categorical(
        key, logits, axis=axis, shape=shape, mode="_hard", rng_mode=rng_mode
    )
    expected = sj.random.categorical(
        key, logits, axis=axis, shape=shape, mode="hard", rng_mode=rng_mode
    )
    common.assert_allclose(actual, expected, tol=0.0)


@pytest.mark.parametrize("logits, axis, shape", CATEGORICAL_SOFT_SHAPE_CASES)
def test_categorical_soft_shape_contract(logits, axis, shape):
    key = jax.random.key(27)

    actual = sj.random.categorical(
        key, logits, axis=axis, shape=shape, softness=1.0, mode="smooth"
    )
    expected_indices = jax.random.categorical(key, logits, axis=axis, shape=shape)

    assert actual.shape == expected_indices.shape + (logits.shape[axis],)
    common.assert_simplex(actual)


@pytest.mark.parametrize("mode", ["hard", "_hard", "smooth"])
@pytest.mark.parametrize("axis", [2, -3])
def test_categorical_invalid_axis_errors_for_all_modes(mode, axis):
    key = jax.random.key(29)
    logits = jnp.zeros((2, 3))

    with pytest.raises((IndexError, ValueError), match="out of bounds"):
        sj.random.categorical(key, logits, axis=axis, mode=mode)


@pytest.mark.parametrize("mode", ["smooth", "c0", "c1", "c2"])
def test_categorical_soft_modes_are_simplex_and_differentiable(mode):
    key = jax.random.key(2)
    logits = jnp.array([0.1, 1.5, -0.2])
    weights = jnp.array([0.3, -0.7, 1.1])

    out = sj.random.categorical(
        key, logits, softness=1.0, mode=mode, method="softsort"
    )
    common.assert_simplex(out)
    common.assert_finite(out, msg=f"categorical {mode}")

    def loss(z):
        sample = sj.random.categorical(
            key, z, softness=1.0, mode=mode, method="softsort"
        )
        return jnp.sum(sample * weights)

    grad = jax.grad(loss)(logits)
    common.assert_finite(grad, msg=f"categorical grad {mode}")


@pytest.mark.parametrize("mode", ["smooth", "c0", "c1", "c2"])
def test_categorical_near_hard_matches_hard(mode):
    key = jax.random.key(14)
    logits = jnp.array([0.1, 1.5, -0.2])

    actual = sj.random.categorical(
        key,
        logits,
        softness=common.NEAR_HARD_SOFTNESS,
        mode=mode,
        method="softsort",
    )
    expected = sj.random.categorical(key, logits, mode="hard")
    common.assert_allclose(actual, expected, tol=common.TOLERANCE)


@pytest.mark.parametrize("method", ["softsort", "neuralsort", "sorting_network"])
@pytest.mark.parametrize("mode", ["smooth", "c0", "c1", "c2"])
def test_categorical_methods_are_simplex(method, mode):
    key = jax.random.key(3)
    logits = jnp.array([0.1, 1.5, -0.2])

    out = sj.random.categorical(
        key, logits, softness=1.0, mode=mode, method=method
    )
    common.assert_simplex(out)


def test_categorical_return_log_probs_matches_probs():
    key = jax.random.key(4)
    logits = jnp.array([0.1, 1.5, -0.2])

    probs = sj.random.categorical(key, logits, softness=1.0, mode="smooth")
    log_probs = sj.random.categorical(
        key, logits, softness=1.0, mode="smooth", return_log_probs=True
    )

    common.assert_allclose(jnp.exp(log_probs), probs)
    common.assert_allclose(jax.nn.logsumexp(log_probs, axis=-1), 0.0)


def test_categorical_return_log_probs_log_prob_eps_matches_manual():
    key = jax.random.key(16)
    logits = jnp.array([0.1, 1.5, -0.2])
    eps = 1e-3

    probs = sj.random.categorical(key, logits, softness=1.0, mode="c0")
    log_probs = sj.random.categorical(
        key,
        logits,
        softness=1.0,
        mode="c0",
        return_log_probs=True,
        log_prob_eps=eps,
    )
    clipped = jnp.maximum(probs, eps)
    expected = jnp.log(clipped / jnp.sum(clipped, axis=-1, keepdims=True))

    common.assert_allclose(log_probs, expected)
    common.assert_allclose(jax.nn.logsumexp(log_probs, axis=-1), 0.0)


def test_categorical_return_log_probs_validation():
    key = jax.random.key(17)
    logits = jnp.array([0.1, 1.5, -0.2])

    with pytest.raises(ValueError, match="return_log_probs=True"):
        sj.random.categorical(key, logits, log_prob_eps=1e-3)

    with pytest.raises(ValueError, match="log_prob_eps must be in"):
        sj.random.categorical(
            key, logits, return_log_probs=True, log_prob_eps=0.0
        )


def test_categorical_hard_log_probs_have_negative_infinity():
    key = jax.random.key(18)
    logits = jnp.array([0.1, 1.5, -0.2])

    probs = sj.random.categorical(key, logits, mode="hard")
    log_probs = sj.random.categorical(key, logits, mode="hard", return_log_probs=True)

    common.assert_allclose(jnp.exp(log_probs), probs)
    assert jnp.any(jnp.isneginf(log_probs))


def test_categorical_return_log_probs_smooth_has_finite_gradients():
    key = jax.random.key(24)
    logits = jnp.array([0.1, 1.5, -0.2], dtype=jnp.float64)
    weights = jnp.array([0.3, -0.7, 1.1], dtype=jnp.float64)

    def loss(z):
        log_probs = sj.random.categorical(
            key,
            z,
            softness=1.0,
            mode="smooth",
            return_log_probs=True,
        )
        return jnp.sum(log_probs * weights)

    grad = jax.grad(loss)(logits)
    common.assert_finite(grad, msg="categorical smooth log_probs gradient")


def test_categorical_log_prob_eps_has_finite_sparse_gradients():
    key = jax.random.key(25)
    logits = jnp.array([0.1, 1.5, -0.2], dtype=jnp.float64)
    weights = jnp.array([0.3, -0.7, 1.1], dtype=jnp.float64)

    def loss(z):
        log_probs = sj.random.categorical(
            key,
            z,
            softness=0.01,
            mode="c0",
            return_log_probs=True,
            log_prob_eps=1e-12,
        )
        return jnp.sum(log_probs * weights)

    grad = jax.grad(loss)(logits)
    common.assert_finite(grad, msg="categorical c0 safe log_probs gradient")


def test_categorical_replace_false_hard_matches_jax_and_soft_raises():
    key = jax.random.key(5)
    logits = jnp.array([0.1, 1.5, -0.2])

    actual = sj.random.categorical(
        key, logits, shape=(2,), replace=False, mode="hard"
    )
    expected = _categorical_hard_expected(
        key, logits, shape=(2,), replace=False
    )
    common.assert_jax_parity(actual, expected, msg="categorical replace=False hard")

    with pytest.raises(NotImplementedError, match="replace=False"):
        sj.random.categorical(key, logits, shape=(2,), replace=False, mode="smooth")


def test_categorical_st_forward_and_grad():
    key = jax.random.key(6)
    logits = jnp.array([0.1, 1.5, -0.2])
    weights = jnp.array([0.3, -0.7, 1.1])

    actual = sj.random.categorical_st(key, logits, softness=1.0, mode="smooth")
    expected = sj.random.categorical(key, logits, mode="hard")
    common.assert_allclose(actual, expected, tol=0.0)

    def st_loss(z):
        return jnp.sum(
            sj.random.categorical_st(key, z, softness=1.0, mode="smooth") * weights
        )

    def soft_loss(z):
        return jnp.sum(
            sj.random.categorical(key, z, softness=1.0, mode="smooth") * weights
        )

    common.assert_allclose(jax.grad(st_loss)(logits), jax.grad(soft_loss)(logits))


@pytest.mark.parametrize("mode", ["smooth", "c0", "c1", "c2"])
def test_categorical_st_return_log_probs_handles_negative_infinity(mode):
    key = jax.random.key(26)
    logits = jnp.array([0.1, 1.5, -0.2])

    log_probs = sj.random.categorical_st(
        key, logits, softness=1.0, mode=mode, return_log_probs=True
    )
    hard_probs = sj.random.categorical(key, logits, mode="hard")

    assert not jnp.any(jnp.isnan(log_probs))
    common.assert_allclose(jnp.exp(log_probs), hard_probs)


def test_categorical_jit_and_vmap():
    key = jax.random.key(12)
    logits = jnp.array([0.1, 1.5, -0.2])

    jitted = jax.jit(
        lambda z: sj.random.categorical(key, z, softness=1.0, mode="smooth")
    )(logits)
    common.assert_simplex(jitted)

    keys = jax.random.split(key, 2)
    batch_logits = jnp.array([[0.1, 1.5, -0.2], [1.2, -0.4, 0.3]])
    vmapped = jax.vmap(
        lambda k, z: sj.random.categorical(k, z, softness=1.0, mode="smooth")
    )(keys, batch_logits)
    assert vmapped.shape == batch_logits.shape
    common.assert_simplex(vmapped)


def test_categorical_grad_matches_finite_diff():
    key = jax.random.key(19)
    logits = jnp.array([0.1, 1.5, -0.2], dtype=jnp.float64)
    weights = jnp.array([0.3, -0.7, 1.1], dtype=jnp.float64)

    def loss(z):
        sample = sj.random.categorical(
            key, z, softness=1.0, mode="smooth", method="softsort"
        )
        return jnp.sum(sample * weights)

    common.assert_grad_matches_finite_diff(loss, logits, msg="categorical")


def test_categorical_take_along_axis_pipeline_grad_matches_finite_diff():
    key = jax.random.key(20)
    x = jnp.array([[1.0, 3.0, 2.0], [5.0, 1.0, 4.0]], dtype=jnp.float64)

    def pipeline(values):
        soft_idx = sj.random.categorical(
            key, values, axis=-1, softness=1.0, mode="smooth"
        )
        soft_idx = jnp.expand_dims(soft_idx, axis=-2)
        selected = sj.take_along_axis(values, soft_idx, axis=-1)
        return jnp.sum(selected)

    grad = jax.grad(pipeline)(x)
    common.assert_finite(grad, msg="categorical take_along_axis pipeline")
    common.assert_grad_matches_finite_diff(pipeline, x, msg="categorical pipeline")


@pytest.mark.parametrize("shape", [None, (3,), (2, 3)])
@pytest.mark.parametrize("rng_mode", ["low", "high"])
def test_bernoulli_hard_matches_jax(shape, rng_mode):
    key = jax.random.key(7)
    p = jnp.array([0.2, 0.7, 0.4])

    actual = sj.random.bernoulli(key, p, shape=shape, mode="hard", rng_mode=rng_mode)
    expected = jax.random.bernoulli(
        key, p=p, shape=shape, mode=rng_mode
    ).astype(p.dtype)
    common.assert_jax_parity(actual, expected, msg="bernoulli hard")


@pytest.mark.parametrize("rng_mode", ["low", "high"])
def test_bernoulli_private_hard_matches_hard(rng_mode):
    key = jax.random.key(8)
    p = jnp.array([0.2, 0.7, 0.4])

    actual = sj.random.bernoulli(key, p, mode="_hard", rng_mode=rng_mode)
    expected = sj.random.bernoulli(key, p, mode="hard", rng_mode=rng_mode)
    common.assert_allclose(actual, expected, tol=0.0)


@pytest.mark.parametrize("p, shape", BERNOULLI_SOFT_SHAPE_CASES)
def test_bernoulli_soft_shape_contract(p, shape):
    key = jax.random.key(28)

    actual = sj.random.bernoulli(key, p, shape=shape, softness=1.0, mode="smooth")
    expected = jax.random.bernoulli(key, p=p, shape=shape)

    assert actual.shape == expected.shape
    common.assert_softbool(actual)


@pytest.mark.parametrize("mode", ["smooth", "c0", "c1", "c2"])
def test_bernoulli_soft_modes_are_softbool_and_differentiable(mode):
    key = jax.random.key(9)
    p = jnp.array([0.35, 0.5, 0.65])
    weights = jnp.array([0.3, -0.7, 1.1])

    out = sj.random.bernoulli(key, p, softness=1.0, mode=mode)
    common.assert_softbool(out)
    common.assert_finite(out, msg=f"bernoulli {mode}")

    def loss(prob):
        return jnp.sum(sj.random.bernoulli(key, prob, softness=1.0, mode=mode) * weights)

    grad = jax.grad(loss)(p)
    common.assert_finite(grad, msg=f"bernoulli grad {mode}")


@pytest.mark.parametrize("mode", ["smooth", "c0", "c1", "c2"])
def test_bernoulli_near_hard_matches_hard(mode):
    key = jax.random.key(10)
    p = jnp.array([0.2, 0.7, 0.4, 0.9])

    actual = sj.random.bernoulli(
        key, p, softness=common.NEAR_HARD_SOFTNESS, mode=mode
    )
    expected = sj.random.bernoulli(key, p, mode="hard")
    common.assert_allclose(actual, expected, tol=common.TOLERANCE)


def test_bernoulli_st_forward_and_grad():
    key = jax.random.key(10)
    p = jnp.array([0.35, 0.5, 0.65])
    weights = jnp.array([0.3, -0.7, 1.1])

    actual = sj.random.bernoulli_st(key, p, softness=1.0, mode="smooth")
    expected = sj.random.bernoulli(key, p, mode="hard")
    common.assert_allclose(actual, expected, tol=0.0)

    def st_loss(prob):
        return jnp.sum(
            sj.random.bernoulli_st(key, prob, softness=1.0, mode="smooth") * weights
        )

    def soft_loss(prob):
        return jnp.sum(
            sj.random.bernoulli(key, prob, softness=1.0, mode="smooth") * weights
        )

    common.assert_allclose(jax.grad(st_loss)(p), jax.grad(soft_loss)(p))


def test_bernoulli_jit_and_vmap():
    key = jax.random.key(13)
    p = jnp.array([0.35, 0.5, 0.65])

    jitted = jax.jit(lambda prob: sj.random.bernoulli(key, prob, softness=1.0))(p)
    common.assert_softbool(jitted)

    keys = jax.random.split(key, 2)
    batch_p = jnp.array([[0.35, 0.5, 0.65], [0.2, 0.7, 0.4]])
    vmapped = jax.vmap(lambda k, prob: sj.random.bernoulli(k, prob, softness=1.0))(
        keys, batch_p
    )
    assert vmapped.shape == batch_p.shape
    common.assert_softbool(vmapped)


def test_bernoulli_grad_matches_finite_diff():
    key = jax.random.key(21)
    p = jnp.array([0.35, 0.5, 0.65], dtype=jnp.float64)
    weights = jnp.array([0.3, -0.7, 1.1], dtype=jnp.float64)

    def loss(prob):
        return jnp.sum(
            sj.random.bernoulli(key, prob, softness=1.0, mode="smooth") * weights
        )

    common.assert_grad_matches_finite_diff(loss, p, msg="bernoulli")


def test_bernoulli_where_pipeline_grad_matches_finite_diff():
    key = jax.random.key(22)
    p = jnp.array([0.35, 0.5, 0.65], dtype=jnp.float64)
    x = jnp.array([1.0, 3.0, 2.0], dtype=jnp.float64)
    y = jnp.array([-1.0, -2.0, 0.5], dtype=jnp.float64)

    def pipeline(prob):
        condition = sj.random.bernoulli(key, prob, softness=1.0, mode="smooth")
        return jnp.sum(sj.where(condition, x, y))

    grad = jax.grad(pipeline)(p)
    common.assert_finite(grad, msg="bernoulli where pipeline")
    common.assert_grad_matches_finite_diff(pipeline, p, msg="bernoulli pipeline")


def test_random_shape_and_rng_mode_errors():
    key = jax.random.key(23)

    with pytest.raises(ValueError, match="broadcast-compatible"):
        sj.random.categorical(
            key, jnp.array([[0.1, 1.5, -0.2], [1.2, -0.4, 0.3]]), shape=(3,)
        )

    with pytest.raises(ValueError, match="broadcast-compatible"):
        sj.random.bernoulli(key, jnp.array([0.2, 0.7, 0.4]), shape=(2,))

    with pytest.raises(ValueError, match="expected 'high' or 'low'"):
        sj.random.bernoulli(key, jnp.array([0.5]), rng_mode="medium")

    with pytest.raises(TypeError, match="integer type"):
        sj.random.bernoulli(key, jnp.array([0.5]), shape=(1.2,))


def test_random_invalid_noise_errors():
    key = jax.random.key(11)
    with pytest.raises(ValueError, match="noise='gumbel'"):
        sj.random.categorical(key, jnp.array([0.1, 0.2]), noise="uniform")
    with pytest.raises(ValueError, match="noise='uniform'"):
        sj.random.bernoulli(key, jnp.array([0.5]), noise="gumbel")
