# topk_ot.py  (jax-ott ≥ 0.5)


import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp


# from ott.geometry import geometry
# from ott.problems.linear import linear_problem
# from ott.solvers.linear import sinkhorn


@jax.custom_jvp
def _projection_unit_simplex(values: jax.Array) -> jax.Array:
    """Projection onto the unit simplex. Taken from optax.

    Taken from https://github.com/google/jaxopt/blob/cf28b4563f5ad9354b76433622dbb9ee32af5f09/jaxopt/_src/projection.py#L96
    """
    s = 1.0
    n_features = values.shape[0]
    u = jnp.sort(values)[::-1]
    cumsum_u = jnp.cumsum(u)
    ind = jnp.arange(n_features) + 1
    cond = s / ind + (u - cumsum_u / ind) > 0
    idx = jnp.count_nonzero(cond)
    return jax.nn.relu(s / idx + (values - cumsum_u[idx - 1] / idx))


@_projection_unit_simplex.defjvp
def _projection_unit_simplex_jvp(
    primals: list[jax.Array], tangents: list[jax.Array]
) -> tuple[jax.Array, jax.Array]:
    (values,) = primals
    (values_dot,) = tangents
    primal_out = _projection_unit_simplex(values)
    supp = primal_out > 0
    card = jnp.count_nonzero(supp)
    tangent_out = supp * values_dot - (jnp.dot(supp, values_dot) / card) * supp
    return primal_out, tangent_out


def _canonicalize_axis(axis: int | None, num_dims: int) -> int:
    if axis is None:
        raise ValueError("axis must be specified")
    if not -num_dims <= axis < num_dims:
        raise ValueError(
            f"axis {axis} is out of bounds for array of dimension {num_dims}"
        )
    if axis < 0:
        axis += num_dims
    return axis


# @partial(jax.jit, static_argnames=("max_iter", "epsilon", "implicit_diff"), inline=True)
# def _sinkhorn_ott(
#     C: jax.Array,
#     mu: jax.Array,
#     nu: jax.Array,
#     max_iter: int = 1000,
#     epsilon: float = 1.0,
#     implicit_diff: bool = False,
# ) -> jax.Array:
#     geom = geometry.Geometry(cost_matrix=C, epsilon=epsilon)
#     ot_prob = linear_problem.LinearProblem(geom, a=mu, b=nu)
#     solver = sinkhorn.Sinkhorn(max_iterations=max_iter, implicit_diff=implicit_diff)
#     out = solver(ot_prob)
#     return out.matrix


# @partial(jax.jit, static_argnames=("max_iter"), inline=True)
def _sinkhorn(
    C: jax.Array,
    mu: jax.Array,
    nu: jax.Array,
    max_iter: int = 1,
) -> jax.Array:
    """
    Compute the Sinkhorn transport plan Gamma in log-space for numerical stability.

    Args:
        C: Cost matrix of shape (n, m)
        mu: Source distribution of shape (n,)
        nu: Target distribution of shape (m,)
        epsilon: Entropic regularization strength
        max_iter: Number of Sinkhorn iterations

    Returns:
        Gamma: Optimal transport matrix of shape (n, m)
    """

    n, m = C.shape
    log_mu = jnp.log(mu + 1e-20)
    log_nu = jnp.log(nu + 1e-20)

    log_K = -C  # shape (n, m)
    log_u = jnp.zeros(n)
    log_v = jnp.zeros(m)

    def update_u(log_v):
        return log_mu - logsumexp(log_K + log_v[None, :], axis=1)

    def update_v(log_u):
        return log_nu - logsumexp(log_K.T + log_u[None, :], axis=1)

    def step_fn(_, state):
        log_u, log_v = state
        log_u = update_u(log_v)
        log_v = update_v(log_u)
        return (log_u, log_v)

    log_u, log_v = jax.lax.fori_loop(0, max_iter, step_fn, (log_u, log_v))
    log_P = log_K + log_u[:, None] + log_v[None, :]
    return jnp.exp(log_P)
