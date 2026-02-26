"""Quick Example — prints all outputs for README and documentation."""

import jax
import jax.numpy as jnp

import softjax as sj


jnp.set_printoptions(precision=4, suppress=True)
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "high")

# ── Elementwise operators ─────────────────────────────────────────────────────

x = jnp.array([-0.2, -1.0, 0.3, 1.0])

print("\nJAX absolute:", jnp.abs(x))
print("SoftJAX absolute (hard mode):", sj.abs(x, mode="hard"))
print("SoftJAX absolute (soft mode):", sj.abs(x))

print("\nJAX clip:", jnp.clip(x, -0.5, 0.5))
print("SoftJAX clip (hard mode):", sj.clip(x, -0.5, 0.5, mode="hard"))
print("SoftJAX clip (soft mode):", sj.clip(x, -0.5, 0.5))

print("\nJAX heaviside:", jnp.heaviside(x, 0.5))
print("SoftJAX heaviside (hard mode):", sj.heaviside(x, mode="hard"))
print("SoftJAX heaviside (soft mode):", sj.heaviside(x))

print("\nJAX ReLU:", jax.nn.relu(x))
print("SoftJAX ReLU (hard mode):", sj.relu(x, mode="hard"))
print("SoftJAX ReLU (soft mode):", sj.relu(x))

print("\nJAX round:", jnp.round(x))
print("SoftJAX round (hard mode):", sj.round(x, mode="hard"))
print("SoftJAX round (soft mode):", sj.round(x))

print("\nJAX sign:", jnp.sign(x))
print("SoftJAX sign (hard mode):", sj.sign(x, mode="hard"))
print("SoftJAX sign (soft mode):", sj.sign(x))

# ── Array-valued operators ────────────────────────────────────────────────────

print("\nJAX max:", jnp.max(x))
print("SoftJAX max (hard mode):", sj.max(x, mode="hard"))
print(f"SoftJAX max (soft mode): {sj.max(x):.4f}")

print("\nJAX min:", jnp.min(x))
print("SoftJAX min (hard mode):", sj.min(x, mode="hard"))
print(f"SoftJAX min (soft mode): {sj.min(x):.4f}")

print("\nJAX sort:", jnp.sort(x))
print("SoftJAX sort (hard mode):", sj.sort(x, mode="hard"))
print("SoftJAX sort (soft mode):", sj.sort(x))

print(f"\nJAX quantile: {jnp.quantile(x, q=0.2):.4f}")
print(f"SoftJAX quantile (hard mode): {sj.quantile(x, q=0.2, mode='hard'):.4f}")
print(f"SoftJAX quantile (soft mode): {sj.quantile(x, q=0.2):.4f}")

print(f"\nJAX median: {jnp.median(x):.4f}")
print(f"SoftJAX median (hard mode): {sj.median(x, mode='hard'):.4f}")
print(f"SoftJAX median (soft mode): {sj.median(x):.4f}")


print("\nJAX top_k:", jax.lax.top_k(x, k=3)[0])
print("SoftJAX top_k (hard mode):", sj.top_k(x, k=3, mode="hard")[0])
print("SoftJAX top_k (soft mode):", sj.top_k(x, k=3)[0])

print("\nJAX rank:", jnp.argsort(jnp.argsort(x)))
print("SoftJAX rank (hard mode):", sj.rank(x, mode="hard", descending=False))
print("SoftJAX rank (soft mode):", sj.rank(x, descending=False))

# ── Sort: sweep over methods ──────────────────────────────────────────────────

print("\nJAX sort:", jnp.sort(x))
print("SoftJAX sort (softsort):", sj.sort(x, method="softsort", softness=0.1))
print("SoftJAX sort (neuralsort):", sj.sort(x, method="neuralsort", softness=0.1))
print("SoftJAX sort (fast_soft_sort):", sj.sort(x, method="fast_soft_sort", softness=2.0))
print("SoftJAX sort (smooth_sort):", sj.sort(x, method="smooth_sort", softness=0.2))
print("SoftJAX sort (ot):", sj.sort(x, method="ot", softness=0.1))
print("SoftJAX sort (sorting_network):", sj.sort(x, method="sorting_network", softness=0.1))

# ── Sort: sweep over modes ───────────────────────────────────────────────────

print("\nJAX sort:", jnp.sort(x))
for mode in ["hard", "smooth", "c0", "c1", "c2"]:
    print(f"SoftJAX sort ({mode}):", sj.sort(x, softness=jnp.array(0.5), mode=mode))

# ── Operators returning indices ───────────────────────────────────────────────

print("\nJAX argmax:", jnp.argmax(x))
print("SoftJAX argmax (hard mode):", sj.argmax(x, mode="hard"))
print("SoftJAX argmax (soft mode):", sj.argmax(x))

print("\nJAX argmin:", jnp.argmin(x))
print("SoftJAX argmin (hard mode):", sj.argmin(x, mode="hard"))
print("SoftJAX argmin (soft mode):", sj.argmin(x))

print("\nJAX argquantile:", "Not implemented in standard JAX")
print("SoftJAX argquantile (hard mode):", sj.argquantile(x, q=0.2, mode="hard"))
print("SoftJAX argquantile (soft mode):", sj.argquantile(x, q=0.2))

print("\nJAX argmedian:", "Not implemented in standard JAX")
print("SoftJAX argmedian (hard mode):", sj.argmedian(x, mode="hard"))
print("SoftJAX argmedian (soft mode):", sj.argmedian(x))

print("\nJAX argsort:", jnp.argsort(x))
print("SoftJAX argsort (hard mode):", sj.argsort(x, mode="hard"))
print("SoftJAX argsort (soft mode):", sj.argsort(x))

print("\nJAX argtop_k:", jax.lax.top_k(x, k=3)[1])
print("SoftJAX argtop_k (hard mode):", sj.top_k(x, k=3, mode="hard")[1])
print("SoftJAX argtop_k (soft mode):", sj.top_k(x, k=3)[1])

# ── Comparison operators ──────────────────────────────────────────────────────

y = jnp.array([0.2, -0.5, 0.5, -1.0])

print("\nJAX greater:", jnp.greater(x, y))
print("SoftJAX greater (hard mode):", sj.greater(x, y, mode="hard"))
print("SoftJAX greater (soft mode):", sj.greater(x, y))

print("\nJAX greater equal:", jnp.greater_equal(x, y))
print("SoftJAX greater equal (hard mode):", sj.greater_equal(x, y, mode="hard"))
print("SoftJAX greater equal (soft mode):", sj.greater_equal(x, y))

print("\nJAX less:", jnp.less(x, y))
print("SoftJAX less (hard mode):", sj.less(x, y, mode="hard"))
print("SoftJAX less (soft mode):", sj.less(x, y))

print("\nJAX less equal:", jnp.less_equal(x, y))
print("SoftJAX less equal (hard mode):", sj.less_equal(x, y, mode="hard"))
print("SoftJAX less equal (soft mode):", sj.less_equal(x, y))

print("\nJAX equal:", jnp.equal(x, y))
print("SoftJAX equal (hard mode):", sj.equal(x, y, mode="hard"))
print("SoftJAX equal (soft mode):", sj.equal(x, y))

print("\nJAX not equal:", jnp.not_equal(x, y))
print("SoftJAX not equal (hard mode):", sj.not_equal(x, y, mode="hard"))
print("SoftJAX not equal (soft mode):", sj.not_equal(x, y))

print("\nJAX isclose:", jnp.isclose(x, y))
print("SoftJAX isclose (hard mode):", sj.isclose(x, y, mode="hard"))
print("SoftJAX isclose (soft mode):", sj.isclose(x, y))

# ── Logical operators ─────────────────────────────────────────────────────────

fuzzy_a = jnp.array([0.1, 0.2, 0.8, 1.0])
fuzzy_b = jnp.array([0.7, 0.3, 0.1, 0.9])
bool_a = fuzzy_a >= 0.5
bool_b = fuzzy_b >= 0.5

print("\nJAX AND:", jnp.logical_and(bool_a, bool_b))
print("SoftJAX AND:", sj.logical_and(fuzzy_a, fuzzy_b))

print("\nJAX OR:", jnp.logical_or(bool_a, bool_b))
print("SoftJAX OR:", sj.logical_or(fuzzy_a, fuzzy_b))

print("\nJAX NOT:", jnp.logical_not(bool_a))
print("SoftJAX NOT:", sj.logical_not(fuzzy_a))

print("\nJAX XOR:", jnp.logical_xor(bool_a, bool_b))
print("SoftJAX XOR:", sj.logical_xor(fuzzy_a, fuzzy_b))

print("\nJAX ALL:", jnp.all(bool_a))
print(f"SoftJAX ALL: {sj.all(fuzzy_a):.4f}")

print("\nJAX ANY:", jnp.any(bool_a))
print("SoftJAX ANY:", sj.any(fuzzy_a))

# Selection operators
print("\nJAX Where:", jnp.where(bool_a, x, y))
print("SoftJAX Where:", sj.where(fuzzy_a, x, y))

# ── Straight-through operators ────────────────────────────────────────────────

print("\nStraight-through ReLU:", sj.relu_st(x))
print("Straight-through sort:", sj.sort_st(x))
print("Straight-through argtop_k:", sj.top_k_st(x, k=3)[1])
print("Straight-through greater:", sj.greater_st(x, y))

# ── Autograd-safe operators ───────────────────────────────────────────────────

x = jnp.array([-0.2, -1.0, 0.3, 1.0])

print("\nJAX sqrt:", jnp.sqrt(jnp.abs(x)))
print("SoftJAX sqrt:", sj.sqrt(jnp.abs(x)))

print("\nJAX arcsin:", jnp.arcsin(x))
print("SoftJAX arcsin:", sj.arcsin(x))

print("\nJAX arccos:", jnp.arccos(x))
print("SoftJAX arccos:", sj.arccos(x))

print("\nJAX log:", jnp.log(jnp.array([0.0, 0.5, 1.0, 2.0])))
print("SoftJAX log:", sj.log(jnp.array([0.0, 0.5, 1.0, 2.0])))

print("\nJAX div (1/0):", jnp.array(1.0) / jnp.array(0.0))
print("SoftJAX div (1/0):", sj.div(jnp.array(1.0), jnp.array(0.0)))

print("\nJAX norm (zeros):", jnp.linalg.norm(jnp.zeros(3)))
print("SoftJAX norm (zeros):", sj.norm(jnp.zeros(3)))

print("\nGrad jnp.arcsin at x=1:", jax.grad(lambda z: jnp.arcsin(z))(1.0))
print("Grad sj.arcsin at x=1:", jax.grad(lambda z: sj.arcsin(z))(1.0))
print("Grad jnp.log at x=0:", jax.grad(lambda z: jnp.log(z))(0.0))
print("Grad sj.log at x=0:", jax.grad(lambda z: sj.log(z))(0.0))
