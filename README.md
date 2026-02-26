<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/_static/logo/softjax_logo_white_transparent.png">
    <source media="(prefers-color-scheme: light)" srcset="docs/_static/logo/softjax_logo_black_transparent.png">
    <img alt="SoftJAX logo" src="docs/_static/logo/softjax_logo_black_transparent.png" style="width:60%; max-width:320px; height:auto;">
  </picture>
</p>

# SoftJAX

[![PyPI version](https://img.shields.io/pypi/v/softjax)](https://pypi.org/project/softjax/)
[![Python version](https://img.shields.io/pypi/pyversions/softjax)](https://pypi.org/project/softjax/)
[![License](https://img.shields.io/pypi/l/softjax)](https://github.com/a-paulus/softjax/blob/main/LICENSE)

## In a nutshell

SoftJAX provides soft differentiable drop-in replacements for traditionally non-differentiable functions in [JAX](https://github.com/google/jax), including

- elementwise operators: `abs`, `relu`, `clip`, `sign`, `round` and `heaviside`;
- array-valued operators: `(arg)max`, `(arg)min`, `(arg)quantile`, `(arg)median`, `(arg)sort`, `(arg)top_k` and `rank`;
- comparison operators such as: `greater`, `equal` or `isclose`;
- logical operators such as: `logical_and`, `all` or `any`;
- selection operators such as: `where`, `take_along_axis`, `dynamic_index_in_dim` or `choose`.

All operators offer multiple modes and adjustable strength of softening, allowing for e.g. smoothness of the soft function or boundedness of the softened region, depending on the user needs.

Moreover, we tightly integrate functionality for deploying functions using [straight-through-estimation](https://docs.jax.dev/en/latest/advanced-autodiff.html#straight-through-estimator-using-stop-gradient), where we use non-differentiable functions in the forward pass and their differentiable replacements in the backward pass.

The SoftJAX library is designed to require minimal user effort, by simply replacing the non-differentiable JAX function with the SoftJAX counterparts.
However, keep in mind that special care needs to be taken when using functions operating on indices, as we relax the notion of an index into a distribution over indices, thereby modifying the shape of returned/accepted values.


## Installation
Requires Python 3.11+.
```
pip install softjax
```


## Documentation

Available at https://a-paulus.github.io/softjax/.


## Quick example
```python
import jax
import jax.numpy as jnp
import softjax as sj

x = jnp.array([-0.2, -1.0, 0.3, 1.0])

# Elementwise operators
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
```
```
JAX absolute: [0.2 1.  0.3 1. ]
SoftJAX absolute (hard mode): [0.2 1.  0.3 1. ]
SoftJAX absolute (soft mode): [0.1523 0.9999 0.2715 0.9999]

JAX clip: [-0.2 -0.5  0.3  0.5]
SoftJAX clip (hard mode): [-0.2 -0.5  0.3  0.5]
SoftJAX clip (soft mode): [-0.1952 -0.4993  0.2873  0.4993]

JAX heaviside: [0. 0. 1. 1.]
SoftJAX heaviside (hard mode): [0. 0. 1. 1.]
SoftJAX heaviside (soft mode): [0.1192 0.     0.9526 1.    ]

JAX ReLU: [0.  0.  0.3 1. ]
SoftJAX ReLU (hard mode): [0.  0.  0.3 1. ]
SoftJAX ReLU (soft mode): [0.0127 0.     0.3049 1.    ]

JAX round: [-0. -1.  0.  1.]
SoftJAX round (hard mode): [-0. -1.  0.  1.]
SoftJAX round (soft mode): [-0.0465 -1.      0.1189  1.    ]

JAX sign: [-1. -1.  1.  1.]
SoftJAX sign (hard mode): [-1. -1.  1.  1.]
SoftJAX sign (soft mode): [-0.7616 -0.9999  0.9051  0.9999]
```

```python
# Array-valued operators
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
```
```
JAX max: 1.0
SoftJAX max (hard mode): 1.0
SoftJAX max (soft mode): 0.8874

JAX min: -1.0
SoftJAX min (hard mode): -1.0
SoftJAX min (soft mode): -0.8996

JAX sort: [-1.  -0.2  0.3  1. ]
SoftJAX sort (hard mode): [-1.  -0.2  0.3  1. ]
SoftJAX sort (soft mode): [-0.8792 -0.1641  0.2767  0.8738]

JAX quantile: -0.5200
SoftJAX quantile (hard mode): -0.5200
SoftJAX quantile (soft mode): -0.4501

JAX median: 0.0500
SoftJAX median (hard mode): 0.0500
SoftJAX median (soft mode): 0.0563

JAX top_k: [ 1.   0.3 -0.2]
SoftJAX top_k (hard mode): [ 1.   0.3 -0.2]
SoftJAX top_k (soft mode): [ 0.8738  0.2767 -0.1641]

JAX rank: [1 0 2 3]
SoftJAX rank (hard mode): [2. 1. 3. 4.]
SoftJAX rank (soft mode): [1.995  1.0548 3.0239 3.9228]
```

```python
# Sort: sweep over methods
print("\nJAX sort:", jnp.sort(x))
print("SoftJAX sort (softsort):", sj.sort(x, method="softsort", softness=0.1))
print("SoftJAX sort (neuralsort):", sj.sort(x, method="neuralsort", softness=0.1))
print("SoftJAX sort (fast_soft_sort):", sj.sort(x, method="fast_soft_sort", softness=2.0))
print("SoftJAX sort (smooth_sort):", sj.sort(x, method="smooth_sort", softness=0.2))
print("SoftJAX sort (ot):", sj.sort(x, method="ot", softness=0.1))
print("SoftJAX sort (sorting_network):", sj.sort(x, method="sorting_network", softness=0.1))

# Sort: sweep over modes
print("\nJAX sort:", jnp.sort(x))
for mode in ["hard", "smooth", "c0", "c1", "c2"]:
    print(f"SoftJAX sort ({mode}):", sj.sort(x, softness=jnp.array(0.5), mode=mode))
```
```
JAX sort: [-1.  -0.2  0.3  1. ]
SoftJAX sort (softsort): [-0.8996 -0.1705  0.2847  0.8874]
SoftJAX sort (neuralsort): [-0.8792 -0.1641  0.2767  0.8738]
SoftJAX sort (fast_soft_sort): [-0.7462 -0.1971  0.2938  0.8569]
SoftJAX sort (smooth_sort): [-0.8572 -0.2221  0.2973  0.8821]
SoftJAX sort (ot): [-0.7324 -0.2396  0.3286  0.7434]
SoftJAX sort (sorting_network): [-0.7999 -0.2672  0.3847  0.7863]

JAX sort: [-1.  -0.2  0.3  1. ]
SoftJAX sort (hard): [-1.  -0.2  0.3  1. ]
SoftJAX sort (smooth): [-0.6057 -0.1997  0.2729  0.6281]
SoftJAX sort (c0): [-1.     -0.6313  0.6525  0.9824]
SoftJAX sort (c1): [-0.9982 -0.5432  0.5814  0.9837]
SoftJAX sort (c2): [-0.9978 -0.4905  0.5425  0.9903]
```

```python
# Operators returning indices
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
```
```
JAX argmax: 3
SoftJAX argmax (hard mode): [0. 0. 0. 1.]
SoftJAX argmax (soft mode): [0.0215 0.0022 0.1176 0.8586]

JAX argmin: 1
SoftJAX argmin (hard mode): [0. 1. 0. 0.]
SoftJAX argmin (soft mode): [0.0922 0.8885 0.0169 0.0023]

JAX argquantile: Not implemented in standard JAX
SoftJAX argquantile (hard mode): [0.6 0.4 0.  0. ]
SoftJAX argquantile (soft mode): [0.5403 0.3693 0.0902 0.0001]

JAX argmedian: Not implemented in standard JAX
SoftJAX argmedian (hard mode): [0.5 0.  0.5 0. ]
SoftJAX argmedian (soft mode): [0.4714 0.0246 0.4699 0.0342]

JAX argsort: [1 0 2 3]
SoftJAX argsort (hard mode): [[0. 1. 0. 0.]
 [1. 0. 0. 0.]
 [0. 0. 1. 0.]
 [0. 0. 0. 1.]]
SoftJAX argsort (soft mode): [[0.1494 0.8496 0.0009 0.    ]
 [0.8009 0.0491 0.1498 0.0002]
 [0.1418 0.0001 0.7899 0.0681]
 [0.0011 0.     0.1784 0.8205]]

JAX argtop_k: [3 2 0]
SoftJAX argtop_k (hard mode): [[0. 0. 0. 1.]
 [0. 0. 1. 0.]
 [1. 0. 0. 0.]]
SoftJAX argtop_k (soft mode): [[0.0011 0.     0.1784 0.8205]
 [0.1418 0.0001 0.7899 0.0681]
 [0.8009 0.0491 0.1498 0.0002]]
```

```python
y = jnp.array([0.2, -0.5, 0.5, -1.0])

# Comparison operators
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
```
```
JAX greater: [False False False  True]
SoftJAX greater (hard mode): [0. 0. 0. 1.]
SoftJAX greater (soft mode): [0.018  0.0067 0.1192 1.    ]

JAX greater equal: [False False False  True]
SoftJAX greater equal (hard mode): [0. 0. 0. 1.]
SoftJAX greater equal (soft mode): [0.018  0.0067 0.1192 1.    ]

JAX less: [ True  True  True False]
SoftJAX less (hard mode): [1. 1. 1. 0.]
SoftJAX less (soft mode): [0.982  0.9933 0.8808 0.    ]

JAX less equal: [ True  True  True False]
SoftJAX less equal (hard mode): [1. 1. 1. 0.]
SoftJAX less equal (soft mode): [0.982  0.9933 0.8808 0.    ]

JAX equal: [False False False False]
SoftJAX equal (hard mode): [0. 0. 0. 0.]
SoftJAX equal (soft mode): [0.0414 0.0143 0.358  0.    ]

JAX not equal: [ True  True  True  True]
SoftJAX not equal (hard mode): [1. 1. 1. 1.]
SoftJAX not equal (soft mode): [0.9586 0.9857 0.642  1.    ]

JAX isclose: [False False False False]
SoftJAX isclose (hard mode): [0. 0. 0. 0.]
SoftJAX isclose (soft mode): [0.0414 0.0143 0.358  0.    ]
```

```python
# Logical operators
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
```
```
JAX AND: [False False False  True]
SoftJAX AND: [0.07 0.06 0.08 0.9 ]

JAX OR: [ True False  True  True]
SoftJAX OR: [0.73 0.44 0.82 1.  ]

JAX NOT: [ True  True False False]
SoftJAX NOT: [0.9 0.8 0.2 0. ]

JAX XOR: [ True False  True False]
SoftJAX XOR: [0.6411 0.3464 0.7256 0.1   ]

JAX ALL: False
SoftJAX ALL: 0.0160

JAX ANY: True
SoftJAX ANY: 1.0

JAX Where: [ 0.2 -0.5  0.3  1. ]
SoftJAX Where: [ 0.16 -0.6   0.34  1.  ]
```

```python
# Straight-through operators: Use hard function on forward and soft on backward
print("Straight-through ReLU:", sj.relu_st(x))
print("Straight-through sort:", sj.sort_st(x))
print("Straight-through argtop_k:", sj.top_k_st(x, k=3)[1])
print("Straight-through greater:", sj.greater_st(x, y))
# And many more...
```
```
Straight-through ReLU: [0.  0.  0.3 1. ]
Straight-through sort: [-1.  -0.2  0.3  1. ]
Straight-through argtop_k: [[0. 0. 0. 1.]
 [0. 0. 1. 0.]
 [1. 0. 0. 0.]]
Straight-through greater: [0. 0. 0. 1.]
```

```python
# Autograd-safe operators: safe gradients at boundary points
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
```
```
JAX sqrt: [0.4472 1.     0.5477 1.    ]
SoftJAX sqrt: [0.4472 1.     0.5477 1.    ]

JAX arcsin: [-0.2014 -1.5708  0.3047  1.5708]
SoftJAX arcsin: [-0.2014 -1.5708  0.3047  1.5708]

JAX arccos: [1.7722 3.1416 1.2661 0.    ]
SoftJAX arccos: [1.7722 3.1416 1.2661 0.    ]

JAX log: [   -inf -0.6931  0.      0.6931]
SoftJAX log: [ 0.     -0.6931  0.      0.6931]

JAX div (1/0): inf
SoftJAX div (1/0): 0.0

JAX norm (zeros): 0.0
SoftJAX norm (zeros): 0.0

Grad jnp.arcsin at x=1: inf
Grad sj.arcsin at x=1: 0.0
Grad jnp.log at x=0: inf
Grad sj.log at x=0: 0.0
```


## Citation

If this library helped your academic work, please consider citing:

```bibtex
@article{paulus2026softjax,
  title={{SoftJAX} \& {SoftTorch}: Empowering Automatic Differentiation Libraries with Informative Gradients},
  author={Paulus, Anselm and Geist, A.\ Ren\'e and Musil, V\'it and Hoffmann, Sebastian and Beker, Onur and Martius, Georg},
  journal={arXiv preprint},
  year={2026}
}
```

Also consider starring the project [on GitHub](https://github.com/a-paulus/softjax)!

Special thanks and credit go to [Patrick Kidger](https://kidger.site) for the awesome [JAX repositories](https://github.com/patrick-kidger) that served as the basis for the documentation of this project.


## Feedback

This project is still relatively young, if you have any suggestions for improvement or other feedback, please [reach out](mailto:paulus.anselm@gmail.com) or raise a GitHub issue!


## See also

### Other libraries in the JAX ecosystem

**Always useful**  
[Equinox](https://github.com/patrick-kidger/equinox): neural networks and everything not already in core JAX!  
[jaxtyping](https://github.com/patrick-kidger/jaxtyping): type annotations for shape/dtype of arrays.  

**Deep learning**  
[Optax](https://github.com/deepmind/optax): first-order gradient (SGD, Adam, ...) optimisers.  
[Orbax](https://github.com/google/orbax): checkpointing (async/multi-host/multi-device).  
[Levanter](https://github.com/stanford-crfm/levanter): scalable+reliable training of foundation models (e.g. LLMs).  
[paramax](https://github.com/danielward27/paramax): parameterizations and constraints for PyTrees.  

**Scientific computing**  
[Diffrax](https://github.com/patrick-kidger/diffrax): numerical differential equation solvers.  
[Optimistix](https://github.com/patrick-kidger/optimistix): root finding, minimisation, fixed points, and least squares.  
[Lineax](https://github.com/patrick-kidger/lineax): linear solvers.  
[BlackJAX](https://github.com/blackjax-devs/blackjax): probabilistic+Bayesian sampling.  
[sympy2jax](https://github.com/patrick-kidger/sympy2jax): SymPy<->JAX conversion; train symbolic expressions via gradient descent.  
[PySR](https://github.com/milesCranmer/PySR): symbolic regression. (Non-JAX honourable mention!)  

**Awesome JAX**  
[Awesome JAX](https://github.com/n2cholas/awesome-jax): a longer list of other JAX projects.  

### Other libraries on differentiable programming

**Differentiable sorting, top-k and rank**  
[DiffSort](https://github.com/Felix-Petersen/diffsort): Differentiable sorting networks in PyTorch.  
[DiffTopK](https://github.com/Felix-Petersen/difftopk): Differentiable top-k in PyTorch.  
[FastSoftSort](https://github.com/google-research/fast-soft-sort): Fast differentiable sorting and rank in JAX.  
[Differentiable Top-k with Optimal Transport](https://gist.github.com/thomasahle/48e9b3f17ead6c3ef11325f25de3655e) in JAX.  
[SoftSort](https://github.com/sprillo/softsort): Differentiable argsort in PyTorch and TensorFlow.  

**Other**  
[DiffLogic](https://github.com/Felix-Petersen/difflogic): Differentiable logic gate networks in PyTorch.  
[SmoothOT](https://github.com/mblondel/smooth-ot): Smooth and Sparse Optimal Transport.  
[JaxOpt](https://github.com/google/jaxopt): Differentiable optimization in JAX.  

### Papers on differentiable algorithms
SoftJAX builds on / implements various different algorithms for e.g. differentiable `argtop_k`, `sorting` and `rank`, including:

[Projection onto the probability simplex: An efficient algorithm with a simple proof, and an application](https://arxiv.org/pdf/1309.1541)  
[Differentiable Ranks and Sorting using Optimal Transport](https://arxiv.org/pdf/1905.11885)  
[Differentiable Top-k with Optimal Transport](https://papers.nips.cc/paper/2020/file/ec24a54d62ce57ba93a531b460fa8d18-Paper.pdf)  
[SoftSort: A Continuous Relaxation for the argsort Operator](https://arxiv.org/pdf/2006.16038)  
[Sinkhorn Distances: Lightspeed Computation of Optimal Transportation Distances](https://arxiv.org/abs/1306.0895)  
[Smooth and Sparse Optimal Transport](https://arxiv.org/abs/1710.06276)  
[Smooth Approximations of the Rounding Function](https://arxiv.org/pdf/2504.19026v1)  
[Fast Differentiable Sorting and Ranking](https://arxiv.org/pdf/2002.08871)  
[Differentiable Sorting Networks for Scalable Sorting and Ranking Supervision](https://arxiv.org/abs/2105.04019)  

Please check the [API Documentation](https://a-paulus.github.io/softjax/api/softjax_operators) for implementation details.