# Softjax

Softjax provides soft differentiable drop-in replacements for traditionally non-differentiable functions in [JAX](https://github.com/google/jax), including

- simple elementwise functions: `abs`, `relu`, `clip`, `sign` and `round`;
- functions operating on arrays: `max`, `min`, `median`, `sort`, `ranking` and `top_k`;
- functions returning indices: `argmax`, `argmin`, `argmedian`, `argsort` and `argtop_k`;
- functions returning boolean values such as: `greater`, `equal` or `isclose`;
- functions for selection with indices such as: `take_along_axis`, `dynamic_index_in_dim` and `choose`;
- functions for logical manipulation such as: `logical_and`, `all` and `where`.

Many functions offer multiple modes for softening, allowing for e.g. smoothness of the soft function or boundedness of the softened region, depending on the user needs.
Moreover, we tightly integrate functionality for deploying functions using [straight-through-estimation](https://docs.jax.dev/en/latest/advanced-autodiff.html#straight-through-estimator-using-stop-gradient), where we use non-differentiable functions in the forward pass and their differentiable replacements in the backward pass.

The Softjax library is designed to require minimal user effort, by simply replacing the non-differentiable JAX function with the Softjax counterparts.
However, keep in mind that special care needs to be taken when using functions operating on indices, as we relax the notion of an index into a distribution over indices, thereby modifying the shape of returned/accepted values.


## Installation
Requires Python 3.10+.
```
pip install softjax
```


## Documentation

Available at https://docs.a-paulus.github.io/softjax.


## Quick example
```python
import jax.numpy as jnp
import softjax as sj

softness = 0.1
x = jnp.array([-0.1, 0.0, 0.7, 1.8])
y = jnp.array([0.2, -0.5, 0.5, -1.0])

# Elementwise functions
print("Hard ReLU:", jax.nn.relu(x))
print("Soft ReLU:", sj.relu(x, softness=softness))
print("Hard Clip:", jnp.clip(x, -0.5, 0.5))
print("Soft Clip:", sj.clip(x, -0.5, 0.5, softness=softness))
print("Hard Absolute:", jnp.abs(x))
print("Soft Absolute:", sj.abs(x, softness=softness))
print("Hard Sign:", jnp.sign(x))
print("Soft Sign:", sj.sign(x, softness=softness))
print("Hard round:", jnp.round(x))
print("Soft round:", sj.round(x, softness=softness))

# Functions on arrays
print("Hard max:", jnp.max(x))
print("Soft max:", sj.max(x, softness=softness))
print("Hard min:", jnp.min(x))
print("Soft min:", sj.min(x, softness=softness))
print("Hard median:", jnp.median(x))
print("Soft median:", sj.median(x, softness=softness))
print("Hard top_k:", jax.lax.top_k(x, k=3)[0])
print("Soft top_k:", sj.top_k(x, k=3, softness=softness)[0])
print("Hard sort:", jnp.sort(x))
print("Soft sort:", sj.sort(x, softness=softness))
print("Hard ranking:", jnp.argsort(jnp.argsort(x)))
print("Soft ranking:", sj.ranking(x, softness=softness))

# Straight-through estimation: Use hard function on forward and soft on backward
print("Straight-through sort:", sj.sort_st(x, softness=softness))

# Functions returning indices
print("Hard argmax:", jnp.argmax(x))
print("Soft argmax:", sj.argmax(x, softness=softness))
print("Hard argmin:", jnp.argmin(x))
print("Soft argmin:", sj.argmin(x, softness=softness))
print("Hard argmedian:", "Not implemented in standard JAX")
print("Soft argmedian:", sj.argmedian(x, softness=softness))
print("Hard argtop_k:", jax.lax.top_k(x, k=3)[1])
print("Soft argtop_k:", sj.top_k(x, k=3, softness=softness)[1])
print("Hard argsort:", jnp.argsort(x))
print("Soft argsort:", sj.argsort(x, softness=softness))

## SoftBool generation
print("Hard heaviside:", jnp.heaviside(x, 0.5))
print("Soft heaviside:", sj.heaviside(x, softness=softness))
print("Hard greater:", x > y)
print("Soft greater:", sj.greater(x, y, softness=softness))
print("Hard greater equal:", x >= y)
print("Soft greater equal:", sj.greater_equal(x, y, softness=softness))
print("Hard less:", x < y)
print("Soft less:", sj.less(x, y, softness=softness))
print("Hard less equal:", x <= y)
print("Soft less equal:", sj.less_equal(x, y, softness=softness))
print("Hard equal:", x == y)
print("Soft equal:", sj.equal(x, y, softness=softness))
print("Hard not equal:", x != y)
print("Soft not equal:", sj.not_equal(x, y, softness=softness))
print("Hard isclose:", jnp.isclose(x, y))
print("Soft isclose:", sj.isclose(x, y, softness=softness))

## SoftBool manipulation
fuzzy_a = jnp.array([0.1, 0.2, 0.8, 1.0])
fuzzy_b = jnp.array([0.7, 0.3, 0.1, 0.9])
print("Soft AND:", sj.logical_and(fuzzy_a, fuzzy_b))
print("Soft OR:", sj.logical_or(fuzzy_a, fuzzy_b))
print("Soft NOT:", sj.logical_not(fuzzy_a))
print("Soft XOR:", sj.logical_xor(fuzzy_a, fuzzy_b))
print("Soft ALL:", sj.all(fuzzy_a))
print("Soft ANY:", sj.any(fuzzy_a))

## SoftBool selection
print("Where:", sj.where(fuzzy_a, x, y))
```

```
Hard ReLU: [0.  0.  0.7 1.8]
Soft ReLU: [0.03132617 0.06931472 0.70009115 1.8       ]
Hard Clip: [-0.1  0.   0.5  0.5]
Soft Clip: [-9.84325757e-02 -3.46944695e-18  4.87307813e-01  4.99999774e-01]
Hard Absolute: [0.1 0.  0.7 1.8]
Soft Absolute: [0.04621172 0.         0.69872453 1.79999995]
Hard Sign: [-1.  0.  1.  1.]
Soft Sign: [-0.46211716  0.          0.9981779   0.99999997]
Hard round: [-0.  0.  1.  2.]
Soft round: [-0.09064511  0.          0.71513653  1.81513653]
Hard max: 1.8
Soft max: 1.7999815903777097
Hard min: -0.1
Soft min: -0.07291629800981214
Hard median: 0.35
Soft median: 0.24772037254528773
Hard top_k: [1.8 0.7 0. ]
Soft top_k: [ 1.79998159  0.69911281 -0.02640987]
Hard sort: [-0.1  0.   0.7  1.8]
Soft sort: [-0.0729163  -0.02640987  0.69911281  1.79998159]
Hard ranking: [0 1 2 3]
Soft ranking: [2.73063414e+00 2.26809603e+00 1.00156413e+00 1.67486891e-05]
Straight-through sort: [-0.1  0.   0.7  1.8]
Hard argmax: 3
Soft argmax: [5.60270275e-09 1.52297251e-08 1.67014215e-05 9.99983278e-01]
Hard argmin: 0
Soft argmin: [7.30879333e-01 2.68875480e-01 2.45182702e-04 4.09496812e-09]
Hard argmedian: Not implemented in standard JAX
Soft argmedian: [0.23233226 0.38305115 0.38305115 0.00156544]
Hard argtop_k: [3 2 1]
Soft argtop_k: [[5.60270275e-09 1.52297251e-08 1.67014215e-05 9.99983278e-01]
 [3.35039123e-04 9.10730760e-04 9.98737550e-01 1.66806157e-05]
 [2.68762251e-01 7.30571543e-01 6.66195015e-04 1.11265898e-08]]
Hard argsort: [0 1 2 3]
Soft argsort: [[7.30879333e-01 2.68875480e-01 2.45182702e-04 4.09496812e-09]
 [2.68762251e-01 7.30571543e-01 6.66195015e-04 1.11265898e-08]
 [3.35039123e-04 9.10730760e-04 9.98737550e-01 1.66806157e-05]
 [5.60270275e-09 1.52297251e-08 1.67014215e-05 9.99983278e-01]]
Hard heaviside: [0.  0.5 1.  1. ]
Soft heaviside: [0.26894142 0.5        0.99908895 0.99999998]
Hard greater: [False  True  True  True]
Soft greater: [0.04742587 0.99330715 0.88079708 1.        ]
Hard greater equal: [False  True  True  True]
Soft greater equal: [0.04742587 0.99330715 0.88079708 1.        ]
Hard less: [ True False False False]
Soft less: [9.52574127e-01 6.69285092e-03 1.19202922e-01 6.91446900e-13]
Hard less equal: [ True False False False]
Soft less equal: [9.52574127e-01 6.69285093e-03 1.19202922e-01 6.91446900e-13]
Hard equal: [False False False False]
Soft equal: [4.74258732e-02 6.69285093e-03 1.19202922e-01 6.91446900e-13]
Hard not equal: [ True  True  True  True]
Soft not equal: [0.95257413 0.99330715 0.88079708 1.        ]
Hard isclose: [False False False False]
Soft isclose: [4.74267813e-02 6.69318401e-03 1.19208182e-01 6.91446900e-13]
Soft AND: [0.26457513 0.24494897 0.28284271 0.9486833 ]
Soft OR: [0.48038476 0.25166852 0.57573593 0.99999684]
Soft NOT: [0.9 0.8 0.2 0. ]
Soft XOR: [0.58702688 0.43498731 0.63937484 0.17309871]
Soft ALL: 0.3556558820077846
Soft ANY: 0.9980519925071494
Where: [ 0.17 -0.4   0.66  1.8 ]
```


## Citation

If this library helped your academic work, please consider citing:

```bibtex
@misc{Softjax2025,
  author = {Paulus, Anselm and Geist, Rene and Martius, Georg},
  title = {Softjax},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/a-paulus/softjax}}
}
```

Also consider starring the project [on GitHub](https://github.com/a-paulus/softjax)!

Special thanks and credit go to [Patrick Kidger](https://kidger.site) for the awesome [JAX repositories](https://github.com/patrick-kidger) that served as the basis for the documentation of this project.


## Feedback

This project is still relatively young, if you have any suggestions for improvement or other feedback, please [reach out](mailto:anselm-valentin.paulus@uni-tuebingen.de) or raise a GitHub issue!


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
[Optimistix](https://github.com/patrick-kidger/optimistix): root finding, minimisation, fixed points, and least squares.  
[Lineax](https://github.com/patrick-kidger/lineax): linear solvers.  
[BlackJAX](https://github.com/blackjax-devs/blackjax): probabilistic+Bayesian sampling.  
[sympy2jax](https://github.com/patrick-kidger/sympy2jax): SymPy<->JAX conversion; train symbolic expressions via gradient descent.  
[PySR](https://github.com/milesCranmer/PySR): symbolic regression. (Non-JAX honourable mention!)  

**Awesome JAX**  
[Awesome JAX](https://github.com/n2cholas/awesome-jax): a longer list of other JAX projects.  

### Other libraries on differentiable programming

**Differentiable sorting, top-k and ranking**  
[DiffSort](https://github.com/Felix-Petersen/diffsort): Differentiable sorting networks in PyTorch.  
[DiffTopK](https://github.com/Felix-Petersen/difftopk): Differentiable top-k in PyTorch.  
[FastSoftSort](https://github.com/google-research/fast-soft-sort): Fast differentiable sorting and ranking in JAX.  
[Differentiable Top-k with Optimal Transport](https://gist.github.com/thomasahle/48e9b3f17ead6c3ef11325f25de3655e) in JAX.  
[SoftSort](https://github.com/sprillo/softsort): Differentiable argsort in PyTorch and TensorFlow.  

**Other**  
[DiffLogic](https://github.com/Felix-Petersen/difflogic): Differentiable logic gate networks in PyTorch.  
[SmoothOT](https://github.com/mblondel/smooth-ot): Smooth and Sparse Optimal Transport.  
[JaxOpt](https://github.com/google/jaxopt): Differentiable optimization in JAX.  

### Papers on differentiable algorithms
Softjax builds on / implements various different algoithms for e.g. differentiable `argtop_k`, `sorting` and `ranking`, including:

[Projection onto the probability simplex: An efficient algorithm with a simple proof, and an application](https://arxiv.org/pdf/1309.1541)  
[Fast Differentiable Sorting and Ranking](https://arxiv.org/pdf/2002.08871).  
[Differentiable Ranks and Sorting using Optimal Transport](https://arxiv.org/pdf/1905.11885)  
[Differentiable Top-k with Optimal Transport](https://papers.nips.cc/paper/2020/file/ec24a54d62ce57ba93a531b460fa8d18-Paper.pdf)  
[SoftSort: A Continuous Relaxation for the argsort Operator](https://arxiv.org/pdf/2006.16038)  
[Sinkhorn Distances: Lightspeed Computation of Optimal Transportation Distances](https://arxiv.org/abs/1306.0895)  
[Smooth and Sparse Optimal Transport](https://arxiv.org/abs/1710.06276)  

Please check the [API Documentation](https://docs.a-paulus.github.io/softjax/api/soft_indices) for implementation details.