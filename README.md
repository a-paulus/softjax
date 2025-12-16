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

Available at https://a-paulus.github.io/softjax/.


## Quick example
```python
import jax.numpy as jnp
import softjax as sj

x = jnp.array([-0.2, -1.0, 0.3, 1.0])
y = jnp.array([0.2, -0.5, 0.5, -1.0])

# Elementwise functions
print("Hard ReLU:", jax.nn.relu(x))
print("Soft ReLU:", sj.relu(x))
print("Hard Clip:", jnp.clip(x, -0.5, 0.5))
print("Soft Clip:", sj.clip(x, -0.5, 0.5))
print("Hard Absolute:", jnp.abs(x))
print("Soft Absolute:", sj.abs(x))
print("Hard Sign:", jnp.sign(x))
print("Soft Sign:", sj.sign(x))
print("Hard round:", jnp.round(x))
print("Soft round:", sj.round(x))
print("Hard heaviside:", jnp.heaviside(x, 0.5))
print("Soft heaviside:", sj.heaviside(x))
```
```
Hard ReLU: [0.  0.  0.3 1. ]
Soft ReLU: [1.26928011e-02 4.53988992e-06 3.04858735e-01 1.00000454e+00]
Hard Clip: [-0.2 -0.5  0.3  0.5]
Soft Clip: [-0.19523241 -0.4993285   0.28734074  0.4993285 ]
Hard Absolute: [0.2 1.  0.3 1. ]
Soft Absolute: [0.15231883 0.9999092  0.27154448 0.9999092 ]
Hard Sign: [-1. -1.  1.  1.]
Soft Sign: [-0.76159416 -0.9999092   0.90514825  0.9999092 ]
Hard round: [-0. -1.  0.  1.]
Soft round: [-0.04651704 -1.          0.1188737   1.        ]
Hard heaviside: [0. 0. 1. 1.]
Soft heaviside: [1.19202922e-01 4.53978687e-05 9.52574127e-01 9.99954602e-01]
```

```python
# Functions on arrays
print("Hard max:", jnp.max(x))
print("Soft max:", sj.max(x))
print("Hard min:", jnp.min(x))
print("Soft min:", sj.min(x))
print("Hard median:", jnp.median(x))
print("Soft median:", sj.median(x))
print("Hard top_k:", jax.lax.top_k(x, k=3)[0])
print("Soft top_k:", sj.top_k(x, k=3)[0])
print("Hard sort:", jnp.sort(x))
print("Soft sort:", sj.sort(x))
print("Hard ranking:", jnp.argsort(jnp.argsort(x)))
print("Soft ranking:", sj.ranking(x, descending=False))
```
```
Hard max: 1.0
Soft max: 0.9993548976691374
Hard min: -1.0
Soft min: -0.9997287789452775
Hard median: 0.04999999999999999
Soft median: 0.05000033589501627
Hard top_k: [ 1.   0.3 -0.2]
Soft top_k: [ 0.9993549   0.29728716 -0.19691387]
Hard sort: [-1.  -0.2  0.3  1. ]
Soft sort: [-0.99972878 -0.19691387  0.29728716  0.9993549 ]
Hard ranking: [1 0 2 3]
Soft ranking: [1.00636968e+00 3.39874686e-04 1.99421369e+00 2.99907667e+00]
```

```python
# Functions returning indices
print("Hard argmax:", jnp.argmax(x))
print("Soft argmax:", sj.argmax(x))
print("Hard argmin:", jnp.argmin(x))
print("Soft argmin:", sj.argmin(x))
print("Hard argmedian:", "Not implemented in standard JAX")
print("Soft argmedian:", sj.argmedian(x))
print("Hard argtop_k:", jax.lax.top_k(x, k=3)[1])
print("Soft argtop_k:", sj.top_k(x, k=3)[1])
print("Hard argsort:", jnp.argsort(x))
print("Soft argsort:", sj.argsort(x))
```
```
Hard argmax: 3
Soft argmax: [6.13857697e-06 2.05926316e-09 9.11045600e-04 9.99082814e-01]
Hard argmin: 1
Soft argmin: [3.35349372e-04 9.99662389e-01 2.25956629e-06 2.06045775e-09]
Hard argmedian: Not implemented in standard JAX
Soft argmedian: [4.99999764e-01 5.62675608e-08 4.99999764e-01 4.15764163e-07]
Hard argtop_k: [3 2 0]
Soft argtop_k: [[6.13857697e-06 2.05926316e-09 9.11045600e-04 9.99082814e-01]
 [6.68677917e-03 2.24316451e-06 9.92406021e-01 9.04957153e-04]
 [9.92970214e-01 3.33104397e-04 6.69058067e-03 6.10101985e-06]]
Hard argsort: [1 0 2 3]
Soft argsort: [[3.35349372e-04 9.99662389e-01 2.25956629e-06 2.06045775e-09]
 [9.92970214e-01 3.33104397e-04 6.69058067e-03 6.10101985e-06]
 [6.68677917e-03 2.24316451e-06 9.92406021e-01 9.04957153e-04]
 [6.13857697e-06 2.05926316e-09 9.11045600e-04 9.99082814e-01]]
```

```python
## SoftBool generation
print("Hard greater:", x > y)
print("Soft greater:", sj.greater(x, y))
print("Hard greater equal:", x >= y)
print("Soft greater equal:", sj.greater_equal(x, y))
print("Hard less:", x < y)
print("Soft less:", sj.less(x, y))
print("Hard less equal:", x <= y)
print("Soft less equal:", sj.less_equal(x, y))
print("Hard equal:", x == y)
print("Soft equal:", sj.equal(x, y))
print("Hard not equal:", x != y)
print("Soft not equal:", sj.not_equal(x, y))
print("Hard isclose:", jnp.isclose(x, y))
print("Soft isclose:", sj.isclose(x, y))
```
```
Hard greater: [False False False  True]
Soft greater: [0.01798621 0.00669285 0.11920292 1.        ]
Hard greater equal: [False False False  True]
Soft greater equal: [0.01798621 0.00669285 0.11920292 1.        ]
Hard less: [ True  True  True False]
Soft less: [9.82013790e-01 9.93307149e-01 8.80797078e-01 2.06115369e-09]
Hard less equal: [ True  True  True False]
Soft less equal: [9.82013790e-01 9.93307149e-01 8.80797078e-01 2.06115369e-09]
Hard equal: [False False False False]
Soft equal: [1.79862100e-02 6.69285093e-03 1.19202922e-01 2.06115369e-09]
Hard not equal: [ True  True  True  True]
Soft not equal: [0.98201379 0.99330715 0.88079708 1.        ]
Hard isclose: [False False False False]
Soft isclose: [1.79865650e-02 6.69318401e-03 1.19208182e-01 2.06135997e-09]
```

```python
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
Soft AND: [0.26457513 0.24494897 0.28284271 0.9486833 ]
Soft OR: [0.48038476 0.25166852 0.57573593 0.99999684]
Soft NOT: [0.9 0.8 0.2 0. ]
Soft XOR: [0.58702688 0.43498731 0.63937484 0.17309871]
Soft ALL: 0.35565588200778464
Soft ANY: 0.9980519925071494
Where: [ 0.16 -0.6   0.34  1.  ]
```

```python
# Straight-through estimation: Use hard function on forward and soft on backward
print("Straight-through ReLU:", sj.relu_st(x))
print("Straight-through sort:", sj.sort_st(x))
print("Straight-through argtop_k:", sj.top_k_st(x, k=3)[1])
print("Straight-through greater:", sj.greater_st(x, y))
```
```
Straight-through ReLU: [0.  0.  0.3 1. ]
Straight-through sort: [-1.  -0.2  0.3  1. ]
Straight-through argtop_k: [[0. 0. 0. 1.]
 [0. 0. 1. 0.]
 [1. 0. 0. 0.]]
Straight-through greater: [0. 0. 0. 1.]
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