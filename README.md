# Softjax

## Disclaimer

Softjax is not yet fully released!
We are currently finalizing the library, and are planning on releasing it (alongside a similar "[Softtorch](https://github.com/a-paulus/softtorch)" library) officially until the end of the year.
If you somehow stumbled upon this library already, feel free to use and test the github code, and please reach out if you encounter any issues or have suggestions for improvement. Thanks!

Note also that some of the API and internals are still subject to potentially bigger changes until the official release.
The pip install will also only be available after official release.


## In a nutshell

Softjax provides soft differentiable drop-in replacements for traditionally non-differentiable functions in [JAX](https://github.com/google/jax), including

- simple elementwise functions: `abs`, `relu`, `clip`, `sign` and `round`;
- functions operating on arrays: `max`, `min`, `median`, `sort`, `ranking` and `top_k`;
- functions returning indices: `argmax`, `argmin`, `argmedian`, `argsort` and `top_k`;
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
import jax
import jax.numpy as jnp
import softjax as sj

x = jnp.array([-0.2, -1.0, 0.3, 1.0])

# Elementwise functions
print("\nJAX ReLU:", jax.nn.relu(x))
print("SoftJAX ReLU (hard mode):", sj.relu(x, mode="hard"))
print("SoftJAX ReLU (soft mode):", sj.relu(x))

print("\nJAX Clip:", jnp.clip(x, -0.5, 0.5))
print("SoftJAX Clip (hard mode):", sj.clip(x, -0.5, 0.5, mode="hard"))
print("SoftJAX Clip (soft mode):", sj.clip(x, -0.5, 0.5))

print("\nJAX Absolute:", jnp.abs(x))
print("SoftJAX Absolute (hard mode):", sj.abs(x, mode="hard"))
print("SoftJAX Absolute (soft mode):", sj.abs(x))

print("\nJAX Sign:", jnp.sign(x))
print("SoftJAX Sign (hard mode):", sj.sign(x, mode="hard"))
print("SoftJAX Sign (soft mode):", sj.sign(x))

print("\nJAX round:", jnp.round(x))
print("SoftJAX round (hard mode):", sj.round(x, mode="hard"))
print("SoftJAX round (soft mode):", sj.round(x))

print("\nJAX heaviside:", jnp.heaviside(x, 0.5))
print("SoftJAX heaviside (hard mode):", sj.heaviside(x, mode="hard"))
print("SoftJAX heaviside (soft mode):", sj.heaviside(x))
```
```
JAX ReLU: [0.  0.  0.3 1. ]
SoftJAX ReLU (hard mode): [0.  0.  0.3 1. ]
SoftJAX ReLU (soft mode): [1.26928011e-02 4.53988992e-06 3.04858735e-01 1.00000454e+00]

JAX Clip: [-0.2 -0.5  0.3  0.5]
SoftJAX Clip (hard mode): [-0.2 -0.5  0.3  0.5]
SoftJAX Clip (soft mode): [-0.19523241 -0.4993285   0.28734074  0.4993285 ]

JAX Absolute: [0.2 1.  0.3 1. ]
SoftJAX Absolute (hard mode): [0.2 1.  0.3 1. ]
SoftJAX Absolute (soft mode): [0.15231883 0.9999092  0.27154448 0.9999092 ]

JAX Sign: [-1. -1.  1.  1.]
SoftJAX Sign (hard mode): [-1. -1.  1.  1.]
SoftJAX Sign (soft mode): [-0.76159416 -0.9999092   0.90514825  0.9999092 ]

JAX round: [-0. -1.  0.  1.]
SoftJAX round (hard mode): [-0. -1.  0.  1.]
SoftJAX round (soft mode): [-0.04651704 -1.          0.1188737   1.        ]

JAX heaviside: [0. 0. 1. 1.]
SoftJAX heaviside (hard mode): [0. 0. 1. 1.]
SoftJAX heaviside (soft mode): [1.19202922e-01 4.53978687e-05 9.52574127e-01 9.99954602e-01]
```

```python
# Functions on arrays
print("\nJAX max:", jnp.max(x))
print("SoftJAX max (hard mode):", sj.max(x, mode="hard"))
print("SoftJAX max (soft mode):", sj.max(x))

print("\nJAX min:", jnp.min(x))
print("SoftJAX min (hard mode):", sj.min(x, mode="hard"))
print("SoftJAX min (soft mode):", sj.min(x))

print("\nJAX median:", jnp.median(x))
print("SoftJAX median (hard mode):", sj.median(x, mode="hard"))
print("SoftJAX median (soft mode):", sj.median(x))

print("\nJAX top_k:", jax.lax.top_k(x, k=3)[0])
print("SoftJAX top_k (hard mode):", sj.top_k(x, k=3, mode="hard")[0])
print("SoftJAX top_k (soft mode):", sj.top_k(x, k=3)[0])

print("\nJAX sort:", jnp.sort(x))
print("SoftJAX sort (hard mode):", sj.sort(x, mode="hard"))
print("SoftJAX sort (soft mode):", sj.sort(x))

print("\nJAX ranking:", jnp.argsort(jnp.argsort(x)))
print("SoftJAX ranking (hard mode):", sj.ranking(x, mode="hard", descending=False))
print("SoftJAX ranking (soft mode):", sj.ranking(x, descending=False))
```
```
JAX max: 1.0
SoftJAX max (hard mode): 1.0
SoftJAX max (soft mode): 0.9993548976691374

JAX min: -1.0
SoftJAX min (hard mode): -1.0
SoftJAX min (soft mode): -0.9997287789452775

JAX median: 0.04999999999999999
SoftJAX median (hard mode): 0.04999999999999999
SoftJAX median (soft mode): 0.05000033589501627

JAX top_k: [ 1.   0.3 -0.2]
SoftJAX top_k (hard mode): [ 1.   0.3 -0.2]
SoftJAX top_k (soft mode): [ 0.9993549   0.29728716 -0.19691387]

JAX sort: [-1.  -0.2  0.3  1. ]
SoftJAX sort (hard mode): [-1.  -0.2  0.3  1. ]
SoftJAX sort (soft mode): [-0.99972878 -0.19691387  0.29728716  0.9993549 ]

JAX ranking: [1 0 2 3]
SoftJAX ranking (hard mode): [1. 0. 2. 3.]
SoftJAX ranking (soft mode): [1.00636968e+00 3.39874686e-04 1.99421369e+00 2.99907667e+00]
```

```python
# Functions returning indices
print("\nJAX argmax:", jnp.argmax(x))
print("SoftJAX argmax (hard mode):", sj.argmax(x, mode="hard"))
print("SoftJAX argmax (soft mode):", sj.argmax(x))

print("\nJAX argmin:", jnp.argmin(x))
print("SoftJAX argmin (hard mode):", sj.argmin(x, mode="hard"))
print("SoftJAX argmin (soft mode):", sj.argmin(x))

print("\nJAX argmedian:", "Not implemented in standard JAX")
print("SoftJAX argmedian (hard mode):", sj.argmedian(x, mode="hard"))
print("SoftJAX argmedian (soft mode):", sj.argmedian(x))

print("\nJAX argtop_k:", jax.lax.top_k(x, k=3)[1])
print("SoftJAX argtop_k (hard mode):", sj.top_k(x, k=3, mode="hard")[1])
print("SoftJAX argtop_k (soft mode):", sj.top_k(x, k=3)[1])

print("\nJAX argsort:", jnp.argsort(x))
print("SoftJAX argsort (hard mode):", sj.argsort(x, mode="hard"))
print("SoftJAX argsort (soft mode):", sj.argsort(x))
```
```
JAX argmax: 3
SoftJAX argmax (hard mode): [0. 0. 0. 1.]
SoftJAX argmax (soft mode): [6.13857697e-06 2.05926316e-09 9.11045600e-04 9.99082814e-01]

JAX argmin: 1
SoftJAX argmin (hard mode): [0. 1. 0. 0.]
SoftJAX argmin (soft mode): [3.35349372e-04 9.99662389e-01 2.25956629e-06 2.06045775e-09]

JAX argmedian: Not implemented in standard JAX
SoftJAX argmedian (hard mode): [0.5 0.  0.5 0. ]
SoftJAX argmedian (soft mode): [4.99999764e-01 5.62675608e-08 4.99999764e-01 4.15764163e-07]

JAX argtop_k: [3 2 0]
SoftJAX argtop_k (hard mode): [[0. 0. 0. 1.]
 [0. 0. 1. 0.]
 [1. 0. 0. 0.]]
SoftJAX argtop_k (soft mode): [[6.13857697e-06 2.05926316e-09 9.11045600e-04 9.99082814e-01]
 [6.68677917e-03 2.24316451e-06 9.92406021e-01 9.04957153e-04]
 [9.92970214e-01 3.33104397e-04 6.69058067e-03 6.10101985e-06]]

JAX argsort: [1 0 2 3]
SoftJAX argsort (hard mode): [[0. 1. 0. 0.]
 [1. 0. 0. 0.]
 [0. 0. 1. 0.]
 [0. 0. 0. 1.]]
SoftJAX argsort (soft mode): [[3.35349372e-04 9.99662389e-01 2.25956629e-06 2.06045775e-09]
 [9.92970214e-01 3.33104397e-04 6.69058067e-03 6.10101985e-06]
 [6.68677917e-03 2.24316451e-06 9.92406021e-01 9.04957153e-04]
 [6.13857697e-06 2.05926316e-09 9.11045600e-04 9.99082814e-01]]
```

```python
y = jnp.array([0.2, -0.5, 0.5, -1.0])

# SoftBool generation
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
SoftJAX greater (soft mode): [0.01798621 0.00669285 0.11920292 1.        ]

JAX greater equal: [False False False  True]
SoftJAX greater equal (hard mode): [0. 0. 0. 1.]
SoftJAX greater equal (soft mode): [0.01798621 0.00669285 0.11920292 1.        ]

JAX less: [ True  True  True False]
SoftJAX less (hard mode): [1. 1. 1. 0.]
SoftJAX less (soft mode): [9.82013790e-01 9.93307149e-01 8.80797078e-01 2.06115369e-09]

JAX less equal: [ True  True  True False]
SoftJAX less equal (hard mode): [1. 1. 1. 0.]
SoftJAX less equal (soft mode): [9.82013790e-01 9.93307149e-01 8.80797078e-01 2.06115369e-09]

JAX equal: [False False False False]
SoftJAX equal (hard mode): [0. 0. 0. 0.]
SoftJAX equal (soft mode): [1.79862100e-02 6.69285093e-03 1.19202922e-01 2.06115369e-09]

JAX not equal: [ True  True  True  True]
SoftJAX not equal (hard mode): [1. 1. 1. 1.]
SoftJAX not equal (soft mode): [0.98201379 0.99330715 0.88079708 1.        ]

JAX isclose: [False False False False]
SoftJAX isclose (hard mode): [0. 0. 0. 0.]
SoftJAX isclose (soft mode): [1.79865650e-02 6.69318401e-03 1.19208182e-01 2.06135997e-09]
```

```python
# SoftBool manipulation
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
print("SoftJAX ALL:", sj.all(fuzzy_a))

print("\nJAX ANY:", jnp.any(bool_a))
print("SoftJAX ANY:", sj.any(fuzzy_a))

# SoftBool selection
print("\nJAX Where:", jnp.where(bool_a, x, y))
print("SoftJAX Where:", sj.where(fuzzy_a, x, y))
```
```
JAX AND: [False False False  True]
SoftJAX AND: [0.26457513 0.24494897 0.28284271 0.9486833 ]

JAX OR: [ True False  True  True]
SoftJAX OR: [0.48038476 0.25166852 0.57573593 0.99999684]

JAX NOT: [ True  True False False]
SoftJAX NOT: [0.9 0.8 0.2 0. ]

JAX XOR: [ True False  True False]
SoftJAX XOR: [0.58702688 0.43498731 0.63937484 0.17309871]

JAX ALL: False
SoftJAX ALL: 0.35565588200778464

JAX ANY: True
SoftJAX ANY: 0.9980519925071494

JAX Where: [ 0.2 -0.5  0.3  1. ]
SoftJAX Where: [ 0.16 -0.6   0.34  1.  ]
```

```python
# Straight-through estimation: Use hard function on forward and soft on backward
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


## Citation

If this library helped your academic work, please consider citing:

```bibtex
@misc{Softjax2025,
  author = {Paulus, Anselm and Geist, Ren\'e and Martius, Georg},
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