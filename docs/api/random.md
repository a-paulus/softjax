# Random sampling

SoftJAX random sampling functions live under `softjax.random` and mirror the corresponding `jax.random` APIs while returning relaxed samples. `categorical` returns a `SoftIndex` over sampled categories; `bernoulli` returns a numeric `SoftBool`.

In `mode="hard"` these functions call the corresponding JAX sampler. Soft modes reuse the same random perturbation or threshold and replace the hard decision with a soft relaxation. The `_st` variants return hard samples in the forward pass and use the soft relaxation for gradients.

!!! note "Shapes"
    `categorical` follows `jax.random.categorical` sample-shape semantics and appends the soft category axis as the last dimension. `bernoulli` follows `jax.random.bernoulli` shape broadcasting.

!!! note "Log probabilities"
    `categorical(..., return_log_probs=True)` returns log relaxed sample weights, not categorical log-likelihoods under the original logits. Use `log_prob_eps` if sparse modes need bounded log values.

::: softjax.random.categorical

::: softjax.random.categorical_st

::: softjax.random.bernoulli

::: softjax.random.bernoulli_st
