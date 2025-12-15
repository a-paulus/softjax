#!/bin/bash

# 1. CREATE ENV
uv venv --python 3.12 .venv
source .venv/bin/activate

echo "Active environment: $VIRTUAL_ENV"

# 2. INSTALL JAX GPU
uv pip install --upgrade pip
uv pip install --upgrade "jax[cuda12]"
# uv pip install --upgrade "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

uv pip install -e . --no-cache-dir --group dev
