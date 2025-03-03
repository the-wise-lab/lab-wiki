---
title: "General best practices"
---

## Building models

* If possible, build models using [JAX](https://jax.readthedocs.io/)
* Make sure code is modular and well-documented
  * Components that could be reused should be made into functions
* Contribute model code to [`behavioural_modelling`](https://github.com/tobywise/behavioural-modelling) if appropriate

## Fitting models

* **See [this paper](https://elifesciences.org/articles/49547) for some useful advice on fitting models**
* Model parameters should be estimated using a hierarchical Bayesian approach, with posterior samples obtained using MCMC
  * MCMC should be run using [Numpyro](http://num.pyro.ai/) (unless this isn't possible for any reason)
  * Posteriors should be checked using traceplots and other diagnostics (e.g., Rhat)
* _We are in the process of evaluating simulation-based inference (SBI) as an alternative model-fitting procedure_
* Models should be properly evaluated using parameter and model recovery checks
