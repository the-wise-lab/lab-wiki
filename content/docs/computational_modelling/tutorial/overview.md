---
title: "1. Overview"
---

This tutorial will take you through the implementation of a commonly-used reinforcement learning model, the Rescorla-Wagner model. We will implement an asymmetric version of this model that allows for differential updating of value estimates in response to positive and negative prediction errors (see [here](https://www.nature.com/articles/s41562-017-0067)).

We will implement this using [JAX](https://jax.readthedocs.io/en/latest/). In all honesty, **JAX is not the most user-friendly library for beginners**, but it is incredibly powerful and efficient. This means that there are various quirks and tricks that need to be used, but once you get the hang of it the process becomes quite easy.
