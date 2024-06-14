---
title: "2. Modelling multiple participants"
description: "This page outlines how to use models with multiple participants"
summary: ""
date: 2023-09-07T16:04:48+02:00
lastmod: 2023-09-07T16:04:48+02:00
draft: false
weight: 810
toc: true
seo:
  title: "" # custom title (optional)
  description: "" # custom description (recommended)
  canonical: "" # custom canonical URL (optional)
  noindex: false # false (default) or true
---

In the previous section, we discussed how to model a single participant. In this section, we will discuss how to model multiple participants.

## Applying our model across multiple participants

### Nested loops

When we model multiple participants, we can use the same model code for each participant. In standard Python, we might do this using nested `for` loops.

For example:

```python
# Number of participants
n_participants = 10

# Initialise a numpy array of values, assuming 100 trials
values = np.ones((n_participants, 100)) * 0.5

# Generate some trial outcomes - these are the same for every participant
outcomes = np.random.binomial(1, 0.5, 100)

# Set learning rates for each participant
alphas = np.random.uniform(0.1, 0.5, n_participants)

# Loop over participants
for participant in range(n_participants):
  # Loop over trials
  for trial, outcome in enumerate(outcomes):
    values[participant, trial + 1] = rescorla_wagner_update(values[participant, trial], outcome, alphas[participant])
```

### Vectorisation

This code is straightforward, but it can be slow for large numbers of participants or trials. We can speed this up through vectorisation, which is a technique that allows us to perform operations on entire arrays at once.

For example:

```python
# Number of participants
n_participants = 10

# Initialise a numpy array of values, assuming 100 trials
values = np.ones((n_participants, 100)) * 0.5

# Generate some trial outcomes - these are the same for every participant
outcomes = np.random.binomial(1, 0.5, 100)

# Set learning rates for each participant
alphas = np.random.uniform(0.1, 0.5, n_participants)

# Loop over trials
for trial, outcome in enumerate(outcomes):
  values[:, trial + 1] = rescorla_wagner_update(values[:, trial], outcome, alphas)
```

This code is much faster than the nested loop version, as it takes advantage of the underlying NumPy operations to perform the update across all participants in a single step.

### Vectorisation in JAX

Vectorisation is fairly straightforward in the above example, but can be more complicated in many situations. JAX provides a straightforward way to vectorise code using the `vmap` function. The `vmap` function allows us to apply a function to a batch of inputs, and returns a batch of outputs. Because it compiles the code using XLA, it can be much faster than using standard Python loops or NumPy operations.

To facilitate this, we can create a function that runs our update rule function for a single participant over a bunch of trials, which we can later vectorise across participants.

For example:

```python
def rescorla_wagner_trial_iterator(value, outcomes, alpha):
  """Update the value estimate using the Rescorla-Wagner rule over a series of trials."""
  # Use partial to "bake in" the learning rate
  rescorla_wagner_update_partial = partial(rescorla_wagner_update, alpha=alpha)

  # Loop over trials
  _, values = jax.lax.scan(
      rescorla_wagner_update,  # The function we want to apply
      initial_value  # The starting value
      outcomes  # The outcomes on each trial
  )

  return values
```

We can then use the `vmap` function to create a vectorised version of this function that applies the update rule to each participant.

```python
# Number of participants
n_participants = 10

# Generate some trial outcomes - these are the same for every participant
outcomes = np.random.binomial(1, 0.5, 100)

# Set learning rates for each participant
alphas = np.random.uniform(0.1, 0.5, n_participants)

# Vectorise the function across participants
rescorla_wagner_trial_iterator_vmap = jax.vmap(rescorla_wagner_trial_iterator, in_axes=(None, None, 0))

# Use this function
values = rescorla_wagner_trial_iterator_vmap(0.5, outcomes, alphas)
```
