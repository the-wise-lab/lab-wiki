---
title: "3. MCMC sampling"
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

The previous sections outline how to generate simulated data using a computational model. In this section, we will discuss how to use Markov Chain Monte Carlo (MCMC) sampling to fit a model to data.

## What is MCMC sampling?

MCMC sampling is a method for estimating the parameters of a model by sampling from the posterior distribution of the parameters. This means we can estimate a full probability distribution for each parameter, rather than just a point estimate. As we're thinking about probability distributions, these models are often referred to as "Bayesian" or "probabilistic" models.

MCMC sampling is the "gold standard" for fitting computational models as it is accurate and provides a full representation of the uncertainty in the parameter estimates. However, it can be slow and complicated (the fact that a [whole paper](https://psycnet.apa.org/record/2023-57852-001) has been written on debugging these models is a testament to this).

## MCMC sampling using NumPyro

In this section, we will use the [NumPyro](https://num.pyro.ai/) library to perform MCMC sampling. NumPyro is a probabilistic programming library that is built on top of JAX. Because it is built on JAX, NumPyro is able to take advantage of JAX's ability to compile code to run fast on different hardware (e.g., GPUs). As a result, it is able to perform MCMC sampling quickly and efficiently.

### Creating a NumPyro model

Terminology can get confusing within the world of computational model fitting, as we tend to use the word "model" to refer to different things. Generally, we'll talk about two different types of models:

* **Behavioural model**: This is the model that generates the behavioural data. This is the model we've been discussing in the previous sections.
* **Statistical model**: This is the model that describes the relationship between the parameters of the behavioural model and the data. This is the model we're going to discuss in this section.

In NumPyro, we define the statistical model using a Python function. So, for our Rescorla-Wagner model, we might define a statistical model like this:

```python
def model(outcomes, observations, n_participants):
    # Define the priors for the learning rate
    alpha = numpyro.sample("alpha", dist.Uniform(0, 1), sample_shape=(n_participants,))

    # Define the priors for the softmax temperature parameter
    beta = numpyro.sample("beta", dist.Uniform(0, 1), sample_shape=(n_participants,))

    # Get values from our model
    values = rescorla_wagner_trial_iterator_vmap(0.5, outcomes, alpha)

    # Apply a softmax function to the values
    choice_probabilities = softmax(values, beta)

    # Define the likelihood
    numpyro.sample("obs", dist.Binomial(1, choice_probabilities, obs=observations))
```

To break this down:

#### Priors

```python
# Define the priors for the learning rate
alpha = numpyro.sample("alpha", dist.Uniform(0, 1), sample_shape=(n_participants,))

# Define the priors for the softmax temperature parameter
beta = numpyro.sample("beta", dist.Uniform(0, 1), sample_shape=(n_participants,))
```
> **Note**: We've included a softmax function here to transalte the values into choice probabilities. This is a common step in models that generate choices.

We define the priors for the learning rate and softmax temperature parameter. These are the parameters we want to estimate. Here we've used a uniform distribution, but you can use any appropriate distribution. Given that these values must lie between 0 and 1, you could also use a Beta distribution.

For example:

```python
alpha = numpyro.sample("alpha", dist.Beta(2, 2), sample_shape=(n_participants,))
```

The `sample_shape` argument allows us to sample multiple values for each participant. This is useful when we want to estimate the parameters for multiple participants.

#### Behavioural model

Next, we generate data from our behavioural model using the parameters we've sampled:

```python
# Get values from our model
values = rescorla_wagner_trial_iterator_vmap(0.5, outcomes, alpha)
```

This does exactly what we've described in the previous sections: it generates values from the Rescorla-Wagner model for each participant.

We also use a softmax function to convert these values into choice probabilities:

```python
# Apply a softmax function to the values
choice_probabilities = softmax(values, beta)
```

#### Likelihood

Finally, we define the likelihood of the data given the model:

```python
numpyro.sample("obs", dist.Binomial(1, choice_probabilities, obs=observations))
```

This is a binomial likelihood, as we're assuming that the data are binary (i.e., the participant either chose option A or option B). The `obs` argument specifies the observed data, which in this case is the choices made by the participant.

This is ultimately the metric used to fit the model to the data. The MCMC algorithm will sample values for `alpha` and `beta` that maximise the likelihood of the data given the model.

### Running MCMC sampling

Once we've defined our model, we can run MCMC sampling using NumPyro. This is done using the `numpyro.infer.MCMC` class:

```python
# Set up the MCMC sampler
mcmc = MCMC(NUTS(model), num_samples=4000, num_warmup=1000, num_chains=4)

# Get a random key
rng_key = jax.random.PRNGKey(seed)

# Get number of participants
n_participants = observations.shape[0]

# Run the MCMC sampling
mcmc.run(rng_key, outcomes, observations, n_participants)
```

> **Note**: We assume here that `observations` is a NumPy array containing the observed data and `outcomes` is a NumPy array containing the task outcomes.

This will run the MCMC sampling algorithm and store the samples in the `mcmc` object. We can then use the samples to estimate the posterior distribution of the parameters.

For example, we can use the `mcmc.get_samples()` method to get the samples from the MCMC run:

```python
samples = mcmc.get_samples()
```

#### Settings for MCMC sampling

In general, the default settings for MCMC sampling in NumPyro are pretty good. However, there are some things you may want to adjust.

* `num_samples`: The number of samples to draw from the posterior distribution. More samples will give you a more accurate estimate of the posterior distribution, but will take longer to run. In general, something like 4000 samples is a good starting point, but more is better if feasible.
* `num_warmup`: The number of warmup samples to draw before starting to draw samples from the posterior distribution. Warmup samples are used to "tune" the sampler and are not included in the final samples. In general, something like 1000 warmup samples is a good starting point.
* `num_chains`: The number of chains to run. More chains will give you a more accurate estimate of the posterior distribution, but will take longer to run. I would normally use 4 chains.

