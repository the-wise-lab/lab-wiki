---
title: "4. Hierarchical models"
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

The statistical model described in the previous section assumes that participants are independent of one another. In reality, participants are drawn from a group that will often have shared characteristics. Hierarchical models are a way to account for this by allowing parameters to vary between participants and groups of participants. This can be particularly useful when the number of observations per participant is small, as it allows information to be shared between participants, boosting the precision of the estimates.

A lot has been written about hierarchical Bayesian modelling (e.g. [here](https://www.pymc.io/projects/examples/en/latest/generalized_linear_models/multilevel_modeling.html)). Here, we will focus on how to implement hierarchical models in the context of cognitive modelling.

## Implementing a hierarchical model in NumPyro

In NumPyro, we can implement a hierarchical model by defining a model within a model. This is similar to how we defined a model within a function in the previous section. The key difference is that we can now define a model within a model, allowing us to specify how parameters vary between participants.

In general, it is best to use what is referred to as a "non-centred" parameterisation. This means that we assume that we have an average parameter value across our sample:

\[
\mu_{group} = \text{Normal}(\mu, \sigma)
\]

This says that the group mean is normally distributed with a mean of \(\mu\) and a standard deviation of \(\sigma\).

We can define this in NumPyro as:

```python
mu_group = numpyro.sample("mu_group", dist.Normal(0, 1))
```

Here, we set the prior for the group mean to be a normal distribution with a mean of 0 and a standard deviation of 1. This will depend on the specific model you are fitting and the ranges of its parameters.

We also assume that the participants are distributed around this group mean, and on average their distribution follows a standard deviation parameter \(\sigma_{group}\):

\[
\sigma_{group} = \text{HalfNormal}(\sigma)
\]

This says that the group standard deviation is half-normal distributed with a standard deviation of \(\sigma\).

We can define this in NumPyro as:

```python
sigma_group = numpyro.sample("sigma_group", dist.HalfNormal(1))
```

Finally, we define the individual participants' parameter values. We first assume that each participant's parameter value is _offset_ from the group mean by some amount:

\[
offset_{participant} = \text{Normal}(mu, sigma)
\]

This says that the offset for each participant is normally distributed with a mean of \(\mu\) and a standard deviation of \(\sigma\).

We can define this in NumPyro as:

```python
offset_participant = numpyro.sample("offset_participant", dist.Normal(mu_group, sigma_group), sample_shape=(n_participants,)
```

Finally, we combine this offset parameter with the group mean and standard deviation. Each participant's parameter value is assumed to be offset from the group mean by a multiple of the standard deviation.

\[
\theta_{participant} = \mu_{group} + offset_{participant} \times \sigma_{group}
\]

We can define this in NumPyro as:

```python
theta_participant = mu_group + offset_participant * sigma_group
```

So, together, for a given parameter we might have:

```python
group_mean = numpyro.sample("group_mean", dist.Normal(0, 1))
group_std = numpyro.sample("group_std", dist.HalfNormal(1))
offset = numpyro.sample("offset", dist.Normal(group_mean, group_std), sample_shape=(n_participants,))
theta = group_mean + offset * group_std
```

Both `group_mean` and `group_std` are shared across all participants, while `offset` is specific to each participant. As a result, `group_mean` and `group_std` are scalars (i.e., they are a single value), while `offset` and `theta` are vectors (i.e., they have a value for each participant).

### Making things more efficient

This can end up producing a lot of code, especially if you have many parameters. To make things more efficient, you can define a function that generates the hierarchical model for a single parameter, and then use this function to generate the model for each parameter.

For example:

```python
def create_subject_params(
    name: str, n_subs: int
) -> Union[dist.Normal, dist.HalfNormal, dist.Normal]:
    """
    Creates group mean, group sd and subject-level offset parameters.
    Args:
        name (str): Name of the parameter
        n_subs (int): Number of subjects
    Returns:
        Union[dist.Normal, dist.HalfNormal, dist.Normal]: Group mean, group sd, and subject-level offset parameters
    """

    group_mean = numpyro.sample("{0}_group_mean".format(name), dist.Normal(0, 1))
    group_sd = numpyro.sample("{0}_group_sd".format(name), dist.HalfNormal(1))
    offset = numpyro.sample(
        "{0}_offset".format(name), dist.Normal(0, 1), sample_shape=(n_subs,)
    )

    return group_mean, group_sd, offset
```
