---
title: "1. Building basic models"
description: "This page outlines how to build learning models using JAX"
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

This guide will take you through the process of building trial-by-trial learning models. It assumes some knowledge of how these models work, and is more focused on the implementation.

## JAX

After a lot of experimentation with different frameworks, I have settled on [JAX](https://github.com/google/jax) as my go-to library for building models. JAX enables you to write high-performance numerical code in Python that runs on CPUs, GPUs, and TPUs. It does this by compiling Python functions to XLA (Accelerated Linear Algebra) operations, which can then be executed on the hardware of your choice. This gives us a lot of flexibility with model-fitting, as we can easily switch between different hardware accelerators without changing our code.

The other major benefit of JAX is that the code is virtually identical to NumPy, which makes it very easy to pick up if you are already familiar with NumPy. This is in contrast to other libraries, which have their own APIs (or even languages in the case of Stan) that can take some time to learn.

## Architecture of a model

Here I'll break down how we put together a model in JAX.

### The update rule

I generally try to make the modelling code as modular as possible. This means we have a single function that defines the **update rule** for every trial (i.e., how much the estimated value changes on a trial, given a set of inputs).

So, for example, if we have a simple Rescorla-Wagner model, we might define a function like this:

```python
def rescorla_wagner_update(value, reward, alpha):
  """Update the value estimate using the Rescorla-Wagner rule."""
    return value + alpha * (reward - value)
```

> **Note:** The code in this guide is simplified for clarity and has a lot of things missing.

This function takes the current `value` estimate, the `reward` received on the trial, and the learning rate `alpha` as inputs.

### Looping over trials

In ordinary python code, we would loop over trials using a `for` loops, such as:

```python
# Initialise a numpy array of values, assuming 100 trials
values = np.ones(100) * 0.5

# Generate some trial outcomes
outcomes = np.random.binomial(1, 0.5, 100)

# Set the learning rate
alpha = 0.1

# Loop over trials, up to the final trial (we don't need to update the value on the final trial)
for trial, outcome in enumerate(outcomes[:-1]):
  values[trial + 1] = rescorla_wagner_update(values[trial], outcome, alpha)
```

However, this will **NOT** work when using JAX. Instead, we need to use a function called jax.lax.scan, which allows us to loop over trials in a way that can be compiled by JAX. This ultimately makes the code run much faster, so it's worth the extra effort.

The `scan` function can be a little confusing. Essentially, it is designed to apply a function to a sequence of inputs, and accumulate the results. So, in our case, we want to apply the `rescorla_wagner_update` function to a sequence of outcomes, and accumulate the values.

The documentation on `scan` is quite good, so I recommend reading it [here](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scan.html) if you're not sure how it works.

Here's how we would implement the Rescorla-Wagner model using `scan`:

```python
# Initial value estimate
initial_value = 0.5

# Generate some trial outcomes
outcomes = np.random.binomial(1, 0.5, 100)

# Set the learning rate
alpha = 0.1

# Use partial to "bake in" the learning rate
rescorla_wagner_update_partial = partial(rescorla_wagner_update, alpha=alpha)

# Loop over trials
_, values = jax.lax.scan(
  rescorla_wagner_update,  # The function we want to apply
  initial_value  # The starting value
  outcomes  # The outcomes on each trial
)
```

> ❓**What does `partial` do?**
>
> The `partial` function is a way of "baking in" some of the arguments to a function.
>
> In this case, we are baking in the `alpha` argument to the `rescorla_wagner_update` function, as we can't pass in extra arguments to the function when using `scan`.
>
> This means that when we call `rescorla_wagner_update_partial`, we only need to pass in the `value` and `reward` arguments, and the `alpha` argument is already set to 0.1.

This might seem a bit complicated, but it gives us the same output as the traditional `for` loop but much faster.

#### Returning multiple outputs

Sometimes we might want to keep track of other things during the model fitting process, such as the prediction error on each trial. We can do this by modifying the `rescorla_wagner_update` function to return multiple outputs, and then unpacking them in the `scan` function.

Here's an example:

```python
def rescorla_wagner_update(value, reward, alpha):
  """Update the value estimate using the Rescorla-Wagner rule."""
  prediction_error = reward - value
  new_value = value + alpha * prediction_error
  return new_value, prediction_error

# Loop over trials
_, values, prediction_errors = jax.lax.scan(
  rescorla_wagner_update,  # The function we want to apply
  initial_value,  # The starting value
  outcomes  # The outcomes on each trial
)
```

## Other things to watch out for with JAX

There are a few other areas where JAX does things a little differently. Helpfully, there's a whole page on this in the JAX documentation, which I recommend reading [here](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html).

### If/else statements

One thing to watch out for is that JAX doesn't support if statements in the same way of Python. So, for example in standard Python we might implement an asymmetric learning rate like this:

```python
def rescorla_wagner_update(value, reward, alpha_p, alpha_n):
  """Update the value estimate using the Rescorla-Wagner rule."""
  prediction_error = reward - value
  if prediction_error > 0:
    new_value = value + alpha_p * prediction_error
  else:
    new_value = value + alpha_n * prediction_error
  return new_value, prediction_error
```

We can't do this in JAX. While JAX does have a `jax.lax.cond` function that allows you to do conditional branching, it's often easier to rewrite the function to avoid the need for if statements. So, in this case, we might rewrite the function like this:

```python
def rescorla_wagner_update(value, reward, alpha_p, alpha_n):
  """Update the value estimate using the Rescorla-Wagner rule."""
  prediction_error = reward - value
  new_value = value + (alpha_p * prediction_error * (prediction_error > 0) +
                       alpha_n * prediction_error * (prediction_error <= 0))
  return new_value, prediction_error
```

Here, we create a binary variable that is 1 if the prediction error is positive and 0 otherwise. We then multiply the learning rate by this binary variable, which has the effect of setting the learning rate to `alpha_p` if the prediction error is positive and `alpha_n` otherwise.

### JIT compiling

JAX has a `jit` function that allows you to compile a function for faster execution (JIT stands for "just in time"). If you don't do this, the function will run in "interpreted" mode, which is much slower. You can use the `jit` function like this:

```python
@jax.jit
def rescorla_wagner_update(value, reward, alpha):
  """Update the value estimate using the Rescorla-Wagner rule."""
  prediction_error = reward - value
  new_value = value + alpha * prediction_error
  return new_value, prediction_error
```

Or:

```python
rescorla_wagner_update = jax.jit(rescorla_wagner_update)
```

### Dynamic shapes

One of the reasons why JAX-compiled code is fast is because it is _inflexible_. Ordinarily, I could write a function like this:

```python
# Define trial outcomes for 3 actions
outcomes = np.random.binomial(1, 0.5, (100, 3))

def rescorla_wagner_iterator(value, outcomes, alpha):
  """Update the value estimate using the Rescorla-Wagner rule over a series of trials."""

  # Get the number of actions
  n_actions = outcomes.shape[1]

  # Initialise value estimates for each action
  values = np.ones(n_actions) * 0.5

  # Loop over trials, up to the final trial (we don't need to update the value on the final trial)
  for trial in range(outcomes.shape[0] - 1):
    values[trial + 1] = rescorla_wagner_update(values[trial], outcomes[trial, :], alpha)

# Set the learning rate
alpha = 0.1

# Run the iterator
values = rescorla_wagner_iterator(0.5, outcomes, alpha)
```

This function will determine the shape of its `values` array based on the number of actions present in the `outcomes` array. So if I changed the `outcomes` array to have 4 actions, the `values` array would automatically adjust to have 4 elements.

This isn't possible with JAX as it requires _static shapes_. This means that the shape of the arrays must be known at compile time. This can be a bit of a pain, but it's a trade-off for the speed that JAX provides. Flexibility slows our code down.

One way around this is to use the `static_argnums` argument in the `jit` function, which allows you to specify which arguments have static shapes. This means that JAX will compile the function with the shapes that it is given. If you try to call the function with a different shape, you'll get an error.

```python
# Define trial outcomes for 3 actions
outcomes = np.random.binomial(1, 0.5, (100, 3))

def rescorla_wagner_iterator(value, outcomes, alpha):
  """Update the value estimate using the Rescorla-Wagner rule over a series of trials."""

  # Get the number of actions
  n_actions = outcomes.shape[1]

  # Initialise value estimates for each action
  values = np.ones(n_actions) * 0.5

  # Loop over trials, up to the final trial (we don't need to update the value on the final trial)
  for trial in range(outcomes.shape[0] - 1):
    values[trial + 1] = rescorla_wagner_update(values[trial], outcomes[trial, :], alpha)

# JIT compile
rescorla_wagner_iterator = jax.jit(rescorla_wagner_iterator, static_argnums=(1,))

# Set the learning rate
alpha = 0.1

# Run the iterator
values = rescorla_wagner_iterator(0.5, outcomes, alpha)
```

This will compile the function so that it expects the `outcomes` array to have 3 actions. If you try to call the function with an array that has a different number of actions, you'll get an error. Otherwise, it will work as expected.
