---
title: "2. Implementing an update function"
---

# Implementing an update function

First, we need to create a function that implements the core update method for our model (i.e., how it updates its value estimate in response to the outcomes it has received).


## Imports

First, we import necessary packages.



```python
import jax
import numpy as np
import jax.numpy as jnp
from typing import Tuple
```



## The Rescorla-Wagner update rule

In the standard Rescorla-Wagner model, this occurs as follows:

$$V_{t+1} = V_t + \alpha \times \delta_t$$

where $V_{t+1}$ is the value estimate at time $t+1$, $V_t$ is the value estimate at time $t$, $\alpha$ is the learning rate, and $\delta_t$ is the prediction error at time $t$.

The prediction error $$\delta_t$$ is calculated as the difference between the reward received at time $t$ and the value estimate at time $t$:

$$\delta_t = R_t - V_t$$

where $R_t$ is the reward received at time $t$.

## Introducing asymmetry

We can introduce asymmetry into the model by allowing the learning rate to be different for positive and negative prediction errors. This can be implemented as follows. Our prediction error is calculated as normal:

$$\delta_t = R_t - V_t$$

But we now choose our learning rate for the current trial $\alpha_t$ based on the sign of the prediction error:

$$\alpha_t = \alpha^+ \text{ if } \delta_t > 0 \text{ else } \alpha^-$$

where $\alpha^+$ is the learning rate for positive prediction errors, and $\alpha^-$ is the learning rate for negative prediction errors. The update rule then becomes:

$$V_{t+1} = V_t + \alpha_t \times \delta_t$$

## The update function

We can implement this in JAX as follows:

> Note: We will use the `@jit` decorator to compile the function for faster execution.



```python
@jax.jit
def asymmetric_rescorla_wagner_update(
    value: float, outcome: float, alpha_p: float, alpha_n: float
) -> Tuple[float, float]:
    """
    Updates the estimated value of a state or action using the Asymmetric Rescorla-Wagner learning rule.

    The function calculates the prediction error as the difference between the actual outcome and the current
    estimated value. It then updates the estimated value based on the prediction error and the learning rate,
    which is determined by whether the prediction error is positive or negative.

    Args:
        value (float): The current estimated value of a state or action.
        outcome (float): The actual reward received.
        alpha_p (float): The learning rate used when the prediction error is positive.
        alpha_n (float): The learning rate used when the prediction error is negative.

    Returns:
        Tuple[float, float]: The updated value and the prediction error.
    """

    # Calculate the prediction error
    prediction_error = outcome - value

    # Set the learning rate based on the sign of the prediction error
    # Remember - we can't use if else statements here because JAX doesn't tolerate them
    alpha_t = (alpha_p * (prediction_error > 0)) + (alpha_n * (prediction_error < 0))

    # Update the value
    value = value + alpha_t * prediction_error

    return value, prediction_error
```

A few things to note about this implementation:

-   **No if/else statements**: As mentioned in [this guide](/docs/computational_modelling/guide/1.-building-basic-models#other-things-to-watch-out-for-with-jax), we can't use if/else statements in JAX. Instead, we create binary variables that we can use to determine the learning rate through multiplication.
-   **Return values**: We return both the value estimate and the prediction error. The value estimate is critical for our model as this is the key quantity we're estimating. The prediction error isn't vital, but can be useful to return (e.g., we might want to plot it later or link it to neural activity).
-   **Docstring**: We use a docstring to describe the function. This is good practice as it helps others understand what the function does. I like to use [Google format](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) for docstrings.
-   **Type hints**: We use type hints to specify the types of the inputs and outputs. This is good practice as it helps others understand what the function expects and returns.


## Checking that the functon works

If all is well with our implementation, we should be able to run the function and get some output. Let's test this now.

We'll set $\alpha_p$ to a low value and $\alpha_n$ to a high value, so we can see how the value estimate changes in response to positive and negative prediction errors.



```python
# Initialize the value, outcome, and learning rates
value = 0.5
outcome = 1.0
alpha_p = 0.1
alpha_n = 0.9

# Call the function
updated_value, prediction_error = asymmetric_rescorla_wagner_update(
    value, outcome, alpha_p, alpha_n
)

# Print the results
print(f"Updated Value: {updated_value}")
print(f"Prediction Error: {prediction_error}")
```

    Updated Value: 0.55
    Prediction Error: 0.5


Here, we have a positive prediction error and we can see that the value estimate increases by a small amount, as the learning rate for positive prediction errors is low.

Let's see what happens if we have a negative prediction error.



```python
# Initialize the value, outcome, and learning rates
value = 0.5
outcome = 0
alpha_p = 0.1
alpha_n = 0.9

# Call the function
updated_value, prediction_error = asymmetric_rescorla_wagner_update(
    value, outcome, alpha_p, alpha_n
)

# Print the results
print(f"Updated Value: {updated_value}")
print(f"Prediction Error: {prediction_error}")
```

    Updated Value: 0.04999999999999999
    Prediction Error: -0.5


We can see that our estimated value has gone from 0.55 to 0.05 (there's a bit of a precision issue here, but it's close enough). This is because the learning rate for negative prediction errors is high, so the value estimate has decreased by a large amount.


### Working with arrays

Our function can also be applied to arrays. This is useful if we have multiple stimuli/actions that people are learning the value of. We can pass in an array of value estimates and rewards, and get back an array of updated value estimates and prediction errors.



```python
# Initialize the value, outcome, and learning rates
value = np.ones(5) * 0.5
outcome = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
alpha_p = 0.1
alpha_n = 0.9

# Call the function
updated_value, prediction_error = asymmetric_rescorla_wagner_update(
    value, outcome, alpha_p, alpha_n
)

# Print the results
print(f"Updated Value: {updated_value}")
print(f"Prediction Error: {prediction_error}")
```

    Updated Value: [0.55 0.05 0.55 0.05 0.55]
    Prediction Error: [ 0.5 -0.5  0.5 -0.5  0.5]


## Making our function more flexible

It's quite common in learning tasks that people have to estimate the value of multiple stimuli or actions, but only receive feedback for their chosen option on any given trial. Currently, our function will update the value estimate for all stimuli/actions on every trial, which isn't necessarily what we want to do.

We can make our function more flexible by allowing it to update only the value estimate for the chosen option. We can do this by passing in an additional argument that specifies which option was chosen on the current trial. We'll also update the type hints so that they make it clear we can pass in arrays of value estimates and rewards.

> **Note**: We use the `jax.typing.ArrayLike` type hint to specify that the input is an array-like object (e.g., a list, tuple, or JAX array). This is useful as it makes it clear that the user can pass in different types of array-like objects (e.g., lists or JAX arrays).



```python
@jax.jit
def asymmetric_rescorla_wagner_update(
    value: jax.typing.ArrayLike,
    outcome: jax.typing.ArrayLike,
    chosen: jax.typing.ArrayLike,
    alpha_p: float,
    alpha_n: float,
) -> Tuple[jax.typing.ArrayLike, jax.typing.ArrayLike]:
    """
    Updates the estimated value of a state or action using the Asymmetric Rescorla-Wagner learning rule.

    The function calculates the prediction error as the difference between the actual outcome and the current
    estimated value. It then updates the estimated value based on the prediction error and the learning rate,
    which is determined by whether the prediction error is positive or negative.

    Value estimates are only updated for chosen actions. For unchosen actions, the prediction error is set to 0.

    Args:
        value (float): The current estimated value of a state or action.
        outcome (float): The actual reward received.
        chosen (float): Binary indicator of whether the action was chosen (1) or not (0).
        alpha_p (float): The learning rate used when the prediction error is positive.
        alpha_n (float): The learning rate used when the prediction error is negative.

    Returns:
        Tuple[float, float]: The updated value and the prediction error.
    """

    # Calculate the prediction error
    prediction_error = outcome - value

    # Set prediction error to 0 for unchosen actions
    prediction_error = prediction_error * chosen

    # Set the learning rate based on the sign of the prediction error
    # Remember - we can't use if else statements here because JAX doesn't tolerate them
    alpha_t = (alpha_p * (prediction_error > 0)) + (alpha_n * (prediction_error < 0))

    # Update the value
    value = value + alpha_t * prediction_error

    return value, prediction_error
```

Here, we've incorporated information about which option was chosen by multiplying the prediction error by a binary variable that is 1 if the option was chosen and 0 otherwise. This means that the value estimate for the chosen option will be updated, while the value estimates for the other options will remain the same.



```python
# Initialize the value, outcome, choices, and learning rates
value = np.ones(5) * 0.5
outcome = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
chosen = np.array([1, 1, 0, 0, 0])
alpha_p = 0.1
alpha_n = 0.9

# Call the function
updated_value, prediction_error = asymmetric_rescorla_wagner_update(
    value, outcome, chosen, alpha_p, alpha_n
)

# Print the results
print(f"Updated Value: {updated_value}")
print(f"Prediction Error: {prediction_error}")
```

    An NVIDIA GPU may be present on this machine, but a CUDA-enabled jaxlib is not installed. Falling back to cpu.


    Updated Value: [0.55       0.05000001 0.5        0.5        0.5       ]
    Prediction Error: [ 0.5 -0.5  0.  -0.   0. ]


We can see that only the values for the chosen options have been updated, while the rest remain at 0.5.
