---
title: "3. Selecting actions"
---

# Selecting actions

In most learning tasks we are asking participants to _select_ different actions or stimuli based on their estimated value. This means that our model not only needs to estimate the value of different options, but also make choices.


## Imports

First, we import necessary packages.



```python
import jax
import numpy as np
import jax.numpy as jnp
from typing import Tuple
```

## The Softmax function

The softmax function is a common way to convert values into choice probabilities. It is defined as:

$$
P(a) = \frac{e^{Q(a) / \tau}}{\sum_{a'} e^{Q(a') / \tau}}
$$

where $Q(a)$ is the value of action $a$, and $\tau$ is a parameter that controls the randomness of the choices (referred to as a _temperature_ parameter). When $\tau$ is high, the softmax function will output similar probabilities for all actions, while when $\tau$ is low, the softmax function will output probabilities that are close to 0 or 1.

Essentially, the softmax function calculates the probability of a given action based on its value relative to the values of all other actions. The higher the value of an action, the higher its probability of being selected.

For the sake of simplicity and reproducibility, we'll use an existing implementation of the softmax function from the `behavioural_modelling` package.


```python
from behavioural_modelling.decision_rules import softmax

?softmax
```

    [0;31mSignature:[0m     
    [0msoftmax[0m[0;34m([0m[0;34m[0m
    [0;34m[0m    [0mvalue[0m[0;34m:[0m [0mUnion[0m[0;34m[[0m[0mnumpy[0m[0;34m.[0m[0m_typing[0m[0;34m.[0m[0m_array_like[0m[0;34m.[0m[0m_SupportsArray[0m[0;34m[[0m[0mnumpy[0m[0;34m.[0m[0mdtype[0m[0;34m[[0m[0mAny[0m[0;34m][0m[0;34m][0m[0;34m,[0m [0mnumpy[0m[0;34m.[0m[0m_typing[0m[0;34m.[0m[0m_nested_sequence[0m[0;34m.[0m[0m_NestedSequence[0m[0;34m[[0m[0mnumpy[0m[0;34m.[0m[0m_typing[0m[0;34m.[0m[0m_array_like[0m[0;34m.[0m[0m_SupportsArray[0m[0;34m[[0m[0mnumpy[0m[0;34m.[0m[0mdtype[0m[0;34m[[0m[0mAny[0m[0;34m][0m[0;34m][0m[0;34m][0m[0;34m,[0m [0mbool[0m[0;34m,[0m [0mint[0m[0;34m,[0m [0mfloat[0m[0;34m,[0m [0mcomplex[0m[0;34m,[0m [0mstr[0m[0;34m,[0m [0mbytes[0m[0;34m,[0m [0mnumpy[0m[0;34m.[0m[0m_typing[0m[0;34m.[0m[0m_nested_sequence[0m[0;34m.[0m[0m_NestedSequence[0m[0;34m[[0m[0mUnion[0m[0;34m[[0m[0mbool[0m[0;34m,[0m [0mint[0m[0;34m,[0m [0mfloat[0m[0;34m,[0m [0mcomplex[0m[0;34m,[0m [0mstr[0m[0;34m,[0m [0mbytes[0m[0;34m][0m[0;34m][0m[0;34m][0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mtemperature[0m[0;34m:[0m [0mfloat[0m [0;34m=[0m [0;36m1[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m[0;34m)[0m [0;34m->[0m [0mUnion[0m[0;34m[[0m[0mnumpy[0m[0;34m.[0m[0m_typing[0m[0;34m.[0m[0m_array_like[0m[0;34m.[0m[0m_SupportsArray[0m[0;34m[[0m[0mnumpy[0m[0;34m.[0m[0mdtype[0m[0;34m[[0m[0mAny[0m[0;34m][0m[0;34m][0m[0;34m,[0m [0mnumpy[0m[0;34m.[0m[0m_typing[0m[0;34m.[0m[0m_nested_sequence[0m[0;34m.[0m[0m_NestedSequence[0m[0;34m[[0m[0mnumpy[0m[0;34m.[0m[0m_typing[0m[0;34m.[0m[0m_array_like[0m[0;34m.[0m[0m_SupportsArray[0m[0;34m[[0m[0mnumpy[0m[0;34m.[0m[0mdtype[0m[0;34m[[0m[0mAny[0m[0;34m][0m[0;34m][0m[0;34m][0m[0;34m,[0m [0mbool[0m[0;34m,[0m [0mint[0m[0;34m,[0m [0mfloat[0m[0;34m,[0m [0mcomplex[0m[0;34m,[0m [0mstr[0m[0;34m,[0m [0mbytes[0m[0;34m,[0m [0mnumpy[0m[0;34m.[0m[0m_typing[0m[0;34m.[0m[0m_nested_sequence[0m[0;34m.[0m[0m_NestedSequence[0m[0;34m[[0m[0mUnion[0m[0;34m[[0m[0mbool[0m[0;34m,[0m [0mint[0m[0;34m,[0m [0mfloat[0m[0;34m,[0m [0mcomplex[0m[0;34m,[0m [0mstr[0m[0;34m,[0m [0mbytes[0m[0;34m][0m[0;34m][0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
    [0;31mCall signature:[0m [0msoftmax[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
    [0;31mType:[0m           PjitFunction
    [0;31mString form:[0m    <PjitFunction of <function softmax at 0x7f16777c3be0>>
    [0;31mFile:[0m           ~/miniconda3/envs/transition_uncertainty/lib/python3.10/site-packages/behavioural_modelling/decision_rules.py
    [0;31mDocstring:[0m     
    Softmax function, with optional temperature parameter.
    
    Args:
        value (ArrayLike): Array of values to apply softmax to, of shape (n_trials, n_bandits)
        temperature (float, optional): Softmax temperature, in range 0 > inf. Defaults to 1.
    
    Returns:
        ArrayLike: Choice probabilities, of shape (n_trials, n_bandits)

### Demonstrating the softmax function

To demonstrate, we can provide a set of action values and calculate the probabilities of selecting each action according to different temperature parameter values.

> **NOTE**: The function expects our values to be 2-dimensional, as we'll often want to apply it to a a set of values for multiple stimuli across multiple trials.


```python
# Initialize the values
values = jnp.array([[2.0, 3.0, 1.0]])

# Example temperature parameter values
temperature = [0.1, 0.5, 0.9]

# Compute the softmax probabilities using each temperature parameter
for t in temperature:
    print(f"Temperature: {t}")
    print(np.round(softmax(values, t), 2))
```

    Temperature: 0.1
    [[0. 1. 0.]]
    Temperature: 0.5
    [[0.12 0.87 0.02]]
    Temperature: 0.9
    [[0.22999999 0.7        0.08      ]]


## Choosing an action

We also want to actually choose an action based on these estimated probabilities. Again, we'll use an existing function for this.


```python
from behavioural_modelling.utils import choice_from_action_p

?choice_from_action_p
```

    [0;31mSignature:[0m     
    [0mchoice_from_action_p[0m[0;34m([0m[0;34m[0m
    [0;34m[0m    [0mkey[0m[0;34m:[0m [0;34m<[0m[0mfunction[0m [0mPRNGKey[0m [0mat[0m [0;36m0x7f160e170700[0m[0;34m>[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mprobs[0m[0;34m:[0m [0mUnion[0m[0;34m[[0m[0mnumpy[0m[0;34m.[0m[0m_typing[0m[0;34m.[0m[0m_array_like[0m[0;34m.[0m[0m_SupportsArray[0m[0;34m[[0m[0mnumpy[0m[0;34m.[0m[0mdtype[0m[0;34m[[0m[0mAny[0m[0;34m][0m[0;34m][0m[0;34m,[0m [0mnumpy[0m[0;34m.[0m[0m_typing[0m[0;34m.[0m[0m_nested_sequence[0m[0;34m.[0m[0m_NestedSequence[0m[0;34m[[0m[0mnumpy[0m[0;34m.[0m[0m_typing[0m[0;34m.[0m[0m_array_like[0m[0;34m.[0m[0m_SupportsArray[0m[0;34m[[0m[0mnumpy[0m[0;34m.[0m[0mdtype[0m[0;34m[[0m[0mAny[0m[0;34m][0m[0;34m][0m[0;34m][0m[0;34m,[0m [0mbool[0m[0;34m,[0m [0mint[0m[0;34m,[0m [0mfloat[0m[0;34m,[0m [0mcomplex[0m[0;34m,[0m [0mstr[0m[0;34m,[0m [0mbytes[0m[0;34m,[0m [0mnumpy[0m[0;34m.[0m[0m_typing[0m[0;34m.[0m[0m_nested_sequence[0m[0;34m.[0m[0m_NestedSequence[0m[0;34m[[0m[0mUnion[0m[0;34m[[0m[0mbool[0m[0;34m,[0m [0mint[0m[0;34m,[0m [0mfloat[0m[0;34m,[0m [0mcomplex[0m[0;34m,[0m [0mstr[0m[0;34m,[0m [0mbytes[0m[0;34m][0m[0;34m][0m[0;34m][0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mlapse[0m[0;34m:[0m [0mfloat[0m [0;34m=[0m [0;36m0.0[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m[0;34m)[0m [0;34m->[0m [0mint[0m[0;34m[0m[0;34m[0m[0m
    [0;31mCall signature:[0m [0mchoice_from_action_p[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
    [0;31mType:[0m           PjitFunction
    [0;31mString form:[0m    <PjitFunction of <function choice_from_action_p at 0x7f15fc730310>>
    [0;31mFile:[0m           ~/miniconda3/envs/transition_uncertainty/lib/python3.10/site-packages/behavioural_modelling/utils.py
    [0;31mDocstring:[0m     
    Choose an action from a set of action probabilities. Can take probabilities
    in the form of an n-dimensional array, where the last dimension is the
    number of actions.
    
    Noise is added to the choice, with probability `lapse`. This means that
    on "lapse" trials, the subject will choose an action uniformly at random.
    
    Args:
        key (int): Jax random key
        probs (np.ndarray): N-dimension array of action probabilities, of shape (..., n_actions)
        lapse (float, optional): Probability of lapse. Defaults to 0.0.
    Returns:
        int: Chosen action

### Incorporating randomness

We want to make sure our choices are not deterministic: if we have an action probability of 0.75 this means we'll only want to choose this action 75% of the time. JAX is a little complex when it comes to randomness, and you need to supply a random "key" every time you want to generate random numbers. This means that when using this function for choosing actions, you'll need to pass in a key as well.

Because we're supplying a random key, the function will **always** return the same action when is given the same key. This is useful for reproducibility, but it also means that you'll need to pass in a new key every time you want to generate a new action.


```python
# Get a random key
key = jax.random.PRNGKey(0)

# Choose an action using the softmax probabilities
choice_from_action_p(key, softmax(values, t))
```




    Array([1], dtype=int32)



## Incorporating the softmax function into our model

As it stands, we've implemented a model that can estimate the value of different actions. However, we haven't yet implemented a way to select actions based on these values. We can do this by incorporating the softmax function into our model.

In order to keep our code as modular as possible, we will create a new function (`asymmetric_rescorla_wagner_update_choice`) that will use our existing update function to estimate the value of different actions, and then use the softmax function to select an action based on these values, rather than integrating this functionality directly into our existing update function.


```python
# THIS IS OUR EXISTING UPDATE FUNCTION
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
    alpha_t = (alpha_p * (prediction_error > 0)) + (alpha_n * (prediction_error < 0))

    # Update the value
    value = value + alpha_t * prediction_error

    return value, prediction_error

# THIS IS OUR NEW CHOICE FUNCTION
@jax.jit
def asymmetric_rescorla_wagner_update_choice(
    value: jax.typing.ArrayLike,
    outcome: jax.typing.ArrayLike,
    alpha_p: float,
    alpha_n: float,
    temperature: float,
    n_actions: int,
    key: jax.random.PRNGKey,
) -> np.ndarray:
    """
    Updates the value estimate using the asymmetric Rescorla-Wagner algorithm, and chooses an
    option based on the softmax function.

    Args:
        value (jax.typing.ArrayLike): The current value estimate.
        outcome (jax.typing.ArrayLike): The outcome of the action.
        alpha_p (float): The learning rate for positive outcomes.
        alpha_n (float): The learning rate for negative outcomes.
        temperature (float): The temperature parameter for softmax function.
        n_actions (int): The number of actions to choose from.
        key (jax.random.PRNGKey): The random key for the choice function.

    Returns:
        Tuple[np.ndarray, Tuple[jax.typing.ArrayLike, np.ndarray, int, np.ndarray]]:
            - updated_value (jnp.ndarray): The updated value estimate.
            - output_tuple (Tuple[jax.typing.ArrayLike, np.ndarray, int, np.ndarray]):
                - value (jax.typing.ArrayLike): The original value estimate.
                - choice_p (jnp.ndarray): The choice probabilities.
                - choice (int): The chosen action.
                - choice_array (jnp.ndarray): The chosen action in one-hot format.
    """

    # Get choice probabilities
    choice_p = softmax(value[None, :], temperature).squeeze()

    # Get choice
    choice = choice_from_action_p(key, choice_p)

    # Convert it to one-hot format
    choice_array = jnp.zeros(n_actions, dtype=jnp.int16)
    choice_array = choice_array.at[choice].set(1)

    # Get the outcome and update the value estimate
    updated_value, prediction_error = asymmetric_rescorla_wagner_update(
        value,
        choice_array,
        outcome,
        alpha_p,
        alpha_n,
    )

    return updated_value, (value, choice_p, choice_array, prediction_error)
```

There is quite a lot going on here, so let's break it down.

### 1. Inputs to the function

```python
def asymmetric_rescorla_wagner_update_choice(
    value: jax.typing.ArrayLike,
    outcome: jax.typing.ArrayLike,
    chosen: jax.typing.ArrayLike,
    alpha_p: float,
    alpha_n: float,
    temperature: float,
    n_actions: int,
    key: jax.random.PRNGKey,
) -> np.ndarray:
```
As with our previous function, we provide the current value and the outcome received. Note that we don't need to provide the chosen option as we're generating this from scratch. We also provide the learning rates for positive and negative prediction errors and the temperature parameter for the softmax function.

We also need to provide the number of possible actions. This is because we need to generate a one-hot array of the chosen action, and we need to know how long this array should be. **Why can we not just infer this from the length of the value array using e.g., `value.shape`?** This is because JAX needs to know the size of the array at compile time, and the size of the value array is not known until runtime. Otherwise, we will get an error when we try to compile the function.

Finally, we need to provide a random key. This is because we're using JAX's random number generator to generate a random choice, and we need to provide a key to do this.


### 2. Getting choice probabilities from values

```python
choice_p = softmax(value[None, :], temperature).squeeze()
```

As we mentioned earlier, the softmax function calculates the probability of selecting each action based on its value. We pass in our estimated values for each action, and the temperature parameter, and get back a set of probabilities for selecting each action.

By default, the `softmax` function expects a 2-dimensional array of values, where the first dimension corresponds to the number of trials and the second dimension corresponds to the number of actions. However, our `value` array is 1-dimensional as it corresponds to the values or the current trial. We can use the `None` index to add an extra dimension to our array, and then `squeeze` to remove it again.

> ⚠️ **Note**: It is important that we get choice probabilities and select actions **BEFORE** updating the value. When someone makes a choice on **Trial 1**, they are doing this without having received any information - their choice is based on their current expectation. Only after they have made a choice do they receive feedback, which is then used to update their expectation for the next trial.

### 3. Choosing an action

```python
choice = choice_from_action_p(key, choice_p)
```

As we mentioned earlier, we need to pass in a random key in order to generate a random choice. We use the `choice_from_action_p` function to generate a random choice based on the probabilities we calculated using the softmax function.

### 4. Converting to one-hot format

```python
choice_array = jnp.zeros(n_actions, dtype=jnp.int16)
choice_array = choice_array.at[choice].set(1)
```

The `choice_from_action_p` function returns the index of the chosen action. We convert this index into a one-hot array, where all values are 0 except for the chosen action, which is 1. This is the format that our update function expects.

### 5. Updating the estimated value

```python
updated_value = rescorla_wagner_update(
    value,
    choice_array,
    outcomes,
    alpha_p,
    alpha_n,
)
```

We now update our estimated value based on the chosen action and the outcome of that action. We use the `rescorla_wagner_update` function that we defined earlier to do this.

### 6. Returning useful variables

```python
return updated_value, (value, choice_p, choice_array, prediction_error)
```

Finally, we return the updated value, as well as some other useful variables that we might want to keep track of, such as the choice probabilities, the one-hot array of the chosen action, and the prediction error.

This might seem a little odd: **why do we return everything but `updated_value` as a tuple?** We could instead do something like:

```python
return updated_value, choice_p, choice_array, prediction_error
```

However, we will need to use this function within a `jax.lax.scan` loop, and `jax.lax.scan` expects the function to return only two values. The first value is what is fed back into the function at the next time step, and the second value is what is collected at each time step. The only variable that's going to be reused at the next time step is `updated_value`, so we return this as the first value, and everything else as the second value.

There's something else here that's a bit confusing: **why do we return `value` as well as `updated_value`?** We don't actually need to return `value` here, as we're already returning `updated_value`. However, we might want to keep track of the value at each time step **before** it has been updated (e.g., perhaps we want to link expected value on a given trial to neural activity, in which case we want the value before it has been updated).


## Trying out the function

If we try to run our function as it's currently written, we will get an error:


```python
# Initialize the value, outcome, choices, and learning rates
value = np.ones(5) * 0.5
outcome = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
alpha_p = 0.1
alpha_n = 0.9
temperature = 0.5

# Get a random key
key = jax.random.PRNGKey(0)

# Call the function
updated_value, (value, choice_p, choice_array, prediction_error) = asymmetric_rescorla_wagner_update_choice(
    value, outcome, alpha_p, alpha_n, temperature, 5, key
)

# Print the results
print(f"Updated Value: {updated_value}")
print(f"Prediction Error: {prediction_error}")
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    Cell In[46], line 12
          9 key = jax.random.PRNGKey(0)
         11 # Call the function
    ---> 12 updated_value, (value, choice_p, choice_array, prediction_error) = asymmetric_rescorla_wagner_update_choice(
         13     value, outcome, alpha_p, alpha_n, temperature, 5, key
         14 )
         16 # Print the results
         17 print(f"Updated Value: {updated_value}")


        [... skipping hidden 12 frame]


    Cell In[43], line 85, in asymmetric_rescorla_wagner_update_choice(value, outcome, alpha_p, alpha_n, temperature, n_actions, key)
         82 choice = choice_from_action_p(key, choice_p)
         84 # Convert it to one-hot format
    ---> 85 choice_array = jnp.zeros(n_actions, dtype=jnp.int16)
         86 choice_array = choice_array.at[choice].set(1)
         88 # Get the outcome and update the value estimate


    File ~/miniconda3/envs/transition_uncertainty/lib/python3.10/site-packages/jax/_src/numpy/lax_numpy.py:2288, in zeros(shape, dtype)
       2286 if (m := _check_forgot_shape_tuple("zeros", shape, dtype)): raise TypeError(m)
       2287 dtypes.check_user_dtype_supported(dtype, "zeros")
    -> 2288 shape = canonicalize_shape(shape)
       2289 return lax.full(shape, 0, _jnp_dtype(dtype))


    File ~/miniconda3/envs/transition_uncertainty/lib/python3.10/site-packages/jax/_src/numpy/lax_numpy.py:80, in canonicalize_shape(shape, context)
         77 def canonicalize_shape(shape: Any, context: str="") -> core.Shape:
         78   if (not isinstance(shape, (tuple, list)) and
         79       (getattr(shape, 'ndim', None) == 0 or ndim(shape) == 0)):
    ---> 80     return core.canonicalize_shape((shape,), context)  # type: ignore
         81   else:
         82     return core.canonicalize_shape(shape, context)


    File ~/miniconda3/envs/transition_uncertainty/lib/python3.10/site-packages/jax/_src/core.py:2130, in canonicalize_shape(shape, context)
       2128 except TypeError:
       2129   pass
    -> 2130 raise _invalid_shape_error(shape, context)


    TypeError: Shapes must be 1D sequences of concrete values of integer type, got (Traced<ShapedArray(int32[], weak_type=True)>with<DynamicJaxprTrace(level=1/0)>,).
    If using `jit`, try using `static_argnums` or applying `jit` to smaller subfunctions.
    The error occurred while tracing the function asymmetric_rescorla_wagner_update_choice at /tmp/ipykernel_38248/3994267310.py:45 for jit. This concrete value was not available in Python because it depends on the value of the argument n_actions.


There are various clues in the error message as to what's gone wrong:

` Shapes must be 1D sequences of concrete values of integer type, got (Traced<ShapedArray(int32[], weak_type=True)>with<DynamicJaxprTrace(level=1/0)>,).`

`If using jit, try using static_argnums`

`This concrete value was not available in Python because it depends on the value of the argument n_actions.`

This is a common problem when using JAX - we need to provide the size of our arrays at compile time, but the size of our arrays is not known until runtime. 

### Using `static_argnums`

Essentially, the problem is that JAX needs to know the shape of everything in advance - this is partly why it can make our code run so quickly. However, in this case, the size of the `choice_array` variable depends upon the `n_actions` variable, which is not known until runtime. We've supplied `5` here when calling the function, but all JAX sees is an integer that could take any value. 

The easiest solution is to tell JAX to compile the function so that it works _only_ with the value that we've passed in. So we'll get a compiled function that works for `n_actions=5`, but if we try to use it with `n_actions=10`, it will fail. This would mean that we need to recomplie the function every time we want to use it with a different number of actions, but in reality that isn't something we're likely to do.

We can do this using the `static_argnums` argument to `jax.jit`. This tells JAX that the function should be compiled with respect to the arguments that we specify. In this case, we want to compile the function with respect to the `n_actions` argument, so we'll pass in `5` (the index of the `n_actions` argument in the function signature).

> **NOTE**: We need to call `jax.jit()` as a function rather than using it as a decorator if we want to pass in `static_argnums`.


```python
def asymmetric_rescorla_wagner_update_choice(
    value: jax.typing.ArrayLike,
    outcome: jax.typing.ArrayLike,
    alpha_p: float,
    alpha_n: float,
    temperature: float,
    n_actions: int,
    key: jax.random.PRNGKey,
) -> np.ndarray:
    """
    Updates the value estimate using the asymmetric Rescorla-Wagner algorithm, and chooses an
    option based on the softmax function.

    Args:
        value (jax.typing.ArrayLike): The current value estimate.
        outcome (jax.typing.ArrayLike): The outcome of the action.
        alpha_p (float): The learning rate for positive outcomes.
        alpha_n (float): The learning rate for negative outcomes.
        temperature (float): The temperature parameter for softmax function.
        n_actions (int): The number of actions to choose from.
        key (jax.random.PRNGKey): The random key for the choice function.

    Returns:
        Tuple[np.ndarray, Tuple[jax.typing.ArrayLike, np.ndarray, int, np.ndarray]]:
            - updated_value (jnp.ndarray): The updated value estimate.
            - output_tuple (Tuple[jax.typing.ArrayLike, np.ndarray, int, np.ndarray]):
                - value (jax.typing.ArrayLike): The original value estimate.
                - choice_p (jnp.ndarray): The choice probabilities.
                - choice (int): The chosen action.
                - choice_array (jnp.ndarray): The chosen action in one-hot format.
    """

    # Get choice probabilities
    choice_p = softmax(value[None, :], temperature).squeeze()

    # Get choice
    choice = choice_from_action_p(key, choice_p)

    # Convert it to one-hot format
    choice_array = jnp.zeros(n_actions, dtype=jnp.int16)
    choice_array = choice_array.at[choice].set(1)

    # Get the outcome and update the value estimate
    updated_value, prediction_error = asymmetric_rescorla_wagner_update(
        value,
        choice_array,
        outcome,
        alpha_p,
        alpha_n,
    )

    return updated_value, (value, choice_p, choice_array, prediction_error)

asymmetric_rescorla_wagner_update_choice = jax.jit(asymmetric_rescorla_wagner_update_choice, static_argnums=(5,))
```

Now we can try running it again...


```python
# Initialize the value, outcome, choices, and learning rates
value = np.ones(5) * 0.5
outcome = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
alpha_p = 0.1
alpha_n = 0.9
temperature = 0.5

# Get a random key
key = jax.random.PRNGKey(0)

# Call the function
updated_value, (value, choice_p, choice_array, prediction_error) = asymmetric_rescorla_wagner_update_choice(
    value, outcome, alpha_p, alpha_n, temperature, 5, key
)

# Print the results
print(f"Updated Value: {updated_value}")
print(f"Choice probabilities: {choice_p}")
print(f"Choice: {choice_array}")
```

    Updated Value: [0.05000001 0.5        0.55       0.5        0.05000001]
    Choice probabilities: [0.2 0.2 0.2 0.2 0.2]
    Choice: [0 0 1 0 0]


We can see that the function has chosen action number `2` (0-indexed), and this is the only action that has had its value updated.
