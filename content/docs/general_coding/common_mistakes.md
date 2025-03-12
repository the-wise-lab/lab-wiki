---
title: "Common coding mistakes"
---

# Common coding mistakes

When you first start coding, often you will do things in a way that work, but which aren't best practice. This is fine, and it's a natural part of learning. However, it's important to learn from these mistakes and improve your coding style. This page lists some common mistakes that people tend to make, and how to avoid them.

{{< callout context="note" title="Note" icon="outline/info-circle" >}}
These aren't necessary things that will mean your code doesn't run, or produces incorrect results. Instead, these are things that will make your code harder to read, harder to maintain, or harder to debug.
{{< /callout >}}

## 1. Using full directory paths

When you're writing code, it can be tempting to use full directory paths to access files. For example, you might write something like this:

```python
with open("/Users/toby/Projects/my_project/data.csv", "r") as f:
    data = f.read()
```

This is fine, but it makes your code less portable. If you move your code to a different machine, or share it with someone else, the path will likely be different. Instead, you should use relative paths. For example, you could write something like this:

```python
with open("data.csv", "r") as f:
    data = f.read()
```

You can also include checks in your code to make sure that the file exists. For example:

```python
import os
os.path.exists("data.csv")
```

This will return `True` if the file exists, and `False` if it doesn't.

Something I often do is perform a check to make sure code is being run in the correct directory. For example, we might be running a Jupyter notebook that should be run from the root directory of the project, and will expect to find data in the `data` directory. If it's run from another place, it won't work. We can add a check to make sure we're in the right place:

```python
import os
assert os.path.exists("data"), "Data directory not found. Make sure you're running this code from the root directory of the project."
```

Or even automatically change the directory. For example, we might have a `notebooks` directory where the notebook is located, and it might default to running from there.

```python
if not os.path.exists("data") and os.path.exists("../data"):
    os.chdir("..")
    print("Changed directory to root of project.")
```

## 2. Not using Python packages effectively

Often when we're writing code, we'll have code in various `.py` files that we might want to use elsewhere (e.g., Jupyter notebooks). One way in which to ensure this code is accessible is to add the directory to the `path` variable. For example:

```python
import sys
sys.path.append("/Users/toby/Projects/my_project/code")
```

However, this is not the best way to do this. Instead, Python packages provide a convenient way to package up code so that it can be used in other places, and they also have other benefits such as enabling automatic installation of dependencies. You can find a great introduction to the use of packages in research code in [this tutorial](https://goodresearch.dev/setup#install-a-project-package), but here's a very brief guide.

First, make sure you have a directory in your project folder that contains the code you want to use. For example:

```bash
├── my_project
│   ├── project_code
│   ├── data
│   ├── etc..
```

In this case, we have a directory called `project_code` that contains all the code for the project.

In order for Python to know that this is a package, we need to add an `__init__.py` file to the directory. This can be empty, but it needs to be there. So your structure should look like:

```bash
├── my_project
│   ├── project_code
│   │   ├── __init__.py
```

All of your `.py` files can then be placed wthin this directory (or subdirectories):

```bash
├── my_project
│   ├── project_code
│   │   ├── __init__.py
│   │   ├── my_module.py
│   │   ├── my_other_module.py
```

There are then a couple of ways to turn this into a useable Python package.

### The old way, using `setup.py`

The old way to do this is to create a `setup.py` file in the root of the project that looks something like this:

```python
from setuptools import setup, find_packages

setup(
    name="project_code",
    version="0.1",
    packages=find_packages(),
)
```

This will tell Python that the `project_code` package is located in the current directory, and that it contains all the packages in the `project_code` directory. You can then install the package using:

This is being phased out, and at some point in the future will now longer work.

### The new way, using `pyproject.toml`

The new way to do this is to create a `pyproject.toml` file in the root of the project that looks something like this:

```toml
[build-system]
requires = ["setuptools", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "project_code"
version = "0.1.0"

[tool.setuptools.packages.find]
where = ["."]
include = ["project_code*"]
```

This achieves the same thing (i.e., allowing your code to be installed as a package), but will be the preferred way to do this going forward.

### Installing your package

Whichever way you choose to do this, you can then install your package using:

```bash
pip install -e .
```

The `-e` flag tells Python to install the package in "editable" mode, which means that any changes you make to the code will be reflected in the package without needing to reinstall it. This means that as you update your code, you can use it in other places without needing to reinstall the package.

You can then import your code (e.g., in notebooks) very straightforwardly:

```python
from project_code.my_module import my_function
```

## 3. Poorly organised function definitions

When you're writing code, it can be tempting to define functions wherever they're needed. For example, you might write something like this within a cell of a Jupyter notebook.

```python
def my_function(x):
    return x + 1

x = 5
y = my_function(x)
```

This can be fine for functions that are small and used in only this place. However, for larger functions, it can be helpful to have them defined in a separate file. This makes your code easier to read, and easier to maintain. Related to point **(2)**, you can define functions in a Python package and import them as needed.

This has three main benefits:

1. **Reusability**: Functions can be reused across multiple projects
2. **Readability**: Code is easier to find, read, and understand, since it is located in an obvious place rather than buried within a notebook or larger script.
3. **Testability**: We can write tests for functions to ensure they work as expected, which isn't possible if they're defined within a notebook or larger script.

In general, it's best to use notebooks to **run** code and **display** outputs, but not to define significant amounts of code. A notebook should ideally be used as a high-level demonstration of the code, with the code itself stored in a Python package. There will of course sometimes be times where a bit more code is necessary or appropriate within a notebook, but this should be kept to a minimum (N.B. if you happen to be a BSc student reading this, yes I know we ask you to do everything in notebooks - it makes life easier and minimises the number of files we have to check and mark!).

## 4. Messy code

When you're writing code, it can be tempting to write code that works, but isn't very tidy. You might have very long lines that make it difficult to read, for example. This can make it hard for others to read your code, and hard for you to read it when you come back to it later. It's important to write code that is easy to read and understand.

Thankfully, it is **very** easy to format your code automatically to tidy it up. There are multiple auto-formatters out there (e.g., `black`, `autopep8`, `yapf`), but I would recommend using `black` as it is very simple to use and is very widely used. You can install it using:

```bash
pip install black
```

You can then run it on your code using:

```bash
black my_code.py
```

Alternatively, you can install the (`black` extension for VS Code)[https://marketplace.visualstudio.com/items?itemName=ms-python.black-formatter]. You then get an easy way to format your code using `right-click -> Format Document`.

As an example, say you have the following code, which is untidy, has long lines, and has inconsistent spacing:

```python
def hello_world   ( ):
    print(  "Hello, world!"  )
    if  True:
            print("This line is indented inconsistently"   )



def add_numbers( the_first_variable,the_second_variable,    another_variable, another_variable_again, **kwargs):
    if the_first_variable + the_second_variable - another_variable * another_variable_again > another_variable / another_variable_again:
            return the_first_variable + the_second_variable - another_variable * another_variable_again
    else:
        return another_variable / another_variable_again
```

You can run `black` on this code to tidy it up:

```python
def hello_world():
    print("Hello, world!")
    if True:
        print("This line is indented inconsistently")


def add_numbers(
    the_first_variable,
    the_second_variable,
    another_variable,
    another_variable_again,
    **kwargs
):
    if (
        the_first_variable
        + the_second_variable
        - another_variable * another_variable_again
        > another_variable / another_variable_again
    ):
        return (
            the_first_variable
            + the_second_variable
            - another_variable * another_variable_again
        )
    else:
        return another_variable / another_variable_again
```

There really isn't any reason not to use this, since it's so easy to do and makes your code so much easier to read.

### Using `black` in Jupyter notebooks

You can also use `black` in Jupyter notebooks. The VS Code extension will work for this, but otherwise a great package to use is `nbqa`, which performs formatting alongside many other useful tasks. You can install it using:

```bash
pip install nbqa
```

And use it on a notebook using:

```bash
nbqa black my_notebook.ipynb
```

## 5. Messy imports

Related to the above, it's important to keep your imports tidy. This makes it easier to see what packages are being used, and makes it easier to find the relevant imports. It's also important to remove any unused imports, as this can cause annoying errors down the line (e.g., you might run into problems installing a package that you don't actually need).

### Fixing import issues

Thankfully there are tools to help with this. For example, you can use `isort` to automatically sort your imports. You can install it using:

```bash
pip install isort
```

And then run it on your code using:

```bash
isort my_code.py
```

This will order your imports in a standard way.

Alternatively, there are various VS Code extensions that can do this for you.

You can also use `autoflake` to remove unused imports. You can install it using:

```bash
pip install autoflake
```

And run it using:

```bash
autoflake my_code.py
```

This will give you an output that looks like this, showing what has been removed:

```bash
--- my_code.py
+++ my_code.py
@@ -1,8 +1,6 @@
 import argparse
-import csv
 import json
 import os
-import sys
```

Or if you just want to check for unused imports (alongside other issues), you can install `flake8`:

```bash
pip install flake8
```

And run it using:

```bash
flake8 my_code.py
```

This will give you an output that looks like this:

```bash
my_code.py:2:1: F401 'csv' imported but unused
my_code.py:5:1: F401 'sys' imported but unused
```

### Fixing import issues in Jupyter notebooks

You can use `nbqa` (mentioned above) to use `isort` and `autoflake` on Jupyter notebooks. For example, you can use:

```bash
nbqa isort my_notebook.ipynb
```

And:

```bash
nbqa autoflake my_notebook.ipynb
```

## 6. Using global variables extensively

When you're writing code, it can be tempting to use global variables extensively. Global variables are variables that are defined outside any functions/classes etc., but then used within functions etc. For example, you might write something like this:

```python
x = 5

def my_function():
    return x + 1
```

This is fine in certain cases, but often can cause problems (e.g., you try to reuse a function somewhere else where the global variable isn't defined). Instead, it's best to pass variables as arguments to functions and ensure they don't rely on any global variables. For example:

```python
def my_function(x):
    return x + 1
```

## 6. Leaving uncommented code

When you're writing code, it can be tempting to leave code that you don't need commented out - perhaps you might need it later? However, this can make your code harder to read and understand. Someone else reading your code might not know why the code is commented out, or whether it's still needed. It's best to remove any code that isn't being used. If you need it later, you can always get it from version control.

If you do decide to leave code commented out, it's best to add a comment explaining why it's there. For example:

```python
# It might be necessary to use epsilon in this function call
# We need to evaluate this further before deciding
# output = my_function(variable, epsilon=0.1)
output = my_function(variable)
```

I forget to do this all the time, but it's a good habit to get into.

## 7. Unclear variable names

When you're writing code, it can be tempting to use short variable names to save time. For example, you might write something like this:

```python
p_a_1 = 5
p_b_1 = 10
```

This can make life extremely difficult for other people (or your future self) to understand what is going on. For some reason it used to be commonplace to use unintelligible variable names (if you ever look at the Matlab code underpinning SPM you will see what I mean...) but there's no really need to.

Instead, try to be as explicit as possible (without taking up too much space), e.g.:

```python
parameter_alpha_model_1 = 5
parameter_beta_model_1 = 10
```

### Python conventions for variable names

There are some standard conventions for naming variables etc. in code. These are:

1. **snake_case**: This is where you use all lowercase letters, with underscores between words. For example, `my_variable_name`.
2. **CamelCase**: This is where you use capital letters at the start of each word (except the first). For example, `myVariableName`.
3. **PascalCase**: This is where you use capital letters at the start of each word (including the first). For example, `MyVariableName`.
4. **All caps**: This is where you use all capital letters. For example, `MY_VARIABLE_NAME`.

The convention in Python is to use `snake_case` for variable names, and `CamelCase` for class names. All caps can be used for constants (e.g., `PI = 3.14159`). This is not a hard and fast rule, but it's a good convention to follow as it makes it clearer what each variable corresponds to.

So, for example:

```python
# Standard variable (snake_case)
my_variable_name = 5

# Function (snake_case)
def my_function(my_argument_name):
    return my_argument_name + 1

# Class (CamelCase)
class MyClass:
    def __init__(self, my_argument_name):
        self.my_argument_name = my_argument_name

```

### Javascript conventions for variable names

Javascript uses `camelCase` for variable names, and `PascalCase` for class names. So, for example:

```javascript
// Standard variable (camelCase)
let myVariableName = 5;

// Function (camelCase)
function myFunction(myArgumentName) {
    return myArgumentName + 1;
}

// Class (PascalCase)
class MyClass {
    constructor(myArgumentName) {
        this.myArgumentName = myArgumentName;
    }
}
```

## 8. Unclear file names

### Naming and organising

When you're developing code, it can be tempting to use vague file names for simplicity. This is fine if you're just testing things out yourself, but it can get confuding if you receive some code from someone else that looks like:

```bash
├── my_project
│   ├── code
│   │   ├── test.py
        ├── test2.py
        ├── test3.py
```

Ideally, all files should have a relatively clear name that indicates what they do. For example:

```bash
├── my_project
│   ├── code
│   │   ├── data_processing.py
        ├── model_training.py
        ├── model_evaluation.py
```

You might also want to use directory structure to help with this. For example:

```bash
├── my_project
│   ├── code
│   │   ├── data
│   │   │   ├── data_processing.py
        ├── models
        │   ├── model_training.py
        │   ├── model_evaluation.py
```

### Different versions of code

If you have multiple versions of your code, this should ideally be tracked using Git. It's best not to have e.g., `functions.py`, `functions2.py` as it won't always be clear to the person viewing the code (which might be you in the future) what the difference is between the two files.

Usually, if you're creating multiple versions of the same code this is because you're changing something but might not be sure whether you want to keep what you've changed. As with many things, this isn't especially problematic if it's just something small and you're the only person working with the code. However, it's best to use branches within Git to manage this, as it will make it much easier to see what changes have been made and why.

This is actually fairly easy to do. All you need to do is create a new branch, which can be done through the command line or through any Git client. For example, you can create a new branch called `new_feature` using:

```bash
git checkout -b new_feature
```

You can then make changes to the code, and commit them to this branch. If you decide you don't want to keep the changes, you can simply delete the branch. If you do want to keep the changes, you can merge the branch back into the main branch.

This allows you to keep your changes organised, and allows you to easily see what changes have been made and why. This can also be shared with others, which makes collaboration on code far more straightforward. You can also revert back to an earlier version without any trouble.

## 9. Not including docstrings

Docstrings are the part within a function that tells you what the function does. For example:

```python
def my_function(x: int):
    """
    This function adds 1 to the input x.

    Args:
        x (int): The input to the function.

    Returns:
        int: The input x plus 1.
    """
    return x + 1
```

Ideally the docstring should include an outline of the function's purpose. It should also include a description of the arguments it takes as input and any returns.

This is important for a number of reasons:

1. It helps other people understand what your function does.
2. It helps future you understand waht your function does.
3. Writing docstrings forces you to think more carefully about how your function is constructed.
There are different standardised formats for docstrings, but the most common is probably the Google format. You can find more information on this [here](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).

Thankfully, tools like Copilot and ChatGPT are very good at creating docstrings (although you should always double check what they've given you), which makes this process much easier.

