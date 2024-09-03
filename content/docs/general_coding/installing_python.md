---
title: "Installing Python"
---

# Installing Python

The best approach for installing Python is to install an environment manager such as [Anaconda](https://www.anaconda.com/products/individual) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html). This will allow you to create isolated environments for different projects and manage dependencies.

[Anaconda](https://www.anaconda.com/products/individual) comes with many packages pre-installed, which we don't necessarily need. Instead, it is typically better to use [Miniconda](https://docs.conda.io/en/latest/miniconda.html), which is a minimal installer for conda. This allows you to install only the packages you need, which can be useful for keeping your environment clean and avoiding conflicts between packages.

## Installing Miniconda

The quickest way to install Miniconda is via the terminal, as detailed in [this guide](https://docs.anaconda.com/miniconda/#quick-command-line-install).

In Linux:

```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
```

In MacOS:

```bash
mkdir -p ~/miniconda3
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh -o ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
```

In Windows powershell:

```bash
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe -o miniconda.exe
Start-Process -FilePath ".\miniconda.exe" -ArgumentList "/S" -Wait
del miniconda.exe
```

{{< callout context="note" title="Note" icon="outline/info-circle" >}}
If using Windows it is preferable to use [Windows Subsystem for Linux](https://docs.microsoft.com/en-us/windows/wsl/install) and install using the Linux instructions above.
{{< /callout >}}

Once installed (you may need to restart your terminal), you should be able to call `conda` from the terminal (it won't do anything without other options, but it should at least not complain that it doesn't exist).

## Creating a new environment

To create a new environment, you can use the following command:

```bash
conda create --name myenv python=3.10
```

You can replace `myenv` with the name of your environment, and `3.10` with the version of Python you want to use. You can also install other packages at the same time, for example:

### Installing packages

You can then install packages as needed using `conda install` or `pip install`. For example:

```bash
pip install numpy
```

{{< callout context="caution" title="Caution" icon="outline/alert-triangle" >}}
Anaconda recently updated their terms and conditions such that using `conda install` will technically require you to pay them vast amounts of money. It's probably best to just use `pip install` instead.
{{< /callout >}}

## Installing jupyter

If you want to run Jupyter notebooks you may wish to install [Jupyter Lab](https://jupyterlab.readthedocs.io/en/stable/). This can be done using the following command:

```bash
pip install jupyterlab
```

You can then run Jupyter Lab using the following command:

```bash
jupyter lab
```

This will open a new tab in your browser with Jupyter Lab running. You can then create a new notebook by clicking on the `+` symbol in the top left corner.

### Running notebooks in VS Code

If you prefer to use VS Code, you can also run Jupyter notebooks in VS Code. To do this, you will need to install the Python extension for VS Code, which can be found [here](https://marketplace.visualstudio.com/items?itemName=ms-python.python). You can then open a Jupyter notebook in VS Code. It should automatically detect Conda environments and allow you to choose which one to use.

If the environment doesn't seem to be present, you may need to install an ipykernel in your environment:

```bash
conda activate myenv
pip install ipykernel
python -m ipykernel install --user --name=myenv
```

You should then be able to select your environment in the top right corner of the notebook.
