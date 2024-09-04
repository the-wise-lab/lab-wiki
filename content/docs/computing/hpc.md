---
title: "High performance computing"
---

# High performance computing

While many things can be run on a laptop/desktop, you will typically be able to speed things up substantially by taking advantage of high performance computing (HPC) resources. This is especially the case if you have a task that can be easily parallelised. However, versus a laptop, HPC resources are typically faster even for single-threaded tasks.

## Accessing HPC resources at KCL

See the guidance [here](https://docs.er.kcl.ac.uk/CREATE/requesting_access/) for accessing HPC resources at KCL. The cluster is called CREATE, and is generally quite easy to use and very powerful.

{{< callout context="note" title="Note" icon="outline/info-circle" >}}
If you're a student you'll need to ask Toby to request access on behalf.
{{< /callout >}}

You will have three areas to store files:

`/users/<username>`: This is your home directory. It is not intended for large files, and is limited in storage capacity (50GB).

`/scratch/users/<username>`: This limited to 200GB and is not backed up. It is useful for working with larger files.

`/scratch/prj/bcn_neudec`: This is for general lab use and everyone should have access to it (if you don't, ask Toby). We have 1TB storage, but as above this isn't backed up.

## Running jobs on HPC

Our cluster uses [SLURM](https://slurm.schedmd.com/quickstart.html) for job scheduling. Jobs can roughly be divided into two types:

### Interactive jobs

These are useful for testing code, debugging, etc. Essentially, the job logs you into one of the compute nodes and you can use it interactively. To run an interactive job, I typically use something like:

```bash
srun -p cpu --pty -t 5:00:00 --mem=20GB --ntasks=4 --nodes=1 --cpus-per-task=1 /bin/bash
```

This will get me an interactive job for 5 hours (increase if you need more), with 4 CPU cores and 20GB of memory. You can then run your code as you would on your local machine. Technically, you can ask for as many CPU cores and memory as are available on a single node (multi-node jobs are more complicated), which is ~128 CPU cores and ~1TB of memory, but you should rarely need anything like this.

### Batch jobs

These are jobs that you submit to the queue and they run when resources are available. To submit a batch job, you need to create a script that contains the commands you want to run. For example, you might have a script called `my_script.sh` that looks something like this:

```bash
#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --output=my_job.out
#SBATCH --error=my_job.err
#SBATCH --time=1:00:00
#SBATCH --mem=4G

python my_script.py
```

Which you can then submit to the queue using:

```bash
sbatch my_script.sh
```

See the [CREATE documentation](https://docs.er.kcl.ac.uk/CREATE/running_jobs/) for more information about submitting batch jobs.

## Running Jupyter notebooks on HPC

I run most of my analyses in Jupyter notebooks, so it can be useful to run these on HPC. Thankfully, this is relatively straightforward to do via SSH tunnelling - this will allow you to access Python kernels on the HPC through VS code or the browser.

First, you'll need to [set up Python](../general_coding/installing_python/) on the cluster and install Jupyter Lab. Can can then start an interactive job as above, and run Jupyter Lab without the browser and on a specific port:

```bash
jupyter lab --no-browser --port=9997 --ip="*"
```

This will mean that we can connect to this Jupyter process from other locations via port 9997.

Next, you'll need to set up an SSH tunnel from your local machine to the HPC. This can be done using the following command in the terminal (this is for Linux/Mac - it may be different on Windows):

```bash
ssh -L 9997:<node>:9997 <username>@hpc.create.kcl.ac.uk
```

This will connect you to the HPC and set up a tunnel from your local machine to the HPC on port 9997. You can then open a browser and go to `localhost:9997` to access the Jupyter Lab running on the HPC, as you would if it were running locally.

You can also access this Jupyter Lab instance through VS Code:

1. In a notebook, click "Select Kernel" in the top right.
2. In the menu that opens, click "Select another kernel...", then "Existing Jupyter Server".
3. Clck "Enter the URL of the running Jupyter Server..." and then enter the URL of the Jupyter server (e.g., `https://localhost:9997`), and press enter.
4. You will be asked for a password (you will have set this up at some point when setting up Jupyter lab).
5. You should now be connected and able to select remote kernels from within VS Code.
