---
title: "Using Git and GitHub"
---

## The lab GitHub organisation

We have a lab GitHub organisation [here](https://github.com/the-wise-lab). This hosts repositories for many projects, as well as some general resources and code that may be useful for multiple projects.

To get access, please ask Toby.

## Cloning a repository from GitHub

You will first need to ensure you have Git installed on your computer. You can check if you have Git installed by running the following command in your terminal:

```bash
git --version
```

Many operating systems come with Git pre-installed, but if you don't have it, you can download it from [here](https://git-scm.com/downloads).

To clone a repository from GitHub, you can use the following command:

```bash
git clone https://github.com/the-wise-lab/repository-name.git
```

Where `repository-name` is the name of the repository you want to clone. This will create a local copy of the repository on your computer, which you can then work with.

### Private repositories

{{< callout context="caution" title="Caution" icon="outline/alert-triangle" >}}
Many repositories in the lab GitHub organisation are private. If you're a member of the lab GitHub organisation, you should have access to these repositories, but you will need to authenticate with GitHub to access them.
{{< /callout >}}

For private repositories, you will need to authenticate with GitHub. The easiest way to do this is via the GitHub CLI, which you can install from [here](https://cli.github.com/). Once you have the GitHub CLI installed, you can authenticate by running the following command:

```bash
gh auth login
```

An alternative approach is by using authentication tokens, which you can generate from your GitHub account settings. You can then use these tokens to authenticate with GitHub when prompted for a username and password. For more information on how to do this, see [this guide](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token).

## Working with branches

Often, we will create new branches to work on different features. For example, we might want to create a new version of an online task, in which case we might create a new branch called `online-task-v2`. To work with a branch, you can use the following command to switch to that branch:

```bash
git checkout online-task-v2
```

This will switch all of the code etc. over to the `online-task-v2` branch, allowing you to work on that branch without affecting the main branch (which is often called `main` or `master`). You can then make changes and commit them to that branch without affecting the main branch.

## Commiting changes

When you have made changes to the code, you can commit those changes to the branch using the following commands:

```bash
git add .
git commit -m "Your commit message here"
git push origin online-task-v2
```

The `git add .` command stages all of the changes you have made, the `git commit -m "Your commit message here"` command creates a new commit with a message describing the changes you have made, and the `git push origin online-task-v2` command pushes your changes to the remote repository on GitHub.
