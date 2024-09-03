---
title: "Wise Lab Wiki"
description: ""
lead: "Various resources and tutorials"
date: 2023-09-07T16:33:54+02:00
lastmod: 2023-09-07T16:33:54+02:00
draft: false
---

## Contributing

Anyone with access to the [lab GitHub](https://github.com/the-wise-lab) is welcome to add/edit content on this wiki.

The site is built using [Doks](https://getdoks.org/), a Hugo theme for technical documentation and content is written in [Markdown](https://www.markdownguide.org/).

To add a new page, first clone the repository:

```bash
git clone https://github.com/the-wise-lab/lab-wiki.git
cd lab-wiki
```

You can then open the repository in e.g., VS Code, and create a new file in the `content` directory. You can use the `content/_index.md` file as a template for the front matter of your new page.

{{< callout context="tip" title="Front matter" icon="outline/rocket" >}}
Front matter is the metadata at the top of each page that tells Hugo how to render the page. It is written in YAML format and is enclosed by `---`. You can copy the front matter from another page and modify it as needed.

For example:

```yaml
---
title: "My new page"
description: "A description of my new page"
---
```

{{< /callout >}}

Once you have added content to your new page, you add the new file to the repository, commit the changes, and push them to GitHub. Via the terminal, this might look like:

```bash
git add content/my_new_page.md
git commit -m "Add new page"
git push
```

The page will then be automatically built and deployed via GitHub actions, and should then be viewable on the [lab wiki](https://wiki.thewiselab.org).

See the [Doks documentation](https://getdoks.org/docs/) for more information on how to format your content.

{{< callout context="note" title="Note" icon="outline/info-circle" >}}
You can also create pages using Jupyter notebooks. To do this, create a new notebook in Jupyter Lab, and save it in the `content` directory. You can then convert the notebook to markdown using something like the following, which will enable it to be displayed on the webpage (you may need to install some packages to do this):

```bash
cd content/computational_modelling/tutorial
jupyter nbconvert --to markdown MCMC.ipynb --NbConvertApp.output_files_dir=.
```

{{< /callout >}}
