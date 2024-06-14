---
title: "5. Visualising MCMC results"
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

The best way to visualise and check the results when using sampling is to use the [ArviZ](https://arviz-devs.github.io/arviz/) library. This library is designed to work with a range of sampling packages, including NumPyro. It provides a range of functions for visualising the results of sampling, including trace plots, density plots, and posterior predictive checks.

I won't go into detail about how to use ArviZ here, as the [ArviZ documentation](https://arviz-devs.github.io/arviz/) is very good and covers everything you could need.
