# Template Repo for python Projects.

![CI Tests](https://github.com/oke-aditya/template_python/workflows/CI%20Tests/badge.svg)
![Check Formatting](https://github.com/oke-aditya/template_python/workflows/Check%20Formatting/badge.svg)
![Build mkdocs](https://github.com/oke-aditya/template_python/workflows/Build%20mkdocs/badge.svg)
![Deploy mkdocs](https://github.com/oke-aditya/template_python/workflows/Deploy%20mkdocs/badge.svg)
![PyPi Release](https://github.com/oke-aditya/template_python/workflows/PyPi%20Release/badge.svg)
![Install Package](https://github.com/oke-aditya/template_python/workflows/Install%20Package/badge.svg)

A template repository to make all machine learning projects.

## How this speeds up your python development

Most people find it hard to package their python code and do not know how to set up the repo for it.


If the repository is setup in a wrong way, it would become hard to package and deploy the code later on as well.


This repo gives you the batteries required to package your code, CI checks, auto build and deploy docs,
easy PyPi publishing support and docker files.


This serves as a template to quickly have these things setup in your repo.
Machine Learning Repos created from this template can easily be deployed and shipped. It becomes hassle free and easy to debug too.

You can add your code in `template_python` folder. Since this is a package make sure that imports are
from the root. i.e. `from template_python import stuff`


## How to use this: -

- Choose use this repo as template option in GitHub.
- This should create a repo in your GitHub account.

## Files to edit to set up your project.

- You would need to edit some files in order to rename this from `template_python` to your required repo name.
- Just replace `pip install git+git://github.com/oke-aditya/template_python.git` in the `mk-docs-build.yml` and `mk-docs-deploy.yml` workflows in .github folder with your package git url. This will set up docs.
- Please edit **ALL** the `.md` files to include description that you need.
- Edit the `requirements.txt` and `requirements-extra.txt` (optionally).
- Edit the `.gitingore` and `.dockerignore` files if anything extra is needed. I have included most stuff in them.
- Edit the `settings.ini` and ` setup.py` (optionally) . You perhaps need your name and different requirements. Again most stuff is there you need very small tweaks.
- Edit the `LICENSE` you may need a different one.
- Do edit the docs folder. It is built using [mkdocs.org](https://www.mkdocs.org). You can refer to mkdocs to know more how to edit docs. This is just minimalistic docs which does the job.
- Optionally Add tests to `tests` folder using `pytest`.

Also please read the `README` files that are present in the folders. They will help and guide you to setup stuff too.

### I need help to setup my project
- If you face any issues setting up project do raise an issue. I would be happy to help.
- If you have few sugguestions and additions you would like please raise a PR. I would be happy to merge.

### Projects built using this template.
**Note: -** These repos might have diverted little bit from this template.

- [PyTorch CNN trainer](https://github.com/oke-aditya/pytorch_cnn_trainer.git).
- [Fashion Intel](https://github.com/oke-aditya/fashion_intel)

Raise a PR if you have built your project with this template and I will add it here !!

### Inspirations: -
This template was created using lot of repositories , it include these
- [fastai nbdev](https://github.com/fastai/nbdev_template)
-  Made With ML [boilerplate](https://github.com/madewithml/boilerplate).
- [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning)
- [mantisshrimp](https://github.com/airctic/mantisshrimp)

Huge credit to these repos, it would be hard to make this without them.

This template is diffrent from above as this lays emphasis on bundling code in python packages and containers thus ensuring portability.


