## How to use this: -

- Choose use this repo as template option in GitHub.
- This should create a repo in your GitHub account.

## Files to edit to set up your project.

- You would need to edit some files in order to rename this from `template_python` to your required repo name.
- Just replace `pip install git+git://github.com/oke-aditya/template_python.git` in the `mk-docs-build.yml` and `mk-docs-deploy.yml` workflows in .github folder with your package git url. This will set up docs.
- You may need to edit the CI, installation and deployment yml files in .github folder. They need minimal editing for new project.
- Please edit **ALL** the `.md` files to include description that you need.
- Edit the `.gitingore` and `.dockerignore` files if anything extra is needed. I have included most stuff in them.
- Edit the `settings.ini` and ` setup.py` . You perhaps need your name and different requirements. Again most stuff is there you need very small tweaks.
- Edit the `requirements.txt` and `requirements-extra.txt` (optionally).
- Edit the `LICENSE` you may need a different one.
- Do edit the docs folder. It is built using [mkdocs.org](https://www.mkdocs.org). You can refer to mkdocs to know more how to edit docs. This is just minimalistic docs which does the job.
- Optionally Add tests to `tests` folder using `pytest`.

Also please read the `README` files that are present in the folders. They will help and guide you to setup stuff too.


If you have any doubts or queries regarding setting up your new project. Please raise an issue to this repo.
