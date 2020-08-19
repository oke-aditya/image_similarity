# Template Repository Docs

This is docs built for template repository using [mkdocs.org](https://www.mkdocs.org).

## Using Mkdocs

* `mkdocs new [dir-name]` - Create a new project.
* `mkdocs serve` - Start the live-reloading docs server.
* `mkdocs build` - Build the documentation site.
* `mkdocs -h` - Print help message and exit.

## Project layout

- This repository has a prebuilt CI in the `.github` folder.
- It checks for docs building. [Black](https://black.readthedocs.io/en/stable) code formatting. 
- Also it runs the tests written in Pytest.
- By defualt this public repo has all the CI tests passing.
- Docs folder has this documentation as made by mkdocs.
- Test folder contains all tests you would write with pytest.
- template_ml folder contains the source code which you would write.
- It has empty `.py` files just to show how you could structure your code.
- `Dockerfile` and `.dockerignore` files help you building your container. Some starter code is provided.
- `Setup.py` and `settings.ini` help in Publishing this repo to PyPi.

## About the template
- This template is open source under super permissive MIT License. Feel free to edit and re use as you like.
- Do give a * if you use it and share it with others.