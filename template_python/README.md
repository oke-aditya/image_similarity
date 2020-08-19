# How to structure the code ?

- Usually it is good practice to divide code into data, model, engine, train files.
- If you have more than one model / API create another folder and place python files there.

- It is important to include `__init__.py` folder otherwise the package will not be able to import
functions / classes.

- In the `__init__.py` folder import stuff as you need. E.g. `from template_ml.ml_src.app import *`

- Use imports relative to `template_ml` do not use relative imports. This is a better practice.

