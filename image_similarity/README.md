# How the code is structured ?

- `torch_data.py` contains dataset class that can be used for loading data.
- `torch_model.py` contains models which we need to train.
- `torch_engine.py` contains training and validation steps. It also contains code to create embeddings.
- `utils.py` contains utility functions.
- `torch_train.py` contains training script. It makes use of `torch_data.py`, `torch_model.py`, `torch_engine.py` and `utils.py`.
- `torch_infer.py` contains code for inference.

- The `.ipynb` notebooks are provided as standalone scripts that can be used for training and inference.

This folder structure allows ease of programming, modularization and easy re-use.


