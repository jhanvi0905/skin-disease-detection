# Skin Disease

## Tools used in this project
* Cookiecutter: Data And Code Repository Setup
* Poetry: Dependency management
* [pdoc](https://github.com/pdoc3/pdoc): Automatically create an API documentation

## Project structure
```bash
├── data            
│   ├── final                       # data after training the model
│   ├── processed                   # data after processing
│   ├── raw                         # raw data
├── docs                            # documentation for project
├── .gitignore                      # ignore files that cannot commit to Git
├── Makefile                        # store useful commands to set up the environment
├── models                          # store models
├── notebooks                       # store notebooks
├── .pre-commit-config.yaml         # configurations for pre-commit
├── pyproject.toml                  # dependencies for poetry
├── README.md                       # description project
├── src                             # store source code
    ├── __init__.py                 # make src a Python module 
    ├── process.py                  # process data before training model
    └── train_model.py              # train model