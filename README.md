# Skin Disease

## Tools used in this project
* Cookiecutter: Data And Code Repository Setup
* [pdoc](https://github.com/pdoc3/pdoc): Automatically create an API documentation

## Project structure
```bash
├── Classification_Report.png       # Results on given dataset
├── data            
│   ├── final                       # data after training the model
│   ├── raw                         # raw data
├── docs                            # documentation for project
├── .gitignore                      # ignore files that cannot commit to Git
├── Makefile                        # store useful commands to set up the environment
├── models                          # store models
├── notebooks                       # store notebooks         # confi
├── .pre-commit-config.yaml         # configurations for pre-commit
├── pyproject.toml                  # dependencies for poetry
├── README.md                       # description of project structure, run the project
├── requirements.txt                # dependency environment.
├── src                             # store source code
    ├── __init__.py                 # make src a Python module 
    ├── process.py                  # process data before training model
    └── train_model.py              # train model
    └── make_prediction.py          # Make Prediction on given test set
├── summary.md                      # summary of approach, limitations, improvements   
    
```

## 1. Setting Up Environment
python3 -m pip install requirements.txt

## 2. Train the Data  
python3 -m src.train_model -bs batch_size -lr learning rate -sp split_ratio -ep epochs -path data_path

1. The defaults for all flags are set according to the given model and project setup, in case of difference modify the flag.
2. It saves resulting model in models/ folder with image_size-batch_size-epochs as the model name.

## 3. Make Predictions for Unseen Data:
python3 -m src.make_prediction -bs batch_size -path data_path -mod model

1. Assumes the image data is in a folder that is specified in path.
2. Defaults for all flags are set and the best model is added to make predictions by default.
3. `Best performing Model: 256-5-10` (Loaded by default in make_prediction no need to specify)



