ml_project
==============================

mlops hw1

Installation: 
~~~
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
~~~
Usage:
~~~
python src/models/train_model.py
python src/models/predict_model.py
~~~
Also you can change configs/config.yaml or run script with parameters 
~~~
python src/models/train_model.py model=RandomForestClassifier model.scaler=StandardScaler
python src/models/predict_model.py test_path=data/raw/heart_cleveland_upload.csv prediction_path=output/predictions.csv
~~~
Avilable models: LogisticRegression, RandomForestClassifier

Avilable scalers: StandardScaler, MinMaxScaler

To run with mlflow
~~~
python src/models/train_model.py ml_flow=True
mlflow ui
~~~

Test:
~~~
python -m unittest discover src/tests
~~~



Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── test           <- Data for tests.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │    
    │   ├── tests       <- Test script
    │   │   ├── test_train_predict.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │   └── utils.py




--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
