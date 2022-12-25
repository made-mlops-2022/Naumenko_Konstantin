import os

import click
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle


@click.command("download")
@click.argument("input_dir")
@click.argument("model_dir")
def validate(input_dir: str, model_dir: str):
    X_test = pd.read_csv(os.path.join(input_dir, "data_test.csv"))
    y_test = pd.read_csv(os.path.join(input_dir, "target_test.csv"))

    path = os.path.join(model_dir, "model.pkl")
    with open(path, 'rb') as f:
        model = pickle.load(f)

    score = model.score(X_test, y_test)
    print(score)

if __name__ == '__main__':
    validate()