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
def train(input_dir: str, model_dir: str):
    X_train = pd.read_csv(os.path.join(input_dir, "data_train.csv"))
    y_train = pd.read_csv(os.path.join(input_dir, "target_train.csv"))

    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, "model.pkl")
    with open(path, 'wb') as f:
        pickle.dump(model, f)


if __name__ == '__main__':
    train()