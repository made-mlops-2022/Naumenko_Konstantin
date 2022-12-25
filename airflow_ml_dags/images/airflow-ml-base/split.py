import os

import click
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@click.command("split")
@click.argument("input_dir")
def split(input_dir: str):
    X = pd.read_csv(os.path.join(input_dir, "data.csv"))
    y = pd.read_csv(os.path.join(input_dir, "target.csv"))
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

    os.makedirs(input_dir, exist_ok=True)
    X_train.to_csv(os.path.join(input_dir, "data_train.csv"), index=False)
    y_train.to_csv(os.path.join(input_dir, "target_train.csv"), index=False)

    X_test.to_csv(os.path.join(input_dir, "data_test.csv"), index=False)
    y_test.to_csv(os.path.join(input_dir, "target_test.csv"), index=False)

if __name__ == '__main__':
    split()