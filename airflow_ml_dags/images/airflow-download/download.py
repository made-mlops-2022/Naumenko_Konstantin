import os

import click
import numpy as np
from sklearn.datasets import load_iris


@click.command("download")
@click.argument("output_dir")
def download(output_dir: str):
    X, y = load_iris(return_X_y=True, as_frame=True)
    inds = np.random.choice(np.arange(X.shape[0]), size=120)

    X = X.iloc[inds]
    y = y.iloc[inds]

    os.makedirs(output_dir, exist_ok=True)
    X.to_csv(os.path.join(output_dir, "data.csv"), index=False)
    y.to_csv(os.path.join(output_dir, "target.csv"), index=False)


if __name__ == '__main__':
    download()