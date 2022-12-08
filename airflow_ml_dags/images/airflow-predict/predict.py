import os
import pandas as pd

import click
import pickle


@click.command("predict")
@click.option("--input-dir")
@click.option("--output-dir")
@click.option("--model-dir")
def predict(input_dir: str, output_dir, model_dir: str):
    data = pd.read_csv(os.path.join(input_dir, "data.csv"))

    path = os.path.join(model_dir, "model.pkl")
    with open(path, 'rb') as f:
        model = pickle.load(f)

    preds = model.predict(data)

    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(preds).to_csv(os.path.join(output_dir, "data.csv"), index=False)


if __name__ == '__main__':
    predict()