import pickle
import pandas as pd


def read_data(path, features, target=None):
    df = pd.read_csv(path)
    X = df[features]
    if target:
        y = df[target]
        return X, y
    return X


def save_model(model, path):
    with open(path, 'wb') as f:
        pickle.dump(model, f)


def save_prediction(prediction, path):
    pd.DataFrame(prediction).to_csv(path, index=False)


def load_model(path):
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model
