import sys
import unittest
from pathlib import Path

import pandas as pd
from hydra import compose, initialize
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(sys.path[0]).resolve().parent / "models"))
sys.path.append(str(Path(sys.path[0]).resolve().parent / "data"))
from make_dataset import download_data
from predict_model import predict
from train_model import train
from utils import load_model, read_data


class MyTestCase(unittest.TestCase):

    def test_utils(self):
        with initialize(version_base=None, config_path='../../configs'):
            cfg = compose(config_name="config", overrides=['dataset.path=ml_project/data/test/test.csv'])
            download_data(cfg)
            X, y = read_data(cfg.dataset.path, cfg.dataset.features, cfg.dataset.target)
            X_true = [69.0, 1.0, 0.0, 160.0, 234.0, 1.0, 2.0, 131.0, 0.0, 0.1, 1.0, 1.0, 0.0,
                      69.0, 0.0, 0.0, 140.0, 239.0, 0.0, 0.0, 151.0, 0.0, 1.8, 0.0, 2.0, 0.0]
            self.assertEqual(X.values[:2].reshape(-1).tolist(), X_true)

    def test_train_predict(self):
        with initialize(version_base=None, config_path='../../configs'):
            cfg = compose(config_name="config", overrides=[])
            self.assertEqual(cfg.model.name, "LogisticRegression")

            train(cfg)
            model = load_model(cfg.model_path)
            self.assertEqual(type(model['model']), LogisticRegression)
            self.assertEqual(model['scaler'], None)
            predict(cfg)
            preds = pd.read_csv(cfg.prediction_path)
            self.assertEqual(preds.values[:10].reshape(-1).tolist(), [0, 0, 0, 0, 0, 1, 0, 1, 0, 1])

            cfg = compose(config_name="config", overrides=['model=RandomForestClassifier'])
            cfg.model.scaler = 'StandardScaler'
            train(cfg)
            model = load_model(cfg.model_path)
            self.assertEqual(type(model['model']), RandomForestClassifier)
            self.assertEqual(type(model['scaler']), StandardScaler)
            predict(cfg)
            preds = pd.read_csv(cfg.prediction_path)
            self.assertEqual(preds.values[:10].reshape(-1).tolist(), [0, 0, 0, 1, 0, 1, 0, 1, 0, 1])


if __name__ == '__main__':
    test = MyTestCase()
    test.test_train_predict()
    test.test_utils()
