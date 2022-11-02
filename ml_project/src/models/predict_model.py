import numpy as np
import hydra
from omegaconf import DictConfig
import logging
import pickle
from utils import read_data, load_model, save_prediction


@hydra.main(version_base=None,
            config_path='../../configs',
            config_name='config'
            )
def predict(cfg: DictConfig):
    model = load_model(cfg.model_path)
    X = read_data(cfg.test_path, cfg.dataset.features)
    prediction = model.predict(X)
    save_prediction(prediction, cfg.prediction_path)


if __name__ == '__main__':
    predict()
    print('OK')
