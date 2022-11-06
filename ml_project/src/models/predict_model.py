import logging

import hydra
from omegaconf import DictConfig
from utils import load_model, read_data, save_prediction


logger = logging.getLogger('Predict')


@hydra.main(version_base=None,
            config_path='../../configs',
            config_name='config'
            )
def predict(cfg: DictConfig):
    logger.info(f"Start prediction, scaler:{cfg.model.scaler} model:{cfg.model.name}, params:{cfg.model.params}")
    X = read_data(cfg.test_path, cfg.dataset.features)
    logger.info(f"Data shape:{X.shape}")
    model = load_model(cfg.model_path)
    prediction = model.predict(X)
    save_prediction(prediction, cfg.prediction_path)
    logger.info(f"Prediction saved to {cfg.prediction_path}")


if __name__ == '__main__':
    predict()
