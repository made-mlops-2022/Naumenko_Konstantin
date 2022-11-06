import logging

import hydra
import mlflow
from omegaconf import DictConfig
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from utils import read_data, save_model


logger = logging.getLogger('Train')


@hydra.main(version_base=None,
            config_path='../../configs',
            config_name='config'
            )
def train(cfg: DictConfig):
    logger.info(f"Start training, scaler:{cfg.model.scaler} model:{cfg.model.name}, params:{cfg.model.params}")
    X, y = read_data(
        cfg.dataset.path,
        cfg.dataset.features,
        cfg.dataset.target
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=cfg.val_size,
        random_state=cfg.random_state
    )
    logger.info(f"Train data shape:{X_train.shape}, val data shape:{X_val.shape}")
    if cfg.model.scaler == 'StandardScaler':
        scaler = StandardScaler()
    elif cfg.model.scaler == 'MinMaxScaler':
        scaler = MinMaxScaler()
    elif cfg.model.scaler == 'None':
        scaler = None

    if cfg.model.name == 'LogisticRegression':
        model = LogisticRegression(
            **cfg.model.params,
            random_state=cfg.random_state
        )
    elif cfg.model.name == 'RandomForestClassifier':
        model = RandomForestClassifier(
            **cfg.model.params,
            random_state=cfg.random_state
        )

    pipe = Pipeline([('scaler', scaler), ('model', model)])
    pipe.fit(X_train, y_train)
    val_score = pipe.score(X_val, y_val)
    logger.info(f"Validation score:{val_score}")

    if cfg.ml_flow:
        logger.info(f"Metrics saved in mlflow")
        with mlflow.start_run():
            mlflow.log_param("model", cfg.model.name)

            # log single key-value metric
            mlflow.log_metric("val_score", val_score, step=1)

    save_model(pipe, cfg.model_path)
    logger.info(f"Model saved in {cfg.model_path}")


if __name__ == '__main__':
    train()
