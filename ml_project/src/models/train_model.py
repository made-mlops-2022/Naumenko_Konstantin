import hydra
from omegaconf import DictConfig
import logging
import os

from utils import read_data, save_model
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler


@hydra.main(version_base=None,
            config_path='../../configs',
            config_name='config'
            )
def main(cfg: DictConfig):
    print()
    print(cfg.model.params)


@hydra.main(version_base=None,
            config_path='../../configs',
            config_name='config'
            )
def train(cfg: DictConfig):
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
    print('acc:', val_score)

    if cfg.ml_flow:
        with mlflow.start_run():
            mlflow.log_param("model", cfg.model.name)

            # log single key-value metric
            mlflow.log_metric("val_score", val_score, step=1)
            mlflow.log_metric("val_score", val_score+1, step=2)
            mlflow.log_metric("val_score", val_score+2, step=3)

            # logs local file or directory as artifact,
            mlflow.log_artifact("ml_project/configs/config.yaml")

    save_model(model, cfg.model_path)


if __name__ == '__main__':
    train()
