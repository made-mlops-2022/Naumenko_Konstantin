# -*- coding: utf-8 -*-
import logging
import hydra
from omegaconf import DictConfig
import urllib.request


@hydra.main(version_base=None,
            config_path='../../configs',
            config_name='config'
            )
def download_data(cfg: DictConfig):
    urllib.request.urlretrieve(cfg.dataset.download_link, cfg.dataset.path)


if __name__ == '__main__':
    download_data()
