import logging
import os
import urllib.request

import hydra
from omegaconf import DictConfig

logger = logging.getLogger('Download')


@hydra.main(version_base=None,
            config_path='../../configs',
            config_name='config'
            )
def download_data(cfg: DictConfig):
    os.makedirs(os.path.dirname(cfg.dataset.path), exist_ok=True)
    urllib.request.urlretrieve(cfg.dataset.download_link, cfg.dataset.path)
    logger.info(f"Data downloaded to {cfg.dataset.path}")


if __name__ == '__main__':
    download_data()
