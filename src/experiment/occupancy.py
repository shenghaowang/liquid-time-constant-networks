import hydra
import numpy as np
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from data.occupancy import load_data, split_data
from data.utils import cut_in_sequences


# https://stackoverflow.com/questions/62664386/how-to-pass-a-hydra-config-via-command-line
@hydra.main(version_base=None, config_path="../config", config_name="occupancy")
def main(cfg: DictConfig):
    logger.debug(OmegaConf.to_container(cfg))

    # Load training data
    X_train, y_train = load_data(cfg.data.train, cfg.data.features, cfg.data.target)
    X_mean = np.mean(X_train, axis=0)
    X_std = np.std(X_train, axis=0)
    X_train, y_train = cut_in_sequences(X_train, y_train, 16)

    # Split data for training and validation
    X_train, y_train, X_valid, y_valid = split_data(X_train, y_train)
    logger.debug(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    logger.debug(f"X_valid: {X_valid.shape}, y_valid: {y_valid.shape}")

    X_tests, y_tests = [], []
    for test_data_dir in cfg.data.test:
        X_test, y_test = load_data(test_data_dir, cfg.data.features, cfg.data.target)
        X_test = (X_test - X_mean) / X_std

        X_test, y_test = cut_in_sequences(X_test, y_test, 16, 8)
        X_tests.append(X_test)
        y_tests.append(y_test)

    X_test, y_test = np.concatenate(X_tests, axis=1), np.concatenate(y_tests, axis=1)
    logger.debug(f"X_test: {X_test.shape}, y_test: {y_test.shape}")


if __name__ == "__main__":
    main()
