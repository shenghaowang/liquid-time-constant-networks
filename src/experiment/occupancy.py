import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from data.occupancy import load_data
from data.utils import cut_in_sequences


# https://stackoverflow.com/questions/62664386/how-to-pass-a-hydra-config-via-command-line
@hydra.main(version_base=None, config_path="../config", config_name="occupancy")
def main(cfg: DictConfig):
    logger.debug(OmegaConf.to_container(cfg))

    X_train, y_train = load_data(cfg.data.train[0], cfg.data.features, cfg.data.target)
    logger.debug(f"X_train: {X_train.shape}, y_train: {y_train.shape}")

    X_train, y_train = cut_in_sequences(X_train, y_train, 16)
    logger.debug(f"X_train: {X_train.shape}, y_train: {y_train.shape}")

    X_test, y_test = load_data(cfg.data.test[0], cfg.data.features, cfg.data.target)
    logger.debug(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

    X_test, y_test = cut_in_sequences(X_test, y_test, 16, 8)
    logger.debug(f"X_test: {X_test.shape}, y_test: {y_test.shape}")


if __name__ == "__main__":
    main()
