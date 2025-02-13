import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from data.occupancy import DataModule


# https://stackoverflow.com/questions/62664386/how-to-pass-a-hydra-config-via-command-line
@hydra.main(version_base=None, config_path="../config", config_name="occupancy")
def main(cfg: DictConfig):
    logger.debug(OmegaConf.to_container(cfg))

    dm = DataModule(
        train_file=cfg.data.train,
        test_files=cfg.data.test,
        feature_cols=cfg.data.features,
        target_col=cfg.data.target,
    )
    dm.setup()

    # Check one batch
    train_loader = dm.train_dataloader()
    features, labels = next(iter(train_loader))

    logger.debug(f"Features: {features.shape}")
    logger.debug(f"Labels: {labels.shape}")


if __name__ == "__main__":
    main()
