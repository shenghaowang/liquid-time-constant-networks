import hydra
import pytorch_lightning as pl
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from data.occupancy import DataModule
from model.lstm import LSTM


# https://stackoverflow.com/questions/62664386/how-to-pass-a-hydra-config-via-command-line
@hydra.main(version_base=None, config_path="../config", config_name="occupancy")
def main(cfg: DictConfig):
    logger.debug(OmegaConf.to_container(cfg))

    dm = DataModule(
        train_file=cfg.data.train,
        test_files=cfg.data.test,
        feature_cols=cfg.data.features,
        target_col=cfg.data.target,
        seq_len=cfg.data.seq_len,
    )
    dm.setup()

    # Check one batch
    train_loader = dm.train_dataloader()
    features, labels = next(iter(train_loader))

    logger.debug(f"Features: {features.shape}")
    logger.debug(f"Labels: {labels.shape}")

    model = LSTM(
        input_dim=len(cfg.data.features), hidden_dim=32, output_dim=cfg.data.seq_len
    )
    trainer = pl.Trainer(
        max_epochs=3,
        accelerator="auto",
        devices=2,
    )
    trainer.fit(model, dm)

    trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    main()
