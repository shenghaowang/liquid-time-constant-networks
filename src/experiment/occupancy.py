import hydra
import pytorch_lightning as pl
import torch
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

    logger.debug(
        f"Features: {features.shape}"
    )  # Should be [batch_size, seq_len, num_features]
    logger.debug(f"Labels: {labels.shape}")  # Should be [batch_size, seq_len]

    # Initialize the LSTM model
    model = LSTM(
        input_dim=len(cfg.data.features), hidden_dim=32, output_dim=cfg.data.seq_len
    )
    trainer = pl.Trainer(
        max_epochs=3,
        accelerator="auto",
        devices=2,
    )
    trainer.fit(model, dm)

    # Test the model
    trainer.test(model, datamodule=dm)

    # Make predictions on the test data
    model.eval()
    predictions = []
    ground_truths = []
    test_loader = dm.test_dataloader()

    with torch.no_grad():
        for batch in test_loader:
            data, target = batch  # Get the input data from the batch
            output = model(data)
            predictions.append(torch.sigmoid(output))
            ground_truths.append(target)

    # Convert predictions and groundtruths to a single tensor
    predictions = torch.cat(predictions, dim=0)
    ground_truths = torch.cat(ground_truths, dim=0)
    predictions = predictions.numpy()
    ground_truths = ground_truths.numpy()

    logger.debug(f"Shape of predictions: {predictions.shape}")
    logger.debug(f"Shape of groundtruths: {ground_truths.shape}")

    logger.debug(f"First row of predictions: {predictions[0]}")
    logger.debug(f"First row of groundtruths: {ground_truths[0]}")

    # Compute test accuracy
    accuracy = (predictions > 0.5) == ground_truths
    logger.debug(f"Shape of accuracy: {accuracy.shape}")
    logger.debug(f"Average accuracy: {accuracy.mean()}")


if __name__ == "__main__":
    main()
