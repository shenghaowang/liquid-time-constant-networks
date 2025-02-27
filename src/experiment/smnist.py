import hydra
import pytorch_lightning as pl
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from data.smnist import DataModule
from model.lstm import LSTM


# https://stackoverflow.com/questions/62664386/how-to-pass-a-hydra-config-via-command-line
@hydra.main(version_base=None, config_path="../config", config_name="smnist")
def main(cfg: DictConfig):
    logger.debug(OmegaConf.to_container(cfg))

    dm = DataModule()
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
        input_dim=cfg.input_dim,
        hidden_dim=cfg.hidden_dim,
        output_dim=cfg.output_dim,
        is_binary=False,
    )
    trainer = pl.Trainer(
        max_epochs=5,
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
            predictions.append(torch.argmax(output, dim=1))
            ground_truths.append(target)

    # # Convert predictions and groundtruths to a single tensor
    predictions = torch.cat(predictions, dim=0).numpy()
    ground_truths = torch.cat(ground_truths, dim=0).numpy()

    logger.debug(f"Shape of predictions: {predictions.shape}")
    logger.debug(f"Shape of groundtruths: {ground_truths.shape}")

    logger.debug(f"First row of predictions: {predictions[0]}")
    logger.debug(f"First row of groundtruths: {ground_truths[0]}")

    # Compute test accuracy
    accuracy = predictions == ground_truths
    logger.debug(f"Shape of accuracy: {accuracy.shape}")
    logger.debug(f"Average accuracy: {accuracy.mean()}")


if __name__ == "__main__":
    main()
