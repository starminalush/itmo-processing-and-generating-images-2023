import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from torch import optim
from core.autoencoder_modules import Decoder, Encoder
from core.loss import ReconstuctionLoss

torch.manual_seed(42)
pl.seed_everything(42)


class AutoEncoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(
            num_input_channels=3, base_channel_size=32, latent_dim=128
        )
        self.decoder = Decoder(
            num_input_channels=3, base_channel_size=32, latent_dim=128
        )
        self.mse_loss = ReconstuctionLoss()

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = optim.AdamW(self.parameters(), lr=1e-2)
        return optimizer

    def _model_forward(self, x):
        x_hat = self.forward(x)
        loss = self.mse_loss(x, x_hat).mean()
        return loss

    def training_step(self, x, batch_idx):
        loss = self._model_forward(x)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, x, batch_idx):
        loss = self._model_forward(x)
        self.log("val_loss", loss)

    def test_step(self, x, batch_idx):
        loss = self._model_forward(x)
        self.log("test_loss", loss)
