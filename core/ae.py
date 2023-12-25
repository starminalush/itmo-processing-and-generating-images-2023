import pytorch_lightning as pl
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from torch import optim
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryRecall, BinarySpecificity

from core.autoencoder_modules import Decoder, Encoder
from core.loss import ReconstuctionLoss


class AutoEncoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(
            num_input_channels=3, base_channel_size=32, latent_dim=32
        )
        self.decoder = Decoder(
            num_input_channels=3, base_channel_size=32, latent_dim=32
        )
        self.mse_loss = ReconstuctionLoss()

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = optim.AdamW(self.parameters(), lr=1e-2)
        return optimizer
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, mode="min", factor=0.2, patience=10, min_lr=5e-5
        # )
        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": scheduler,
        #     "monitor": "val_loss",
        #}

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
