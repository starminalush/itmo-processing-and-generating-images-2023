import pytorch_lightning as pl
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from torch import optim
import torch.nn as nn
import torch.nn.functional as F


class AE(pl.LightningModule):
    def __init__(self):
        super(AE, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GELU(),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = optim.Adam(self.parameters(), lr=0.001, weight_decay=1e-5)
        return optimizer

    def _model_forward(self, x):
        reconstruction = self.forward(x)
        mse_loss = F.mse_loss(reconstruction, x)
        return mse_loss

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
