import torch.nn.functional as F
from torch import nn


class ReconstuctionLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, x_hat):
        loss = F.mse_loss(x, x_hat, reduction="none")
        return loss.sum(dim=[1,2,3]).mean(dim=[0])
