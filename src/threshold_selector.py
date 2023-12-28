import numpy as np
import torch
import pytorch_lightning as pl
from torchvision.transforms.v2 import RandomCrop
from tqdm import tqdm
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from core.dataset import DefectsDataset
from core.loss import ReconstuctionLoss

from core.ae import AutoEncoder

transform = Compose(
    [
        Resize(size=(38, 38)),
        RandomCrop(size=(32,32)),
        ToTensor(),
        Normalize(mean=0, std=1),
    ]
)




loss = ReconstuctionLoss()
torch.manual_seed(42)
pl.seed_everything(42)

def threshold_selector():
    # построить график распределения и искать порог, по которому можно определять, что там на кадре, пролив или не пролив
    model = AutoEncoder.load_from_checkpoint('models/model.ckpt')
    model.eval()
    mse_losses = list()
    intermediate_dataset = DefectsDataset('data/external/defects/proliv', transform=transform)
    with torch.no_grad():
        for image in intermediate_dataset:
            image = image.unsqueeze(0)
            reconstacted_image = model(image.to('cuda')).cpu()
            mse_losses.append(
                loss(
                    image,
                    reconstacted_image
                ).item()
            )
    print(sorted(mse_losses))


if __name__ == "__main__":
    threshold_selector()
