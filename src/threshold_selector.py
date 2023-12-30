from pathlib import Path

import click
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from core.ae import AE
from core.dataset import DefectsDataset
from core.transforms import get_test_transforms

torch.manual_seed(42)
pl.seed_everything(42)


@click.command()
@click.option("--dataset-path", type=click.Path(path_type=Path))
@click.option("--checkpoint-path", type=click.Path(path_type=Path))
@click.option("--img-size", type=int)
def threshold_selector(
    dataset_path: Path | str, checkpoint_path: Path | str, img_size: int
):
    model = AE.load_from_checkpoint(checkpoint_path)
    model.eval()
    mse_losses = list()
    intermediate_dataset = DefectsDataset(
        dataset_path, transform=get_test_transforms(img_size)
    )
    intermediate_dataloader = DataLoader(intermediate_dataset, batch_size=1)
    with torch.no_grad():
        for image in intermediate_dataloader:
            reconstacted_image = model(image.to("cuda"))
            mse_losses.append(F.mse_loss(image, reconstacted_image.cpu()).item())

    plt.hist(mse_losses, bins=30, density=True, alpha=0.5, color="b")
    plt.title("Loss Distribution")
    plt.xlabel("Loss Value")
    plt.ylabel("Frequency")
    plt.savefig("reports/mse.png")


if __name__ == "__main__":
    threshold_selector()
