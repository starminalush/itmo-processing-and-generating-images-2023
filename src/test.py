from collections import defaultdict
from pathlib import Path

import click
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from PIL import Image
from pytorch_lightning.loggers import WandbLogger
from sklearn.metrics import confusion_matrix
from torchvision.transforms import Compose

from core.ae import AE
from core.transforms import get_test_transforms

torch.manual_seed(42)
pl.seed_everything(42)


def classify_image(
    image_path: Path, transform: Compose, model: pl.LightningModule
) -> int:
    """Make classification on image by threshold.
    Args:
        image_path: Image path.
        transform: Transform for testing.
        model: Trained model.

    Returns:
        1 if image with defects else 0
    """
    with torch.no_grad():
        initial_image = Image.open(image_path).convert("RGB")
        initial_image = transform(initial_image)
        initial_image = initial_image.unsqueeze(0)
        reconstruction_image = model(initial_image.to("cuda"))
        return (
            1
            if F.mse_loss(initial_image, reconstruction_image.cpu()) >= 0.000776
            else 0
        )  # тут с графика на втором шаге цифра


@click.command()
@click.option("--dataset-path", type=click.Path(path_type=Path))
@click.option("--checkpoint-path", type=click.Path(path_type=Path))
@click.option("--img-size", type=int)
@click.option("--project-name", type=str)
def test(
    dataset_path: Path | str,
    checkpoint_path: Path | str,
    img_size: int,
    project_name: str,
) -> None:
    """Test trained model.
    Args:
        dataset_path: Test dataset path.
        checkpoint_path: Trained model checkpoint path.
        img_size: Image size. Needed for transforms.
        project_name: Project name for wandb logging.
    """
    model = AE.load_from_checkpoint(checkpoint_path)
    model.eval()
    transform = get_test_transforms(img_size)
    test_results = defaultdict(dict)
    wandb_logger = WandbLogger(project=project_name)
    dataset_image_dir = dataset_path / "imgs"
    with open(dataset_path / "test_annotation.txt") as file:
        lines = file.readlines()
        for line in lines:
            image_path, label = line.strip().split()
            result = classify_image(dataset_image_dir / image_path, transform, model)
            test_results[image_path] = {"y_true": int(label), "y_pred": int(result)}
    gt = [v["y_true"] for k, v in test_results.items()]
    predictions = [v["y_pred"] for k, v in test_results.items()]
    tn, fp, fn, tp = confusion_matrix(gt, predictions).ravel()
    tnr = tn / (fp + tn)
    tpr = tp / (tp + fn)

    print(f"TNR: {tnr}")
    print(f"TPR: {tpr}")
    wandb_logger.log_metrics({"TNR": tnr, "TPR": tpr})


if __name__ == "__main__":
    test()
