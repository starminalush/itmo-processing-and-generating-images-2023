import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.v2 import RandomCrop
from tqdm import tqdm
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from core.dataset import DefectsDataset

from core.ae import AutoEncoder

transform = Compose(
    [
        Resize(size=(38, 38)),
        RandomCrop(size=(32,32)),
        ToTensor(),
        Normalize(mean=(0.4237, 0.5344, 0.4620), std=(0.0472, 0.0526, 0.0489)),
    ]
)


def find_threshold(mse_losses):
    """
    Находит порог для определения дефектов на основе MSE лоссов.

    Параметры:
    - mse_losses: список значений MSE лоссов

    Возвращает:
    - threshold: найденный порог
    """
    # Используем среднее значение и стандартное отклонение для определения порога
    mean_loss = np.mean(mse_losses)
    std_deviation = np.std(mse_losses)

    # Можно использовать какое-то множитель стандартного отклонения
    # для настройки уровня порога
    threshold = mean_loss + 2 * std_deviation

    return threshold


def mse(initial_image, reconstructed_image):
    loss = F.mse_loss(initial_image, reconstructed_image, reduction="none")
    return loss.mean()


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
                mse(
                    image,
                    reconstacted_image
                ).item()
            )
    print(sorted(mse_losses))
    threshold = find_threshold(mse_losses)
    print(f"threshold: {threshold}")


if __name__ == "__main__":
    threshold_selector()
