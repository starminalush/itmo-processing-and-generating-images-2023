from collections import defaultdict
from pathlib import Path

import pytorch_lightning as pl
import torch
from PIL import Image
from torch import nn
import torch.nn.functional as F
from core.ae import AE
from core.transforms import get_test_transforms
from sklearn.metrics import confusion_matrix


def read_json_file(file_path):
    """
    Читает данные из файла и преобразует их в словарь.

    Параметры:
    - file_path: путь к файлу с данными

    Возвращает:
    - data_dict: словарь, где ключ - имя файла, значение - 0 или 1
    """
    data_dict = {}

    with open(file_path, "r") as file:
        for line in file:
            # Разделение строки на имя файла и значение
            file_name, value = line.strip().split()

            # Преобразование значения в целое число
            value = int(value)

            # Добавление записи в словарь
            data_dict[file_name] = value

    return data_dict


loss = nn.MSELoss(reduction='none')
torch.manual_seed(42)
pl.seed_everything(42)

model = AE.load_from_checkpoint("models/model.ckpt")
model.eval()

transform = get_test_transforms(32)


def classify_image(image_path, image_dataset_dir: Path):
    with torch.no_grad():
        initial_image = Image.open(image_dataset_dir / image_path).convert('RGB')
        initial_image = transform(initial_image)
        initial_image = initial_image.unsqueeze(0)
        reconstruction_image = model(initial_image.to("cuda"))
        return 1 if F.mse_loss(initial_image, reconstruction_image.cpu()) >= 0.0005328908446244895 else 0  # +


def test():
    test_results = defaultdict(dict)
    dataset_image_dir = Path("data/external/defects/test/imgs")
    with open("data/external/defects/test/test_annotation.txt") as file:
        lines = file.readlines()
        for line in lines:
            image_path, label = line.strip().split()
            result = classify_image(image_path, dataset_image_dir)
            test_results[image_path] = {"y_true": int(label), "y_pred": int(result)}
    gt = [v['y_true'] for k,v in test_results.items()]
    predictions = [v['y_pred'] for k, v in test_results.items()]
    tn, fp, fn, tp = confusion_matrix(gt, predictions).ravel()
    print(f"TNR: {1 * (tn / (fp + tn))}")
    print(f"TPR: { 1 * (tp / (tp + fn))}")


if __name__ == "__main__":
    test()
