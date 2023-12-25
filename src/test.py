from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import Resize, ToTensor
from torchvision.transforms.v2 import Compose, Normalize, RandomCrop
from threshold_selector import mse

from core.ae import AutoEncoder


def read_json_file(file_path):
    """
    Читает данные из файла и преобразует их в словарь.

    Параметры:
    - file_path: путь к файлу с данными

    Возвращает:
    - data_dict: словарь, где ключ - имя файла, значение - 0 или 1
    """
    data_dict = {}

    with open(file_path, 'r') as file:
        for line in file:
            # Разделение строки на имя файла и значение
            file_name, value = line.strip().split()

            # Преобразование значения в целое число
            value = int(value)

            # Добавление записи в словарь
            data_dict[file_name] = value

    return data_dict


transform = Compose(
    [
        Resize(38),
        RandomCrop(size=(32, 32)),
        ToTensor(),
        Normalize(mean=(0.4237, 0.5344, 0.4620), std=(0.0472, 0.0526, 0.0489)),
    ]
)

model = AutoEncoder.load_from_checkpoint('models/model.ckpt')
model.eval()


def classify_image(image_path, image_dataset_dir: Path):
    with (torch.no_grad()):
        initial_image = Image.open(image_dataset_dir / image_path)
        initial_image = transform(initial_image)
        initial_image = initial_image.unsqueeze(0)
        reconstruction_image = model(initial_image.to('cuda')).cpu()
        return 1 if mse(initial_image, reconstruction_image) > 0.86 else 0 # +


def test():
    test_results = defaultdict(dict)
    dataset_image_dir = Path('data/external/defects/test/imgs')
    with open('data/external/defects/test/test_annotation.txt') as file:
        lines = file.readlines()
        for line in lines:
            image_path, label = line.strip().split()
            result = classify_image(image_path, dataset_image_dir)
            test_results[image_path] = {'y_true': int(label), 'y_pred': int(result)}
    fp, tp, fn, tn = 0, 0, 0, 0
    for k, v in test_results.items():
        if v['y_pred'] == 1 and v['y_true'] == 0:
            fp += 1
        elif v['y_pred'] == 1 and v['y_true'] == 1:
            tp += 1
        elif v['y_pred'] == 0 and v['y_true'] == 1:
            fn += 1
        else:
            tn += 1
    print(f"TNR: {1 * (tn / (fp + tn))}")
    print(f"TPR: { 1 * (tp / (tp + fn))}")


if __name__=="__main__":
    test()