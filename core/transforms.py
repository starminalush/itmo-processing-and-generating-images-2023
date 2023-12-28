import torch
from torchvision import transforms
from torchvision.transforms import v2


def get_train_transforms(img_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            v2.Resize(38),
            v2.RandomCrop(size=(img_size,img_size)),
            v2.RandomRotation(5),
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.ToTensor(),
            transforms.Normalize(mean=0, std=1),
        ]
    )