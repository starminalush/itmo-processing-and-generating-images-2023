import torch
from torchvision import transforms
from torchvision.transforms import v2


def get_train_transforms(img_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            v2.Resize(38),
            v2.RandomCrop(size=(img_size,img_size)),
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.RandomApply([
                v2.ColorJitter()
            ]),
            v2.ToTensor(),
            v2.Normalize(mean=(0.4237, 0.5344, 0.4620), std=(0.0472, 0.0526, 0.0489)),
        ]
    )