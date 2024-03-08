from torchvision.transforms import (CenterCrop, Compose, GaussianBlur,
                                    Normalize, RandomErasing,
                                    RandomHorizontalFlip, RandomRotation,
                                    RandomVerticalFlip, Resize, ToTensor)


def get_train_transforms(img_size: int) -> Compose:
    return Compose(
        [
            Resize(size=(img_size + 10, img_size + 10)),
            CenterCrop((img_size, img_size)),
            GaussianBlur(3),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            RandomRotation(15),
            ToTensor(),
            Normalize(mean=0, std=1),
            RandomErasing(p=0.1),
        ]
    )


def get_test_transforms(img_size: int) -> Compose:
    return Compose(
        [
            Resize(size=(img_size, img_size)),
            ToTensor(),
            Normalize(mean=0, std=1),
        ]
    )
