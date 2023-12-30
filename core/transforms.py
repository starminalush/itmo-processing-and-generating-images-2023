from torchvision.transforms import Compose, Resize, RandomHorizontalFlip, ToTensor, Normalize, RandomRotation, \
    RandomVerticalFlip, ColorJitter


def get_train_transforms(img_size: int) -> Compose:
    return Compose(
        [
            Resize(size=(img_size, img_size)),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            RandomRotation(15),
            ToTensor(),
            Normalize(mean=0, std=1),
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
