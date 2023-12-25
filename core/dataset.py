from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


class DefectsDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.images = list(Path(img_dir).iterdir())
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        if self.transform:
            image = self.transform(image)
        return image


def collate_fn(batch):
    return torch.stack(batch)
