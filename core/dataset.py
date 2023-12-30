from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class DefectsDataset(Dataset):
    def __init__(self, img_dir, transform=None, labels=None):
        self.images = list(Path(img_dir).iterdir())
        self.transform = transform
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image if not self.labels else (image, self.labels[idx])
