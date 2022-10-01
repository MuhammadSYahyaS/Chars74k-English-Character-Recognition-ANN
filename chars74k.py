import os

import cv2
import pandas as pd
import torch
import torchvision.transforms as transforms


class SquarePad:
    def __call__(self, image):
        max_wh = max(image.size)
        p_left, p_top = [(max_wh - s) // 2 for s in image.size]
        p_right, p_bottom = [max_wh - (s+pad)
                             for s, pad in zip(image.size, [p_left, p_top])]
        padding = (p_left, p_top, p_right, p_bottom)
        return transforms.functional.pad(image, padding, 0, 'constant')


DEFAULT_TRANSFORM = transforms.Compose(
    [
        transforms.ToPILImage(),
        SquarePad(),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),  # HWC to CHW and [0, 255] to [0.0, 1.0]
        # [0.0, 1.0] to [-1.0, 1.0]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


class Chars74kDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir="dataset", subset="train", transform=DEFAULT_TRANSFORM):
        self.dataset_dir = dataset_dir
        self.df = pd.read_csv(os.path.join(dataset_dir, "%s.csv" % subset))
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        """Returned image is RGB."""
        s = self.df.iloc[index]
        img = cv2.imread(os.path.join(self.dataset_dir,
                         str(s["path"])), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # HWC
        if self.transform:
            img = self.transform(img)
        return img, int(s["class"])
