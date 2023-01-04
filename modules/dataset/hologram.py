import glob
import cv2
import numpy as np

from torch.utils.data import Dataset
from torchvision import transforms
class HologramDataset(Dataset):
    def __init__(self, path=None, file_format=None, size=None):
        assert not (path is None or file_format is None or size is None)
        super().__init__()
        self.image_size = size
        self.data_path = path
        self.file_format = file_format
        self.data_names = glob.glob(self.data_path + self.file_format)
        self.data_list = list()
        self.transform = transforms.Compose(
            [
                transforms.ToTensor()
            ]
        )
        self.inverse_transform = transforms.Compose(
            [
                transforms.ToPILImage()
            ]
        )
        for idx in self.data_names:
            img = cv2.imread(idx, cv2.IMREAD_GRAYSCALE)
            img = self.transform(img)
            self.data_list.append(img)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]