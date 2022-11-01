import glob
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, path, transform=None):
        self.path = path

        if not Path(self.path).is_dir():
            raise RuntimeError(f'Invalid directory "{path}"')

        self.phase_img_list = glob.glob(self.path + '/*.png')
        self.transform = transform

    def __len__(self):
        return len(self.phase_img_list)

    def __getitem__(self, index):
        img_path = self.phase_img_list[index]
        img = Image.open(img_path).convert('L')

        if self.transform is not None:
            img = self.transform(img)

        return img