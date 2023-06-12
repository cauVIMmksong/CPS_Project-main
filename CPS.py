import os
from torch.utils.data import Dataset
from PIL import Image




class CPS(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_names = []
        self.root_dir = root_dir
        self.transform = transform
        for image_name in os.listdir(root_dir):
            self.image_names.append(os.path.join(root_dir, image_name))

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image = Image.open(self.image_names[idx]).convert('RGB')


        if self.transform:
            image = self.transform(image)

        return image