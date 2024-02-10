from torch.utils.data import Dataset, DataLoader
import json
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms


class MyDataset(Dataset):
    def __init__(self, folder_name):
        self.data = []
        self.folder = folder_name
        with open(f"./{folder_name}/prompt.json", 'rt') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        image_name = item['target']
        prompt = item['prompt']

        image_path = f"./{self.folder}/" + image_name
        image = Image.open(image_path).convert('RGB')

        resolution = (512, 512)
        center_crop = True
        random_flip = True

        #
        train_transforms = transforms.Compose(
            [
                transforms.Resize(
                    resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(
                    resolution) if center_crop else transforms.RandomCrop(resolution),
                transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        image = train_transforms(image)

        return dict(image=image, label_tr=f"A [V] {prompt}", label_so=f"{prompt}")


def get_dataloader(folder_name, batch_size=4):
    dataset = MyDataset(folder_name)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
