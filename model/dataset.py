from torch.utils.data import Dataset, DataLoader
import json
import cv2


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

        image = cv2.imread(f"./{self.folder}/" + image_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = (image.astype(np.float32) / 127.5) - 1.0

        return dict(image=image, label_tr=f"A [V] {prompt}", label_so=f"{prompt}")


def get_dataloader(folder_name, batch_size=4):
    dataset = MyDataset(folder_name)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
