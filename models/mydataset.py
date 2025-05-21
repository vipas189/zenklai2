from torch.utils.data import Dataset
from PIL import Image
from services.image_processing import image_processing
import os


class MyDataset(Dataset):
    def __init__(self, data_list, labels, validation=False, vit=False):
        self.data_list = data_list
        self.labels = labels
        self.validation = validation
        self.vit = vit

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        filename = item[0]
        roi_data = item[1:]
        img_path = os.path.join("static", "uploads", "images", filename)
        img = Image.open(img_path).convert("RGB")
        img = image_processing(img, roi_data, validation=self.validation, vit=self.vit)
        label = self.labels[idx]
        return img, label
