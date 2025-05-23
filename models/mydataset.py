from torch.utils.data import Dataset
from PIL import Image
from services.image_processing import image_processing
import os

# class MyDataset(Dataset):
#     def __init__(self, data_list, labels, validation=False, vit=False):
#         self.data_list = data_list
#         self.labels = labels
#         self.validation = validation
#         self.vit = vit

#     def __len__(self):
#         return len(self.data_list)

#     def __getitem__(self, idx):
#         item = self.data_list[idx]
#         filename = item[0]
#         roi_data = item[1:]
#         img_path = os.path.join("static", "uploads", "images", filename)
#         img = Image.open(img_path).convert("RGB")
#         img = image_processing(img, roi_data, validation=self.validation, vit=self.vit)
#         label = self.labels[idx]
#         return img, label


class MyDataset(Dataset):
    def __init__(self, data_list, labels, validation=False, vit=False, augment=False):
        self.data_list = data_list
        self.labels = labels
        self.validation = validation
        self.vit = vit
        self.augment = augment

    def __len__(self):
        return len(self.data_list) * (3 if self.augment and not self.validation else 1)

    def __getitem__(self, idx):
        base_idx = idx // 3 if self.augment and not self.validation else idx
        aug_idx = idx % 3 if self.augment and not self.validation else 0

        item = self.data_list[base_idx]
        filename = item[0]
        roi_data = item[1:]
        img_path = os.path.join("static", "uploads", "images", filename)
        img = Image.open(img_path).convert("RGB")

        if self.validation or not self.augment:
            img = image_processing(img, roi_data, validation=True, vit=self.vit)
        else:
            if aug_idx == 0:
                img = image_processing(img, roi_data, validation=False, vit=self.vit)
            else:
                img = image_processing(
                    img, roi_data, validation=False, vit=self.vit, augment=True
                )

        label = self.labels[base_idx]

        return img, label
