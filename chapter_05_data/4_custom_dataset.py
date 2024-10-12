import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class CifarDataset(Dataset):
    label_encoder_ = {
        "Airplane": 0,
        "Automobile": 1,
        "Bird": 2,
        "Cat": 3,
        "Deer": 4,
        "Dog": 5,
        "Frog": 6,
        "Horse": 7,
        "Ship": 8,
        "Truck": 9,
    }

    def __init__(self, root_folder):
        self.image_label_pairs = []
        # construct list of: (image_path, label)
        train_foldername = "images/train"
        train_path = os.path.join(root_folder, train_foldername)
        class_folders = os.listdir(train_path)
        for class_name in class_folders:
            class_folder_path = os.path.join(train_path, class_name)
            image_names = os.listdir(class_folder_path)
            for image_name in image_names:
                image_path = os.path.join(class_folder_path, image_name)
                label = self.encode_label(class_name)
                self.image_label_pairs.append((image_path, label))

    def __len__(self):
        return len(self.image_label_pairs)

    def __getitem__(self, idx):
        image_path, label = self.image_label_pairs[idx]

        img = Image.open(image_path)
        img_array = np.array(img)
        img_tensor = torch.tensor(img_array)
        return img_tensor, label

    def encode_label(self, label_str):
        assert isinstance(label_str, str)
        return CifarDataset.label_encoder_[label_str]


if __name__ == "__main__":
    dataset = CifarDataset("/home/ailing/Downloads/cifar10-raw-images/")
    for i in range(len(dataset)):
        img_data, label = dataset[i]
        print("image: ", img_data.shape, "label: ", label)
