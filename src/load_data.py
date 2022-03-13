import re
from typing import List, Tuple
import numpy as np
import h5py
from PIL import Image

import torch
from torch.utils.data import Dataset

from torchvision import transforms
import torch.multiprocessing

from src.file_locations import BBC_IMAGES_DIR, BBC_INGREDIENTS_FPATH, LINKER_FPATH, IMAGES_DIR, HDF5_STORE_DIR
torch.multiprocessing.set_sharing_strategy('file_system')

GLUTEN_INGREDIENTS = [
    'wheat', 'barley',  'rye', 'triticale', 'farina', 'spelt', 'kamut',
    'farro', 'couscous', 'flour', 'gluten', 'bread', 'sourdough', 'bagels',
    'tortillas', 'malt', 'cake', 'cookie', 'pastry', 'spaghetti', 'penne',
    'beer', 'macaroni', 'pasta', 'penne', 'ravioli', 'lasagne', 'linguine',
    'rigatoni', 'farfalle', 'fusilli', 'loaf', 'cannelloni', 'brioche'
    ]

#  Also can uncomment the following to add ingredients that only sometimes contain gluten
GLUTEN_INGREDIENTS.extend(['soy sauce'])
TRAIN_VAL_TEST_PROPORTIONS = (0.8, 0.1, 0.1)


def main():
    pass


class ImageDataset(Dataset):
    def __init__(self, raw_data, which_split):
        self.labels = get_labels_from_data(raw_data)
        self.raw_data = raw_data
        self.hdf5_fname = HDF5_STORE_DIR / f"{which_split}.hdf5"
        self.dset = self.load_images_hdf5()
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.preprocessing = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
    
    def load_images_hdf5(self, rewrite: bool = False):
        if not self.hdf5_fname.is_file() or rewrite:
            self.save_images_hdf5()
        dset = h5py.File(self.hdf5_fname, "r")
        return dset

    def save_images_hdf5(self) -> None:
        with h5py.File(self.hdf5_fname, "w", libver="latest") as f:
            for ii, row in enumerate(self.raw_data):
                image_fpath = BBC_IMAGES_DIR / f"{row[0]}.jpg"
                with Image.open(image_fpath) as fi:
                    image = np.array(fi)
                f.create_dataset(str(ii), data=image, compression="gzip")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        this_label = float(self.labels[idx])
        y = torch.tensor(this_label)
        image = Image.fromarray(np.array(self.dset[str(idx)]))
        image = self.preprocessing(image)
        name = self.raw_data[idx][0]
        ingredients = self.raw_data[idx][1]

        return image, y, name, ingredients


def split_dset(raw_data: List[List[str]]) ->\
        Tuple[ImageDataset, ImageDataset, ImageDataset]:
    data_length = len(raw_data)
    train_end = int(TRAIN_VAL_TEST_PROPORTIONS[0] * data_length)
    test_start = int(-TRAIN_VAL_TEST_PROPORTIONS[2] * data_length)
    train_data = raw_data[:train_end]
    test_data = raw_data[test_start:]
    val_data = raw_data[train_end:test_start]
    return (
        ImageDataset(train_data, "train"),
        ImageDataset(val_data, "val"),
        ImageDataset(test_data, "test"),
        )


def get_train_val_test() ->\
        Tuple[ImageDataset, ImageDataset, ImageDataset]:
    raw_data = load_image_text_linker()
    train_dataset, val_dataset, test_dataset = split_dset(raw_data)
    return train_dataset, val_dataset, test_dataset


def get_labels_from_data(data: List[List[str]]) -> List[int]:
    labels = []
    for meal in data:
        ingredients = " ".join(meal[1:])
        labels.append(does_dish_contain_gluten(ingredients.lower()))
    return labels


def does_dish_contain_gluten(ingredients: str) -> bool:
    is_glutenous = False
    for gluten_ingredient in GLUTEN_INGREDIENTS:
        if gluten_ingredient in ingredients:
            is_glutenous = True
    return is_glutenous


def load_image_text_linker() -> List[List[str]]:
    with open(BBC_INGREDIENTS_FPATH, "r") as f:
        data = f.readlines()
    data = [x.split(",") for x in data]
    new_data = []
    for dpoint in data:
        ingredients = " ".join(dpoint[1:])
        ingredients = ingredients.split('"')
        ingredients = [x.strip() for x in ingredients if x.strip()]
        ingredients = [re.sub(" +", " ", x) for x in ingredients]
        ingredients = "\n".join(ingredients)
        this_dpoint = [dpoint[0]] + [ingredients]
        new_data.append(this_dpoint)
    return new_data


if __name__ == '__main__':
    main()
