from pathlib import Path
import pandas as pd
from typing import List, Tuple
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

from torchvision import transforms
import torch.multiprocessing

from src.file_locations import LINKER_FPATH, IMAGES_DIR, HDF5_STORE_DIR
torch.multiprocessing.set_sharing_strategy('file_system')

GLUTEN_INGREDIENTS = [
    'wheat', 'barley',  'rye', 'triticale', 'farina', 'spelt', 'kamut',
    'farro', 'couscous', 'flour', 'gluten', 'bread', 'sourdough', 'bagels',
    'tortillas', 'malt', 'cake', 'cookie', 'pastry', 'spaghetti', 'penne',
    'beer', 'macaroni', 'pasta', 'penne', 'ravioli', 'lasagne', 'linguine',
    'rigatoni', 'farfalle', 'fusilli', 'loaf', 'cannelloni'
    ]

#  Also can uncomment the following to add ingredients that only sometimes contain gluten
GLUTEN_INGREDIENTS.extend(['soy sauce'])
TRAIN_VAL_TEST_PROPORTIONS = (0.05, 0.05, 0.9)


def main():
    show_example_dishes(n_rows=4)


class ImageDataset(Dataset):
    def __init__(self, raw_data, which_split):
        self.data_with_labels = add_labels_to_data(raw_data)
        self.data_with_labels = self.data_with_labels.reset_index()
        hdf5_name = Path()
        self.images = self.load_images_hdf5()
    
    def load_images_hdf5(self):
        pass

    def save_images_hdf5(self):
        pass

    def __len__(self) -> int:
        return len(self.data_with_labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        this_label = float(self.data_with_labels['contains_gluten'][idx])
        y = torch.tensor(this_label)

        with Image.open(
            f"{IMAGES_DIR}/{self.data_with_labels['Image_Name'][idx]}.jpg") as f:
            # Image size 274x169
            image = np.array(f)
            # image = image[..., :3]

            image = transforms.functional.to_tensor(image)
            image = transforms.functional.normalize(
                image,
                (0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5))

            return image, y


def split_dset(raw_data: pd.DataFrame) ->\
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


def add_labels_to_data(data: pd.DataFrame) -> pd.DataFrame:
    labels = []
    for _, row in data.iterrows():
        ingredients = str(row['Cleaned_Ingredients']).split(',')
        labels.append(does_dish_contain_gluten(ingredients))
    data['contains_gluten'] = labels
    return data


def does_dish_contain_gluten(ingredients: List[str]) -> bool:
    is_glutenous = False
    for gluten_ingredient in GLUTEN_INGREDIENTS:
        if next((s for s in ingredients if gluten_ingredient in s.lower()), None):
            is_glutenous = True
    return is_glutenous


def load_image_text_linker() -> pd.DataFrame:
    data = pd.read_csv(LINKER_FPATH)
    return data


def clean_dataset() -> None:
    image_text_linker = pd.read_csv(LINKER_FPATH)
    print(f"Size before cleaning: {len(image_text_linker)}")
    for fpath in IMAGES_DIR.iterdir():
        with Image.open(fpath) as image:
            if (
                image.size != (274, 169) or
                image.mode != 'RGB' or
                fpath.stat().st_size/1024 <= 7
                ):
                image_name = fpath.parts[-1][:-4]
                image_text_linker = image_text_linker[
                    image_text_linker['Image_Name'] != image_name]
    print(f"Size after cleaning: {len(image_text_linker)}")
    image_text_linker.to_csv(LINKER_FPATH)


def show_example_dishes(n_rows):
    import matplotlib.pyplot as plt
    data, _, _ = get_train_val_test()
    fig, ax = plt.subplots(n_rows, n_rows)
    fig.suptitle("Does dish contain gluten or not")
    idx = 0
    for ii in range(n_rows):
        for jj in range(n_rows):
            image, label = data.__getitem__(idx)
            ax[ii, jj].imshow(image.permute(2, 1, 0))
            ax[ii, jj].set_xticks([])
            ax[ii, jj].set_yticks([])
            ax[ii, jj].set_title(bool(label))
            idx += 1
    plt.show()


if __name__ == '__main__':
    main()
