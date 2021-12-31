from pathlib import Path
import pandas as pd
from typing import List

from src.file_locations import LINKER_FPATH, IMAGES_FPATH

GLUTEN_INGREDIENTS = [
    'wheat', 'barley',  'rye', 'triticale', 'farina', 'spelt', 'kamut',\
    'farro', 'couscous', 'flour', 'gluten', 'bread', 'sourdough', 'bagels',\
    'tortillas', 'malt', 'cake', 'cookie', 'pastry', 'spaghetti', 'penne',\
    'beer', 'macaroni', 'pasta', 'penne', 'ravioli', 'lasagne', 'linguine',
    'rigatoni', 'farfalle', 'fusilli', 'loaf'
    ]

#  Also can uncomment the following to add ingredients that only sometimes contain gluten
GLUTEN_INGREDIENTS.extend(['soy sauce'])



def main():
    data = load_image_text_linker()
    data = add_labels_to_data(data)


def add_labels_to_data(data: pd.DataFrame) -> pd.DataFrame:
    labels = []
    for _, row in data.iterrows():
        ingredients = row['Cleaned_Ingredients'].split(',')
        labels.append(does_dish_contain_gluten(ingredients))
    data['contains_gluten'] = labels
    breakpoint()


def does_dish_contain_gluten(ingredients: List[str]) -> bool:
    is_glutenous = False
    for gluten_ingredient in GLUTEN_INGREDIENTS:
        if next((s for s in ingredients if gluten_ingredient in s.lower()), None):
            is_glutenous = True
    return is_glutenous


def load_image_text_linker() -> pd.DataFrame:
    data = pd.read_csv(LINKER_FPATH)
    return data


if __name__ == '__main__':
    main()
