from pathlib import Path

DATA_DIR = Path("data")
LINKER_FPATH = DATA_DIR /\
    "Food Ingredients and Recipe Dataset with Image Name Mapping.csv"
IMAGES_DIR = DATA_DIR / "Food Images" / "Food Images"
HDF5_STORE_DIR = DATA_DIR / "hdf5_store"
HDF5_STORE_DIR.mkdir(parents=True, exist_ok=True)

BBC_SCRAPED_DATA = Path("data", "BBC_scraped_data")
BBC_IMAGES_DIR = BBC_SCRAPED_DATA / "images"
BBC_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

BBC_INGREDIENTS_FPATH = BBC_SCRAPED_DATA / "ingredients.csv"

MODEL_CHECKPOINTS_DIR = Path("checkpoints")
MODEL_CHECKPOINTS_DIR.mkdir(exist_ok=True)