from pathlib import Path

DATA_DIR = Path('data')
LINKER_FPATH = DATA_DIR /\
    'Food Ingredients and Recipe Dataset with Image Name Mapping.csv'
IMAGES_DIR = DATA_DIR / 'Food Images' / 'Food Images'
HDF5_STORE_DIR = DATA_DIR / "hdf5_store"
HDF5_STORE_DIR.mkdir(parents=True, exist_ok=True)

