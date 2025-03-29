import kagglehub
import kagglehub.config

from os import system
from src import RAW_DATA_FOLDER


try: assert RAW_DATA_FOLDER.exists()
except AssertionError: print("Resources folder not found.")

dataset_path: str = kagglehub.dataset_download(handle="ankkur13/edmundsconsumer-car-ratings-and-reviews")

assert not system(f"mv -v {dataset_path}/* {RAW_DATA_FOLDER}")
assert not system(f"rm -rf {kagglehub.config.get_cache_folder()+"/ankkur13"}")
