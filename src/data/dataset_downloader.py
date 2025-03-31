import kagglehub
import kagglehub.config
from path import Path

from os import system, listdir
from src import RAW_DATA_FOLDER

def download_dataset() -> None:
    try: assert RAW_DATA_FOLDER.exists()
    except AssertionError: print("Resources folder not found."); raise AssertionError

    kaggle_dataset_cache_folder: Path = Path(kagglehub.config.get_cache_folder()+"/ankkur13")

    try: assert kaggle_dataset_cache_folder.exists()
    except AssertionError: print("'{}' alredy exisits".format(kaggle_dataset_cache_folder)); raise AssertionError
    
    dataset_path: str = kagglehub.dataset_download(handle="ankkur13/edmundsconsumer-car-ratings-and-reviews")

    assert not system(f"mv -v {dataset_path}/* {RAW_DATA_FOLDER}")
    assert not system(f"rm -rf {kaggle_dataset_cache_folder}")

if __name__ == "__main__": 
    download_dataset()
