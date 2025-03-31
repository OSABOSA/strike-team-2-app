import kagglehub
import kagglehub.config
import shutil
import os

from path import Path
from src import RAW_DATA_FOLDER

def download_dataset() -> None:
    try: assert RAW_DATA_FOLDER.exists()
    except AssertionError: print("Resources folder not found."); raise AssertionError

    kaggle_dataset_cache_folder: Path = Path(kagglehub.config.get_cache_folder()+"/datasets/ankkur13")
    
    if kaggle_dataset_cache_folder.exists(): 
        shutil.rmtree(kaggle_dataset_cache_folder)
    
    dataset_path: str = kagglehub.dataset_download(handle="ankkur13/edmundsconsumer-car-ratings-and-reviews")

    if os.name == "nt":
        for file in Path(dataset_path).iterdir():
            shutil.move(str(file), str(RAW_DATA_FOLDER))
        shutil.rmtree(kaggle_dataset_cache_folder, ignore_errors=True)
    else:  # Linux/macOS
        assert not os.system(f"mv -v {dataset_path}/* {RAW_DATA_FOLDER}")
        assert not os.system(f"rm -rf {kaggle_dataset_cache_folder}")


if __name__ == "__main__": 
    download_dataset()



