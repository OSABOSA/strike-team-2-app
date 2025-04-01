from path import Path
from os.path import exists

MAIN_FOLDER: Path = Path(__file__).parent.parent
DATA_FOLDER: Path = MAIN_FOLDER / "resources"
RAW_DATA_FOLDER: Path = DATA_FOLDER / "raw_data"
CLEARED_DATA_FOLDER: Path = DATA_FOLDER / "cleared_data"
EMBEDDINGS_FOLDER: Path = DATA_FOLDER / "embeddings"

if not exists(DATA_FOLDER): DATA_FOLDER.mkdir()
if not exists(RAW_DATA_FOLDER): RAW_DATA_FOLDER.mkdir()
if not exists(CLEARED_DATA_FOLDER): CLEARED_DATA_FOLDER.mkdir()
if not exists(EMBEDDINGS_FOLDER): EMBEDDINGS_FOLDER.mkdir()