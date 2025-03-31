import re

from src import RAW_DATA_FOLDER, CLEARED_DATA_FOLDER

def test_clear_data() -> None:
    test_file = RAW_DATA_FOLDER / "Scraped_Car_Review_hyundai.csv"

    colums: str = ",Review_Date,Author_Name,Vehicle_Title,Review_Title,Review,Rating"
    pattern_1 = r"(?<![0-5])\n"
    pattern_2 = r"\n(?=[^0-9])"
    pattern_3 = r"(?<=\,)[^,\"]+(?=\,[0-5](\.[0-9]+)?\n)"

    with open(test_file, "r") as f:
        data = f.read()
        fixed_data = re.sub(pattern_1, "", data)
        fixed_data = re.sub(pattern_2, "", fixed_data)
        fixed_data = re.sub(pattern_3, r'"\g<0>"', fixed_data)
        fixed_data = fixed_data[:len(colums)] + "\n" + fixed_data[len(colums):]

        with open("test_output.csv", "w") as f:
            f.write(fixed_data)

def clear_data() -> bool:
    """Clear the data in the 'resources/raw_data' folder. If 'resources/cleared_data folder is empty', returns False, True otherwise."""

    assert RAW_DATA_FOLDER.exists()
    assert CLEARED_DATA_FOLDER.exists()

    colums: str = ",Review_Date,Author_Name,Vehicle_Title,Review_Title,Review,Rating"
    pattern_1 = r"(?<![0-5])\n"
    pattern_2 = r"\n(?=[^0-9])"
    pattern_3 = r"(?<=\,)[^,\"]+(?=\,[0-5](\.[0-9]+)?\n)"

    if len(CLEARED_DATA_FOLDER.listdir()): return False

    for file in RAW_DATA_FOLDER.listdir():
        with open(file, "r") as f:
            data = f.read()
            fixed_data = re.sub(pattern_1, "", data)
            fixed_data = re.sub(pattern_2, "", fixed_data)
            fixed_data = re.sub(pattern_3, r'"\g<0>"', fixed_data)
            fixed_data = fixed_data[:len(colums)] + "\n" + fixed_data[len(colums):]

        with open(CLEARED_DATA_FOLDER / file.name, "w") as f: f.write(fixed_data)

    return True

if __name__ == "__main__":
    clear_data()