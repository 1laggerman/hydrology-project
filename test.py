# from utils.general import *
# from utils.configs import *
# from nhWrap.neuralhydrology.neuralhydrology.utils.config import Config
# from pathlib import Path

# build_basins_config('../data/Caravan', 'configs')

import os
import pandas as pd

def find_negative_in_csvs(folder_path: str, column_name: str, show_rows=False):
    """
    Searches all CSV files in the given folder for negative values in the specified column.

    Args:
        folder_path (str): Path to the folder containing CSV files.
        column_name (str): Column to check for negative values.
        show_rows (bool): If True, prints the rows that contain negative values.

    Returns:
        list: List of CSV filenames that contain negative values in the given column.
    """
    csv_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.csv')]
    files_with_negatives = []

    for filename in csv_files:
        file_path = os.path.join(folder_path, filename)
        try:
            df = pd.read_csv(file_path)

            if column_name not in df.columns:
                print(f"⚠️ Column '{column_name}' not found in '{filename}'. Skipping.")
                continue

            negatives = df[df[column_name] < 0]
            if not negatives.empty:
                print(f"❗ Negative values found in '{filename}'")
                files_with_negatives.append(filename)
                if show_rows:
                    print(negatives)
            else:
                print(f"✅ No negatives in '{filename}'")

        except Exception as e:
            print(f"⚠️ Error reading '{filename}': {e}")

    return files_with_negatives


if __name__ == "__main__":
    folder = "I:/csv/il"
    column = "Flow_m3_sec"

    result = find_negative_in_csvs(folder, column, show_rows=False)

    print("\nSummary:")
    if result:
        print("Files with negative values:")
        for f in result:
            print(" -", f)
    else:
        print("No negative values found in any file.")

