import os
import re

import numpy as np
import pandas as pd


def load_data(data_folder_path):
    # List all CSV files in the folder
    file_list = [f for f in os.listdir(data_folder_path) if f.endswith(".csv")]

    # Initialize lists to store data and orientations
    all_data = []
    all_orientations = []
    all_positions = []

    # Regular expression to extract orientation from filename
    orientation_pattern = re.compile(r"orientation_(\d+)_pos_(\d)")

    # Loop through all files
    for file_name in file_list:
        # Extract the orientation from the filename
        match = orientation_pattern.search(file_name)
        if match:
            orientation = int(match.group(1))
            position = match.group(2)

            data = pd.read_csv(
                os.path.join(data_folder_path, file_name), header=None
            ).values

            # Append the data and orientation to the lists
            all_data.append(data)
            all_orientations.append(np.full(data.shape[0], orientation))
            all_positions.append(np.full(data.shape[0], float(position)))

    # Concatenate all data arrays
    all_data = np.vstack(all_data)

    # Concatenate all orientation arrays
    all_orientations = np.hstack(all_orientations)
    all_positions = np.hstack(all_positions)

    return all_data, all_orientations, all_positions
