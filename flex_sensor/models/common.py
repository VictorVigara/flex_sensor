import os
import re

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


# Custom loss function for multi-task learning
def multi_task_loss(
    force_preds,
    force_targets,
    angle_preds,
    angle_targets,
    displacement_preds,
    displacement_targets,
):
    force_loss = F.binary_cross_entropy(force_preds, force_targets)
    angle_loss = F.mse_loss(angle_preds, angle_targets)
    displacement_loss = F.mse_loss(displacement_preds, displacement_targets)

    return force_loss + angle_loss + displacement_loss


# Custom Angular Distance Loss Function for Continuous Angles
def continuous_angular_distance_loss(preds, targets):
    # Ensure angles are in radians
    preds = preds * (2 * np.pi)
    targets = targets * (2 * np.pi)

    # Compute the circular distance
    angular_distance = torch.min(
        torch.abs(preds - targets),
        2 * np.pi - torch.abs(preds - targets),
    )

    # Use Mean Squared Error for the angular distance
    angular_distance_loss = torch.mean(angular_distance**2)

    return angular_distance_loss


# Function to load data
def load_data(data_folder_path):
    # List all CSV files in the folder
    file_list = [f for f in os.listdir(data_folder_path) if f.endswith(".csv")]

    # Initialize lists to store data and orientations
    all_data = []
    all_orientations = []
    all_positions = []
    force_applied = []

    # Regular expression to extract orientation from filename
    orientation_pattern = re.compile(r"orientation_(\d+)_pos_(\d\.\d)")

    # Loop through all files
    for file_name in file_list:
        # Extract the orientation from the filename
        match = orientation_pattern.search(file_name)
        if match:
            orientation = int(match.group(1))
            # if orientation in [45, 135, 225, 315]:
            #     pass
            # else:
            position = float(match.group(2))

            data = pd.read_csv(
                os.path.join(data_folder_path, file_name), header=None
            ).values

            # Append the data and orientation to the lists
            all_data.append(data)
            all_orientations.append(np.full(data.shape[0], orientation))
            all_positions.append(np.full(data.shape[0], position))
            force_applied.append(np.full(data.shape[0], position != 0.0))

    # Concatenate all data arrays
    all_data = np.vstack(all_data)

    # Concatenate all orientation arrays
    all_orientations = np.hstack(all_orientations)
    all_positions = np.hstack(all_positions)
    force_applied = np.hstack(force_applied)

    return all_data, all_orientations, all_positions, force_applied


def calculate_differences(data):
    # Calcular diferencias entre pares de lecturas de sensores
    diff_data = []
    num_sensors = data.shape[1]
    for i in range(num_sensors):
        for j in range(i + 1, num_sensors):
            diff_data.append(data[:, i] - data[:, j])
    return np.array(diff_data).T
