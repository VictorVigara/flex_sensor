import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Function to load data
def load_data(data_folder_path):
    file_list = [f for f in os.listdir(data_folder_path) if f.endswith(".csv")]
    all_data = []
    all_orientations = []
    all_positions = []
    force_applied = []

    no_contact_data = []

    orientation_pattern = re.compile(r"orientation_(\d+)_pos_(\d\.\d)")

    for file_name in file_list:
        match = orientation_pattern.search(file_name)
        if match:
            orientation = int(match.group(1))
            position = float(match.group(2))
            data = pd.read_csv(
                os.path.join(data_folder_path, file_name), header=None
            ).values

            if position == 0.0:
                no_contact_data.append(data)
            else:
                all_data.append(data)
                all_orientations.append(np.full(data.shape[0], orientation))
                all_positions.append(np.full(data.shape[0], position))
                force_applied.append(np.full(data.shape[0], position != 0.0))

    all_data = np.vstack(all_data)
    all_orientations = np.hstack(all_orientations)
    all_positions = np.hstack(all_positions)
    force_applied = np.hstack(force_applied)

    no_contact_data = np.vstack(no_contact_data) if no_contact_data else np.array([])

    return all_data, all_orientations, all_positions, force_applied, no_contact_data

# Function to normalize data
def normalize_data(data):
    normalized_data = np.zeros_like(data)
    for i in range(data.shape[1]):
        sensor_min = np.min(data[:, i])
        sensor_max = np.max(data[:, i])
        normalized_data[:, i] = (data[:, i] - sensor_min) / (sensor_max - sensor_min)
    return normalized_data

# Function to plot sensor outputs for each orientation
def plot_sensor_outputs(
    data, orientations, positions, data_folder_path, model_type, normalized=False, title_suffix=""
):
    unique_orientations = np.unique(orientations)
    n_cols = 2
    n_rows = (len(unique_orientations) + 1) // n_cols

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 10))
    axs = axs.flatten()

    for i, orientation in enumerate(unique_orientations):
        indices = np.where(orientations == orientation)
        sensor_data = data[indices]
        position_data = positions[indices]

        sorted_indices = np.argsort(position_data)
        position_data_sorted = position_data[sorted_indices]
        sensor_data_sorted = sensor_data[sorted_indices]

        for j in range(sensor_data.shape[1]):
            axs[i].plot(sensor_data_sorted[:, j], label=f"Sensor {j+1}")
        axs[i].set_title(f"Orientation {orientation}° {title_suffix}")
        axs[i].set_xlabel("Sample Number")
        axs[i].set_ylabel(
            "Sensor Output (Normalized)" if normalized else "Sensor Output"
        )

        # Plot vertical lines to separate different positions
        unique_positions = np.unique(position_data_sorted)
        for pos in unique_positions[1:]:
            sep_idx = np.where(position_data_sorted == pos)[0][0]
            axs[i].axvline(x=sep_idx, color="r", linestyle="--")

        axs[i].legend()
        axs[i].grid(True)

    plt.tight_layout()
    plt.savefig(
        os.path.join(
            data_folder_path,
            f'sensor_outputs_all_orientations{"_normalized" if normalized else ""}{title_suffix}.png',
        )
    )
    plt.show()

# Function to create radial plot
def radial_plot(
    data, orientations, positions, data_folder_path, model_type, normalized=False, title_suffix=""
):
    unique_orientations = np.unique(orientations)
    unique_positions = np.unique(positions)

    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)

    for i in range(data.shape[1]):
        for orientation in unique_orientations:
            indices = np.where(orientations == orientation)
            position_data = positions[indices]
            sensor_data = data[indices, i]

            sorted_indices = np.argsort(position_data)
            position_data_sorted = position_data[sorted_indices]
            #sensor_data_sorted = sensor_data[sorted_indices]

            theta = np.deg2rad(np.full_like(position_data_sorted, orientation))
            r = np.arange(len(position_data_sorted))

            ax.plot(
                theta,
                r,
                marker="o",
                markersize=3,
                label=f"Sensor {i+1}" if orientation == unique_orientations[0] else "",
            )

    ax.set_title(f"Radial Plot of Sensor Outputs {title_suffix}")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_xticks(np.deg2rad(unique_orientations))
    ax.set_xticklabels([f"{int(angle)}°" for angle in unique_orientations])
    plt.legend(loc="upper right")
    plt.savefig(
        os.path.join(
            data_folder_path,
            f'radial_plot_{model_type}{"_normalized" if normalized else ""}{title_suffix}.png',
        )
    )
    plt.show()

# Function to plot no-contact data
def plot_no_contact_data(no_contact_data, data_folder_path, model_type, normalized=False):
    plt.figure(figsize=(15, 10))
    for i in range(no_contact_data.shape[1]):
        plt.plot(no_contact_data[:, i], label=f"Sensor {i+1}")
    plt.title("No Contact Data")
    plt.xlabel("Sample Number")
    plt.ylabel("Sensor Output (Normalized)" if normalized else "Sensor Output")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            data_folder_path,
            f'no_contact_data{"_normalized" if normalized else ""}.png',
        )
    )
    plt.show()

if __name__ == "__main__":
    data_folder_path = "src/flex_sensor/data/11-07-8orien-5pos"  # Change to your data folder path
    model_type = "data_analysis"

    all_data, all_orientations, all_positions, all_contact, no_contact_data = load_data(data_folder_path)

    plot_sensor_outputs(
        all_data, all_orientations, all_positions, data_folder_path, model_type
    )
    normalized_data = normalize_data(all_data)
    plot_sensor_outputs(
        normalized_data,
        all_orientations,
        all_positions,
        data_folder_path,
        model_type,
        normalized=True,
    )
    """ radial_plot(all_data, all_orientations, all_positions, data_folder_path, model_type)
    radial_plot(
        normalized_data,
        all_orientations,
        all_positions,
        data_folder_path,
        model_type,
        normalized=True,
    ) """

    # Plotting no-contact data
    if no_contact_data.size > 0:
        plot_no_contact_data(no_contact_data, data_folder_path, model_type)
