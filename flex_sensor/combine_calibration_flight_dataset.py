import os

import matplotlib.pyplot as plt
import pandas as pd
from models.common import load_data
from sklearn.utils import resample


def list_folders(directory):
    # Get all entries in the directory
    entries = os.listdir(directory)

    # Filter out only the folders
    folders = [
        entry for entry in entries if os.path.isdir(os.path.join(directory, entry))
    ]

    return folders


def read_flight_datasets(bags_folder, eval_bag):

    bags = list_folders(bags_folder)
    file_list = []

    for bag in bags:
        file_dataset = None
        if bag != eval_bag:
            for f in os.listdir(os.path.join(bags_folder, bag)):
                if f.endswith("flight_dataset.csv"):
                    file_dataset = f
            if file_dataset is not None:
                file_list.append(os.path.join(bags_folder, bag, file_dataset))
            else:
                print(f" Bag {bag} does not have flight_dataset.csv file")

    # Initialize lists to store contact and no contact data
    contact_data = []
    no_contact_data = []

    # Loop through all files
    for file_name in file_list:
        data = pd.read_csv(file_name)

        # Separate contact and no contact data
        contact_data.append(data[data["collision_predicted"] == 1])
        no_contact_data.append(data[data["collision_predicted"] == 0])

    # Concatenate all contact and no contact data
    contact_data = pd.concat(contact_data, ignore_index=True)
    no_contact_data = pd.concat(no_contact_data, ignore_index=True)

    return contact_data, no_contact_data


def balance_dataset(contact_data, no_contact_data):
    # Set displacement to 0 for no contact data
    no_contact_data["displacement_gt"] = 0.0

    # Resample no contact data to match the number of contact samples
    no_contact_data_balanced = resample(
        no_contact_data, replace=False, n_samples=len(contact_data), random_state=42
    )

    return no_contact_data_balanced


def save_eval_bag_dataset(bags_folder, eval_bag):
    eval_bag_folder = os.path.join(bags_folder, eval_bag)
    file_dataset = None

    for f in os.listdir(eval_bag_folder):
        if f.endswith("flight_dataset.csv"):
            file_dataset = f

    if file_dataset is not None:
        eval_data = pd.read_csv(os.path.join(eval_bag_folder, file_dataset))

        eval_path = os.path.join(bags_folder, "eval_flight_dataset.csv")
        eval_data.to_csv(eval_path, index=False)
        print(f"Evaluation dataset saved as eval_flight_dataset.csv in {eval_path}")
    else:
        print(f" Evaluation Bag {eval_bag} does not have flight_dataset.csv file")


def summarize_dataset(dataset):
    # Summary statistics
    total_entries = len(dataset)
    contact_entries = len(dataset[dataset["collision_predicted"] == 1])
    no_contact_entries = len(dataset[dataset["collision_predicted"] == 0])
    different_angles = dataset["angle_gt"].unique()
    different_displacements = dataset["displacement_gt"].unique()

    summary = {
        "Total Entries": total_entries,
        "Contact Entries": contact_entries,
        "No Contact Entries": no_contact_entries,
        "Different Angles": different_angles,
        "Different Displacements": different_displacements,
    }

    summary_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in summary.items()]))
    return summary_df


def plot_angle_distribution(contact_data, bags_folder, name):
    plt.figure(figsize=(10, 6))
    plt.hist(contact_data["angle_gt"], bins=30, edgecolor="k", alpha=0.7)
    plt.title("Distribution of Angles in Contact Data")
    plt.xlabel("Angle (degrees)")
    plt.ylabel("Frequency")
    plt.grid(True)

    # Save the plot to a file
    plot_path = os.path.join(bags_folder, f"{name}.png")
    plt.savefig(plot_path)
    plt.show()


def plot_displacement_distribution(contact_data, bags_folder):
    plt.figure(figsize=(10, 6))
    plt.hist(contact_data["displacement_gt"], bins=30, edgecolor="k", alpha=0.7)
    plt.title("Distribution of Displacements in Contact Data")
    plt.xlabel("Displacement (cm)")
    plt.ylabel("Frequency")
    plt.grid(True)

    # Save the plot to a file
    plot_path = os.path.join(bags_folder, "displacement_distribution_contacts.png")
    plt.savefig(plot_path)
    plt.show()


def main():
    # Path to the flight datasets folder
    bags_folder = (
        "/media/victor/DATA/rosbag_asta_data/11-07-manual-collisions-for-dataset"
    )
    eval_bag = "rosbag2_2024_07_11-12_57_21"

    list_folders(bags_folder)

    # Path to the calibration data folder
    calibration_folder = (
        "/home/victor/ws_sensor_combined/src/flex_sensor/data/11-07-8orien-5pos"
    )

    # Read flight datasets
    contact_flight_data, no_contact_flight_data = read_flight_datasets(
        bags_folder, eval_bag
    )

    # Load calibration data
    calibration_data, orientations, positions, _ = load_data(calibration_folder)
    calibration_data = pd.DataFrame(
        calibration_data,
        columns=["raw_value_1", "raw_value_2", "raw_value_3", "raw_value_4"],
    )
    calibration_data["collision_predicted"] = 1
    calibration_data[
        "angle_gt"
    ] = orientations  # Assuming orientations represent angles
    calibration_data[
        "displacement_gt"
    ] = positions  # Assuming positions represent displacements

    # Combine contact data from flight datasets and calibration data
    combined_contact_data = pd.concat(
        [contact_flight_data, calibration_data], ignore_index=True
    )

    # Balance the dataset
    balanced_no_contact_data = balance_dataset(
        combined_contact_data, no_contact_flight_data
    )

    # Combine balanced dataset
    balanced_dataset = pd.concat(
        [combined_contact_data, balanced_no_contact_data], ignore_index=True
    )

    # Save the balanced dataset to a CSV file
    balanced_dataset.to_csv(
        os.path.join(bags_folder, "balanced_flight_dataset.csv"), index=False
    )

    # Save the evaluation dataset
    save_eval_bag_dataset(bags_folder, eval_bag)

    # Summarize the datasets separately
    flight_summary_df = summarize_dataset(
        pd.concat([contact_flight_data, no_contact_flight_data], ignore_index=True)
    )
    calibration_summary_df = summarize_dataset(calibration_data)
    combined_summary_df = summarize_dataset(balanced_dataset)

    print("Flight Data Summary:\n", flight_summary_df)
    print("Calibration Data Summary:\n", calibration_summary_df)
    print("Combined Data Summary:\n", combined_summary_df)

    # Save the summaries to CSV files
    flight_summary_df.to_csv(
        os.path.join(bags_folder, "flight_data_summary.csv"), index=False
    )
    calibration_summary_df.to_csv(
        os.path.join(bags_folder, "calibration_data_summary.csv"), index=False
    )
    combined_summary_df.to_csv(
        os.path.join(bags_folder, "combined_data_summary.csv"), index=False
    )

    # Plot the angle distribution for contact data
    plot_angle_distribution(
        combined_contact_data, bags_folder, "combined_angle_distribution_contacts"
    )
    plot_angle_distribution(
        calibration_data, bags_folder, "calibration_angle_distribution_contacts"
    )
    plot_angle_distribution(
        contact_flight_data, bags_folder, "flight_angle_distribution_contacts"
    )

    # Plot the displacement distribution for contact data
    plot_displacement_distribution(combined_contact_data, bags_folder)


if __name__ == "__main__":
    main()
