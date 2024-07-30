import os
import matplotlib.pyplot as plt
import pandas as pd
from models.common import load_data
from sklearn.utils import resample

def create_plots(data, bag_folder):
    fig, axs = plt.subplots(4, 1, figsize=(15, 20), sharex=True)

    axs[0].plot(data.index, data["raw_value_1"], label="Sensor 1")
    axs[0].plot(data.index, data["raw_value_2"], label="Sensor 2")
    axs[0].plot(data.index, data["raw_value_3"], label="Sensor 3")
    axs[0].plot(data.index, data["raw_value_4"], label="Sensor 4")
    axs[0].set_ylabel("Raw Values")
    axs[0].set_title("Flex Sensor Raw Values")
    axs[0].legend()

    axs[1].plot(data.index, data["collision_predicted"], label="Contact", color="tab:orange")
    axs[1].set_ylabel("Contact")
    axs[1].set_title("Contact Prediction (0, 1)")
    axs[1].legend()

    axs[2].plot(data.index, data["angle_gt"], label="Angle GT", color="tab:green")
    axs[2].set_ylabel("Angle GT (degrees)")
    axs[2].set_title("Ground Truth Angle (degrees)")
    axs[2].legend()

    axs[3].plot(data.index, data["displacement_gt"], label="Displacement GT", color="tab:red")
    axs[3].set_ylabel("Displacement GT (cm)")
    axs[3].set_title("Ground Truth Displacement (cm)")
    axs[3].legend()

    axs[-1].set_xlabel("Time Steps")

    plt.tight_layout()
    plot_path = os.path.join(bag_folder, "flight_dataset_plots.png")
    plt.savefig(plot_path)
    plt.show()

def list_folders(directory):
    entries = os.listdir(directory)
    folders = [entry for entry in entries if os.path.isdir(os.path.join(directory, entry))]
    return folders

def read_flight_datasets(bags_folder, eval_bags, filtered=False):
    bags = list_folders(bags_folder)
    file_list = []

    if filtered:
        dataset_file = "flight_dataset_filtered.csv"
    else:
        dataset_file = "flight_dataset.csv"

    for bag in bags:
        if bag not in eval_bags:
            for f in os.listdir(os.path.join(bags_folder, bag)):
                if f.endswith(dataset_file):
                    file_list.append(os.path.join(bags_folder, bag, dataset_file))
                    break

    contact_data = []
    no_contact_data = []

    for file_name in file_list:
        data = pd.read_csv(file_name)
        contact_data.append(data[data["collision_predicted"] == 1])
        no_contact_data.append(data[data["collision_predicted"] == 0])

    contact_data = pd.concat(contact_data, ignore_index=True)
    no_contact_data = pd.concat(no_contact_data, ignore_index=True)

    return contact_data, no_contact_data

def balance_dataset(contact_data, no_contact_data):
    no_contact_data["displacement_gt"] = 0.0
    no_contact_data_balanced = resample(no_contact_data, replace=False, n_samples=len(contact_data), random_state=42)
    return no_contact_data_balanced

def save_eval_bag_dataset(bags_folder, eval_bags, filtered=False):
    if filtered:
        dataset_file = "flight_dataset_filtered.csv"
    else:
        dataset_file = "flight_dataset.csv"

    eval_data_frames = []
    for eval_bag in eval_bags:
        eval_bag_folder = os.path.join(bags_folder, eval_bag)
        for f in os.listdir(eval_bag_folder):
            if f.endswith(dataset_file):
                file_dataset = f
                eval_data = pd.read_csv(os.path.join(eval_bag_folder, file_dataset))
                eval_data_frames.append(eval_data)
                break

    if eval_data_frames:
        eval_data_combined = pd.concat(eval_data_frames, ignore_index=True)
        eval_path = os.path.join(bags_folder, "eval_" + dataset_file)
        eval_data_combined.to_csv(eval_path, index=False)
        print(f"Evaluation dataset saved as eval_{dataset_file} in {eval_path}")
    else:
        print(f"No evaluation bags contain {dataset_file} file")

def summarize_dataset(dataset):
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
    plt.hist(contact_data["angle_gt"], bins=16, edgecolor='k', alpha=0.7)
    plt.title(name)
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Frequency')
    plt.grid(True)

    plot_path = os.path.join(bags_folder, f"{name}.png")
    plt.savefig(plot_path)
    plt.show()

def plot_displacement_distribution(contact_data, bags_folder):
    plt.figure(figsize=(10, 6))
    plt.hist(contact_data["displacement_gt"], bins=30, edgecolor='k', alpha=0.7)
    plt.title('Distribution of Displacements in Contact Data')
    plt.xlabel('Displacement (cm)')
    plt.ylabel('Frequency')
    plt.grid(True)

    plot_path = os.path.join(bags_folder, "displacement_distribution_contacts.png")
    plt.savefig(plot_path)
    plt.show()

def main():
    bags_folder = "/media/victor/DATA/rosbag_asta_data/25-07-manual-flight-collisions"
    eval_bags = [
        'rosbag2_2024_07_24-12_50_17',
        'rosbag2_2024_07_24-13_32_39'
    ]

    only_flight = True
    filtered = False

    calib_data_part_from_flight = 8

    if filtered:
        dataset_file = "flight_dataset_filtered.csv"
    else:
        dataset_file = "flight_dataset.csv"

    list_folders(bags_folder)

    calibration_folder = "/home/victor/ws_sensor_combined/src/flex_sensor/data/11-07-8orien-5pos"

    contact_flight_data, no_contact_flight_data = read_flight_datasets(bags_folder, eval_bags, filtered)

    if only_flight:
        combined_contact_data = contact_flight_data
    else:
        calibration_data, orientations, positions, _ = load_data(calibration_folder)
        calibration_data = pd.DataFrame(calibration_data, columns=["raw_value_1", "raw_value_2", "raw_value_3", "raw_value_4"])
        calibration_data["collision_predicted"] = 1
        calibration_data["angle_gt"] = orientations
        calibration_data["displacement_gt"] = positions

        # Discard calibration data with positions = 0.0
        calibration_data = calibration_data[calibration_data["displacement_gt"] != 0.0]

        total_contact_samples = len(contact_flight_data)
        unique_combinations = calibration_data.groupby(['angle_gt', 'displacement_gt']).size().reset_index(name='count')
        samples_per_combination = int(total_contact_samples/calib_data_part_from_flight) // len(unique_combinations)

        sampled_calibration_data = pd.concat(
            [resample(calibration_data[(calibration_data["angle_gt"] == row["angle_gt"]) & (calibration_data["displacement_gt"] == row["displacement_gt"])],
                      replace=False,
                      n_samples=samples_per_combination,
                      random_state=42)
             for _, row in unique_combinations.iterrows()],
            ignore_index=True
        )

        combined_contact_data = pd.concat([contact_flight_data, sampled_calibration_data], ignore_index=True)

    balanced_no_contact_data = balance_dataset(combined_contact_data, no_contact_flight_data)

    balanced_dataset = pd.concat([combined_contact_data, balanced_no_contact_data], ignore_index=True)

    balanced_dataset.to_csv(os.path.join(bags_folder, "balanced_" + dataset_file), index=False)

    save_eval_bag_dataset(bags_folder, eval_bags, filtered)

    flight_summary_df = summarize_dataset(pd.concat([contact_flight_data, no_contact_flight_data], ignore_index=True))
    if not only_flight:
        calibration_summary_df = summarize_dataset(sampled_calibration_data)
    combined_summary_df = summarize_dataset(balanced_dataset)

    print("Flight Data Summary:\n", flight_summary_df)
    if not only_flight:
        print("Calibration Data Summary:\n", calibration_summary_df)
    print("Combined Data Summary:\n", combined_summary_df)

    flight_summary_df.to_csv(os.path.join(bags_folder, "flight_data_summary.csv"), index=False)
    if not only_flight:
        calibration_summary_df.to_csv(os.path.join(bags_folder, "calibration_data_summary.csv"), index=False)
    combined_summary_df.to_csv(os.path.join(bags_folder, "combined_data_summary.csv"), index=False)

    plot_angle_distribution(combined_contact_data, bags_folder, 'combined_angle_distribution_contacts')
    if not only_flight:
        plot_angle_distribution(sampled_calibration_data, bags_folder, 'calibration_angle_distribution_contacts')
    plot_angle_distribution(contact_flight_data, bags_folder, 'flight_angle_distribution_contacts')

    plot_displacement_distribution(combined_contact_data, bags_folder)

    create_plots(balanced_dataset, bags_folder)

if __name__ == "__main__":
    main()
