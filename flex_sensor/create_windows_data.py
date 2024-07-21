import os
import pandas as pd
import numpy as np
from sklearn.utils import resample

def list_folders(directory):
    entries = os.listdir(directory)
    folders = [entry for entry in entries if os.path.isdir(os.path.join(directory, entry))]
    return folders

def create_time_series_windows(data, window_size):
    windows = []
    labels = []
    n_windows = len(data) - window_size
    for i in range(len(data) - window_size):
        print(f"{i}/{n_windows} windows")
        window = data.iloc[i:i + window_size, :-3].values
        label = data.iloc[i + window_size - 1, -3:].values
        windows.append(window)
        labels.append(label)

    return np.array(windows), np.array(labels)

def read_and_process_datasets(bags_folder, eval_bag, window_size, filtered=False):
    bags = list_folders(bags_folder)
    file_list = []

    if filtered: 
        dataset_file = "flight_dataset_filtered.csv"
    else: 
        dataset_file = "flight_dataset.csv"

    for bag in bags:
        file_dataset = None
        if bag != eval_bag:
            for f in os.listdir(os.path.join(bags_folder, bag)):
                if f.endswith(dataset_file):
                    file_dataset = f
            if file_dataset is not None:
                file_list.append(os.path.join(bags_folder, bag, file_dataset))
            else:
                print(f" Bag {bag} does not have {dataset_file} file")

    all_windows = []
    all_labels = []

    for idx, file_name in enumerate(file_list):
        print(f"File {idx}/{len(file_list)}")
        data = pd.read_csv(file_name)
        windows, labels = create_time_series_windows(data, window_size)
        all_windows.append(windows)
        all_labels.append(labels)

    all_windows = np.concatenate(all_windows)
    all_labels = np.concatenate(all_labels)

    return all_windows, all_labels

def create_test_set(bags_folder, eval_bag, window_size, filtered=False):
    dataset_file = "flight_dataset_filtered.csv" if filtered else "flight_dataset.csv"
    eval_bag_folder = os.path.join(bags_folder, eval_bag)

    file_dataset = None
    for f in os.listdir(eval_bag_folder):
        if f.endswith(dataset_file):
            file_dataset = f

    if file_dataset is not None:
        data = pd.read_csv(os.path.join(eval_bag_folder, file_dataset))
        windows, labels = create_time_series_windows(data, window_size)
    else:
        print(f"Evaluation Bag {eval_bag} does not have {dataset_file} file")
        windows, labels = None, None

    return windows, labels

def main():
    bags_folder = "/media/victor/DATA/rosbag_asta_data/11-07-manual-collisions-for-dataset"
    eval_bag = 'rosbag2_2024_07_11-13_43_23'
    window_size = 20  # Define the size of the time series window
    filtered = False

    # Create train and validation dataset
    windows, labels = read_and_process_datasets(bags_folder, eval_bag, window_size, filtered)
    np.save(os.path.join(bags_folder, "windows.npy"), windows)
    np.save(os.path.join(bags_folder, "labels.npy"), labels)

    # Create test dataset
    test_windows, test_labels = create_test_set(bags_folder, eval_bag, window_size, filtered)
    np.save(os.path.join(bags_folder, "test_windows.npy"), test_windows)
    np.save(os.path.join(bags_folder, "test_labels.npy"), test_labels)

if __name__ == "__main__":
    main()
