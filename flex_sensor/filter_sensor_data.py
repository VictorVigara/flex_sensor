import pandas as pd
import numpy as np
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt

# Load the CSV file
file_path = "/media/victor/DATA/rosbag_asta_data/11-07-manual-collisions-for-dataset/rosbag2_2024_07_11-14_01_34"
data = pd.read_csv(file_path + "/flight_dataset.csv")

# Initialize the filter
def init_butter_lowpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

# Apply the filter to a new data point
def butter_lowpass_filter(data, b, a, zi):
    y, zo = lfilter(b, a, [data], zi=zi)
    return y[0], zo

# Parameters for the low-pass filter
cutoff = 4  # desired cutoff frequency of the filter (Hz)
fs = 100.0  # sample rate, Hz
order = 1   # order of the filter

skip_initial = 100

# Initialize filter coefficients and state for each sensor input
filtered_data = pd.DataFrame()
for column in data.columns:
    if column.startswith('raw_value_'):
        b, a = init_butter_lowpass(cutoff, fs, order)
        zi = [0] * order

        # Apply filter in a loop (simulating real-time processing)
        filtered_values = []
        for idx, data_point in enumerate(data[column]):
            filtered_value, zi = butter_lowpass_filter(data_point, b, a, zi)
            if idx > skip_initial:  # Skip the first value
                filtered_values.append(filtered_value)
        
        filtered_data[column] = filtered_values

# Add the rest of the columns from the original dataset to the filtered dataset
for column in data.columns:
    if not column.startswith('raw_value_'):
        filtered_data[column] = data[column][skip_initial:].reset_index(drop=True)  # Skip the first value

# Save the filtered data to a new CSV file
filtered_file_path = file_path + "/flight_dataset_filtered.csv"
filtered_data.to_csv(filtered_file_path, index=False)

# Plot the results for each sensor input
plt.figure(figsize=(12, 6))
for column in data.columns:
    if column.startswith('raw_value_'):
        plt.plot(data[column][skip_initial:].reset_index(drop=True), label=f'Original {column}')
        plt.plot(filtered_data[column], linestyle='--', color=plt.gca().lines[-1].get_color(), label=f'Filtered {column}')
plt.title('Real-Time Low-Pass Filtering of Sensor Data')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.legend()
plt.show()
