import os
import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

data_folder_path = (
    "/home/victor/ws_sensor_combined/src/flex_sensor/data/train_nn_28_05/"
)

file_list = [f for f in os.listdir(data_folder_path) if f.endswith(".csv")]

# Initialize lists to store data and orientations
all_data = []
all_orientations = []

# Regular expression to extract orientation from filename
orientation_pattern = re.compile(r"orientation_(\d+)_pos")

# Loop through all files
for file_name in file_list:
    # Extract the orientation from the filename
    match = orientation_pattern.search(file_name)
    if match:
        orientation = int(match.group(1))

        data = pd.read_csv(data_folder_path + file_name).values.tolist()
        all_data.append(data)
        all_orientations.append(np.full(len(data), orientation))

X = np.vstack(np.array(all_data))
y = np.hstack(np.array(all_orientations))


# Load the data from CSV files
# data_0 = np.array(pd.read_csv('/home/victor/ws_sensor_combined/src/flex_sensor/data/28_05_nn_test/orientation_0_pos_3.5.csv').values.tolist())
# data_90 = np.array(pd.read_csv('/home/victor/ws_sensor_combined/src/flex_sensor/data/28_05_nn_test/orientation_90_pos_3.5.csv').values.tolist())
# data_180 = np.array(pd.read_csv('/home/victor/ws_sensor_combined/src/flex_sensor/data/28_05_nn_test/orientation_180_pos_3.5.csv').values.tolist())
# data_270 = np.array(pd.read_csv('/home/victor/ws_sensor_combined/src/flex_sensor/data/28_05_nn_test/orientation_270_pos_3.5.csv').values.tolist())
# X = np.vstack((data_0, data_90, data_180, data_270))

# orientation_0 = np.full((data_0.shape[0],), 0)
# orientation_90 = np.full((data_90.shape[0],), 90)
# orientation_180 = np.full((data_180.shape[0],), 180)
# orientation_270 = np.full((data_270.shape[0],), 270)
# y = np.hstack((orientation_0, orientation_90, orientation_180, orientation_270))

# Define the PyTorch dataset
class SensorDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Create dataset and dataloader
dataset = SensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Define the neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(4, 16)  # 4 input features
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Initialize the model, loss function, and optimizer
model = SimpleNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 2000
for epoch in range(num_epochs):
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Test the model
# Create a test dataset
test_data_0 = pd.read_csv(
    "/home/victor/ws_sensor_combined/src/flex_sensor/data/45_test_nn/orientation_45_pos_3.5.csv"
)
test_data_0["orientation"] = 0
test_X = test_data_0.iloc[:, :-1].values
test_y = np.radians(test_data_0.iloc[:, -1].values)

# Convert test data to tensors
test_X = torch.tensor(test_X, dtype=torch.float32)
test_y = torch.tensor(test_y, dtype=torch.float32)

# Evaluate the model
model.eval()
with torch.no_grad():
    predictions = model(test_X).squeeze()
    test_loss = criterion(predictions, test_y)
    print(f"Test Loss: {test_loss.item():.4f}")

# Convert predictions back to degrees for interpretation
print(predictions)
