import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, random_split

# Define the folder containing the CSV files
data_folder_path = (
    "/home/victor/ws_sensor_combined/src/flex_sensor/data/train_nn_28_05/"
)

# List all CSV files in the folder
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

        data = pd.read_csv(
            os.path.join(data_folder_path, file_name), header=None
        ).values

        # Append the data and orientation to the lists
        all_data.append(data)
        all_orientations.append(np.full(data.shape[0], orientation))

# Concatenate all data arrays
all_data = np.vstack(all_data)

# Normalize the data
scaler = StandardScaler()
all_data = scaler.fit_transform(all_data)

# Concatenate all orientation arrays
all_orientations = np.hstack(all_orientations)

# Convert angles to radians
all_orientations = np.radians(all_orientations)

# Define the PyTorch dataset
class SensorDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Create dataset and split into training and validation sets
dataset = SensorDataset(all_data, all_orientations)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Define the neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Initialize the model, loss function, and optimizer
model = SimpleNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Store the training and validation losses
training_losses = []
validation_losses = []

# Track the best validation loss
best_val_loss = float("inf")

# Train the model
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_epoch_loss = epoch_loss / len(train_loader)
    training_losses.append(avg_epoch_loss)

    # Evaluate on validation set
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)
    validation_losses.append(avg_val_loss)

    # Save the model if validation loss is the best we've seen so far
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "best_model.pth")

    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_epoch_loss:.4f}, Validation Loss: {avg_val_loss:.4f}"
        )

# Plot the training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(training_losses, label="Training Loss")
plt.plot(validation_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Over Epochs")
plt.legend()
plt.grid(True)
plt.show()

# Load the best model
model.load_state_dict(torch.load("best_model.pth"))

# Test the model
# Create a test dataset
test_data_45 = pd.read_csv(
    "/home/victor/ws_sensor_combined/src/flex_sensor/data/45_test_nn/orientation_45_pos_3.5.csv",
    header=None,
).values

# Normalize the test data
test_data_45 = scaler.transform(test_data_45)

# Convert test data to tensors
test_X = torch.tensor(test_data_45, dtype=torch.float32)

# Evaluate the model
model.eval()
with torch.no_grad():
    predictions = model(test_X).squeeze()

# Convert predictions back to degrees for interpretation
predictions_deg = np.degrees(predictions.numpy())
print("Predictions in degrees:")
print(predictions_deg)
