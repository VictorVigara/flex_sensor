import os
import re

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, random_split


# Define the PyTorch dataset
class OrienDataset(Dataset):
    def __init__(self, X, angles, displacements):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.angles = torch.tensor(angles, dtype=torch.float32)
        self.displacements = torch.tensor(displacements, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.angles[idx], self.displacements[idx]


# Define the neural network for continuous output
class NN_orientation_continuous(nn.Module):
    def __init__(self):
        super(NN_orientation_continuous, self).__init__()
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(
            64, 2
        )  # Output two continuous values (angle and displacement)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Custom loss function for continuous output
def continuous_loss(preds, angle_targets, displacement_targets):
    angle_preds = preds[:, 0]
    displacement_preds = preds[:, 1]

    angle_loss = F.mse_loss(angle_preds, angle_targets)
    displacement_loss = F.mse_loss(displacement_preds, displacement_targets)

    return angle_loss + displacement_loss


# Function to load data
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
            position = float(match.group(2))

            data = pd.read_csv(
                os.path.join(data_folder_path, file_name), header=None
            ).values

            # Append the data and orientation to the lists
            all_data.append(data)
            all_orientations.append(np.full(data.shape[0], orientation))
            all_positions.append(np.full(data.shape[0], position))

    # Concatenate all data arrays
    all_data = np.vstack(all_data)

    # Concatenate all orientation arrays
    all_orientations = np.hstack(all_orientations)
    all_positions = np.hstack(all_positions)

    return all_data, all_orientations, all_positions


if __name__ == "__main__":
    # Define the folder containing the CSV files
    data_folder_path = (
        "/home/victor/ws_sensor_combined/src/flex_sensor/data/17-06-4positions/"
    )

    ### LOAD TRAINING DATA ###
    all_data, all_orientations, all_positions = load_data(data_folder_path)

    ### CREATE DATASET AND DATALOADER ###

    # Normalize the data
    scaler = StandardScaler()
    all_data = scaler.fit_transform(all_data)

    # Save the fitted scaler to a file
    joblib.dump(scaler, data_folder_path + "scaler.pkl")

    # Normalize angles to [0, 1] if necessary
    all_orientations = all_orientations / 360.0

    # Create dataset and split into training and validation sets
    dataset = OrienDataset(all_data, all_orientations, all_positions)

    train_size = int(0.8 * len(dataset))
    val_test_size = len(dataset) - train_size
    train_dataset, val_test_dataset = random_split(dataset, [train_size, val_test_size])
    val_size = int(0.5 * len(val_test_dataset))
    test_size = len(val_test_dataset) - val_size
    val_dataset, test_dataset = random_split(val_test_dataset, [val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=test_size, shuffle=True)

    ### TRAINING ###

    # Initialize the model, loss function, and optimizer
    model = NN_orientation_continuous()
    criterion = continuous_loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Store the training and validation losses
    training_losses = []
    validation_losses = []

    # Track the best validation loss
    best_val_loss = float("inf")

    # Train the model
    num_epochs = 1000
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for inputs, angle_labels, displacement_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, angle_labels, displacement_labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(train_loader)
        training_losses.append(avg_epoch_loss)

        # Evaluate on validation set
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, angle_labels, displacement_labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, angle_labels, displacement_labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        validation_losses.append(avg_val_loss)

        # Save the model if validation loss is the best we've seen so far
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                model.state_dict(), data_folder_path + "best_model_continuous.pth"
            )

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_epoch_loss:.4f}, Validation Loss: {avg_val_loss:.4f}"
            )

    # Load the best model
    model.load_state_dict(torch.load(data_folder_path + "best_model_continuous.pth"))

    # Test the model
    model.eval()
    with torch.no_grad():
        test_inputs, test_angle_labels, test_displacement_labels = next(
            iter(test_loader)
        )
        test_outputs = model(test_inputs)
        predicted_angles = test_outputs[:, 0] * 360.0  # Convert back to degrees
        true_angles = test_angle_labels * 360.0  # Convert back to degrees
        predicted_displacements = test_outputs[:, 1]
        true_displacements = test_displacement_labels

        # Calculate Mean Absolute Error (MAE)
        angle_mae = mean_absolute_error(true_angles, predicted_angles)
        displacement_mae = mean_absolute_error(
            true_displacements, predicted_displacements
        )

        print("Predicted Angles in degrees:")
        print(predicted_angles)
        print("True Angles in degrees:")
        print(true_angles)
        print(f"Angle MAE: {angle_mae:.2f} degrees")

        print("Predicted Displacements:")
        print(predicted_displacements)
        print("True Displacements:")
        print(true_displacements)
        print(f"Displacement MAE: {displacement_mae:.2f} units")

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
