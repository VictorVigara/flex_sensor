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
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, random_split

# Define the beam discretization parameter
beam_discretization = 15  # Number of degrees per class, e.g., 1 degree interval
num_classes = int(360 / beam_discretization)  # Total number of classes

# Define the PyTorch dataset
class SensorDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Define the neural network for classification
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(
            64, num_classes
        )  # Output classes based on beam discretization

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Custom Angular Distance Loss Function
def angular_distance_loss(preds, targets):
    ce_loss = F.cross_entropy(preds, targets)

    # Calculate the predicted and true angles in radians
    pred_angles = torch.argmax(preds, dim=1).float() * (2 * np.pi / num_classes)
    true_angles = targets.float() * (2 * np.pi / num_classes)

    # Compute the circular distance
    angular_distance = torch.min(
        torch.abs(pred_angles - true_angles),
        2 * np.pi - torch.abs(pred_angles - true_angles),
    )

    # Combine Cross-Entropy Loss with Angular Distance
    combined_loss = ce_loss + 0.1 * torch.mean(angular_distance)

    return combined_loss


if __name__ == "__main__":

    # Define the beam discretization parameter
    beam_discretization = 15  # Number of degrees per class, e.g., 1 degree interval
    num_classes = int(360 / beam_discretization)  # Total number of classes

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

    # Save the fitted scaler to a file
    joblib.dump(scaler, f"src/flex_sensor/data/{beam_discretization}deg_scaler.pkl")

    # Concatenate all orientation arrays
    all_orientations = np.hstack(all_orientations)

    # Convert angles to classes based on the beam discretization
    all_classes = (all_orientations / beam_discretization).astype(int) % num_classes

    # Create dataset and split into training and validation sets
    dataset = SensorDataset(all_data, all_classes)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Initialize the model, loss function, and optimizer
    model = SimpleNN()
    criterion = angular_distance_loss
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
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(train_loader)
        training_losses.append(avg_epoch_loss)

        # Evaluate on validation set
        model.eval()
        val_loss = 0
        val_predictions = []
        val_targets = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                val_targets.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        validation_losses.append(avg_val_loss)

        # Calculate validation accuracy
        val_accuracy = accuracy_score(val_targets, val_predictions)

        # Save the model if validation loss is the best we've seen so far
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                model.state_dict(),
                f"best_model_classification_{beam_discretization}deg.pth",
            )

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_epoch_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}"
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
    model.load_state_dict(
        torch.load(f"best_model_classification_{beam_discretization}deg.pth")
    )

    # Test the model
    # Create a test dataset
    test_data_45 = pd.read_csv(
        "/home/victor/ws_sensor_combined/src/flex_sensor/data/250_test_nn/orientation_45_pos_3.5.csv",
        header=None,
    ).values

    # Normalize the test data
    test_data_45 = scaler.transform(test_data_45)

    # Convert test data to tensors
    test_X = torch.tensor(test_data_45, dtype=torch.float32)

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        predictions = model(test_X)
        predicted_classes = torch.argmax(predictions, dim=1).cpu().numpy()

    # Convert predicted classes back to angles
    predicted_angles = predicted_classes * beam_discretization

    # Assuming you have the true angles for the test set
    true_angles = np.full(
        len(test_data_45), 250
    )  # Replace with the actual true angles if available

    # Calculate Mean Absolute Error (MAE)
    mae = mean_absolute_error(true_angles, predicted_angles)

    print("Predictions in degrees:")
    print(predicted_angles)
    print(f"Mean Absolute Error (MAE): {mae:.2f} degrees")
