import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, random_split

if __name__ == "__main__":
    from common import load_data, multi_task_loss
    from result_analysis import displacement_analysis, orientation_analysis
else:
    from .common import load_data, multi_task_loss
    from .result_analysis import displacement_analysis, orientation_analysis


# Define the PyTorch dataset
class FFNNRawDataset(Dataset):
    def __init__(self, X, angles, displacements, contact):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.sin_angles = torch.tensor(np.sin(angles * 2 * np.pi), dtype=torch.float32)
        self.cos_angles = torch.tensor(np.cos(angles * 2 * np.pi), dtype=torch.float32)
        self.displacements = torch.tensor(displacements, dtype=torch.float32)
        self.contact = torch.tensor(contact, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            self.X[idx],
            self.sin_angles[idx],
            self.cos_angles[idx],
            self.displacements[idx],
            self.contact[idx],
        )


# Define the neural network for continuous output
class FNNRaw_sincos(nn.Module):
    def __init__(self):
        super(FNNRaw_sincos, self).__init__()
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc_force = nn.Linear(64, 1)
        self.fc_sin_angle = nn.Linear(64, 1)
        self.fc_cos_angle = nn.Linear(64, 1)
        self.fc_displacement = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        force_applied = torch.sigmoid(self.fc_force(x))
        sin_angle = self.fc_sin_angle(x)
        cos_angle = self.fc_cos_angle(x)
        displacement = self.fc_displacement(x)
        return force_applied, sin_angle, cos_angle, displacement


def continuous_angular_distance_loss(sin_preds, cos_preds, sin_targets, cos_targets):
    sin_cos_distance = torch.sqrt(
        (sin_preds - sin_targets) ** 2 + (cos_preds - cos_targets) ** 2
    )
    return torch.mean(sin_cos_distance)


def multi_task_loss(
    force_preds,
    force_targets,
    sin_angle_preds,
    sin_angle_targets,
    cos_angle_preds,
    cos_angle_targets,
    displacement_preds,
    displacement_targets,
):
    force_loss = F.binary_cross_entropy(force_preds, force_targets)
    angle_loss = continuous_angular_distance_loss(
        sin_angle_preds, cos_angle_preds, sin_angle_targets, cos_angle_targets
    )
    displacement_loss = F.mse_loss(displacement_preds, displacement_targets)

    return force_loss + angle_loss + displacement_loss


if __name__ == "__main__":
    model_type = "FFNNRaw_sincos"
    # Define the folder containing the CSV files
    data_folder_path = (
        "/home/victor/ws_sensor_combined/src/flex_sensor/data/04-07-8pos-5disp/"
    )
    center_orientations = [0, 45, 90, 135, 180, 225, 270, 315]
    center_displacements = [0.5, 1.0, 1.5, 2.0, 2.5]

    model_output_path = os.path.join(data_folder_path, model_type)
    os.makedirs(model_output_path, exist_ok=True)

    ### LOAD TRAINING DATA ###
    all_data, all_orientations, all_positions, all_contact = load_data(data_folder_path)

    ### CREATE DATASET AND DATALOADER ###

    # Normalize the data
    scaler = StandardScaler()
    all_data = scaler.fit_transform(all_data)

    # Save the fitted scaler to a file
    joblib.dump(scaler, model_output_path + "/scaler.pkl")

    # Normalize angles to [0, 1] if necessary
    all_orientations = all_orientations / 360.0

    # Create dataset and split into training and validation sets
    dataset = FFNNRawDataset(all_data, all_orientations, all_positions, all_contact)

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
    model = FNNRaw_sincos()
    criterion = multi_task_loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Store the training and validation losses
    training_losses = []
    validation_losses = []

    # Track the best validation loss
    best_val_loss = float("inf")

    # Train the model
    num_epochs = 30
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for (
            inputs,
            sin_angle_labels,
            cos_angle_labels,
            displacement_labels,
            force_labels,
        ) in train_loader:
            optimizer.zero_grad()
            force_preds, sin_angle_preds, cos_angle_preds, displacement_preds = model(
                inputs
            )
            loss = criterion(
                force_preds.squeeze(),
                force_labels,
                sin_angle_preds.squeeze(),
                sin_angle_labels,
                cos_angle_preds.squeeze(),
                cos_angle_labels,
                displacement_preds.squeeze(),
                displacement_labels,
            )
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(train_loader)
        training_losses.append(avg_epoch_loss)

        # Evaluate on validation set
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for (
                inputs,
                sin_angle_labels,
                cos_angle_labels,
                displacement_labels,
                force_labels,
            ) in val_loader:
                (
                    force_preds,
                    sin_angle_preds,
                    cos_angle_preds,
                    displacement_preds,
                ) = model(inputs)
                loss = criterion(
                    force_preds.squeeze(),
                    force_labels,
                    sin_angle_preds.squeeze(),
                    sin_angle_labels,
                    cos_angle_preds.squeeze(),
                    cos_angle_labels,
                    displacement_preds.squeeze(),
                    displacement_labels,
                )
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        validation_losses.append(avg_val_loss)

        # Save the model if validation loss is the best we've seen so far
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_output_path + f"/model.pth")

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_epoch_loss:.4f}, Validation Loss: {avg_val_loss:.4f}"
            )

    # Load the best model
    model.load_state_dict(torch.load(model_output_path + "/model.pth"))

    # Test the model
    model.eval()
    with torch.no_grad():
        (
            test_inputs,
            test_sin_angle_labels,
            test_cos_angle_labels,
            test_displacement_labels,
            test_force_labels,
        ) = next(iter(test_loader))
        (
            test_force_preds,
            test_sin_angle_preds,
            test_cos_angle_preds,
            test_displacement_preds,
        ) = model(test_inputs)

        predicted_force = test_force_preds.squeeze().round()
        true_force = test_force_labels
        predicted_sin_angles = test_sin_angle_preds.squeeze()
        predicted_cos_angles = test_cos_angle_preds.squeeze()
        true_sin_angles = test_sin_angle_labels
        true_cos_angles = test_cos_angle_labels

        # Convert predicted sin/cos to angles
        predicted_angles = torch.atan2(predicted_sin_angles, predicted_cos_angles) * (
            360.0 / (2 * np.pi)
        )
        true_angles = torch.atan2(true_sin_angles, true_cos_angles) * (
            360.0 / (2 * np.pi)
        )

        predicted_displacements = test_displacement_preds.squeeze()
        true_displacements = test_displacement_labels

        # Calculate Mean Absolute Error (MAE)
        angle_mae = mean_absolute_error(true_angles, predicted_angles)
        displacement_mae = mean_absolute_error(
            true_displacements, predicted_displacements
        )
        force_accuracy = accuracy_score(true_force, predicted_force)

        print(f"Force Detection Accuracy: {force_accuracy:.2f}")
        print(f"Angle MAE: {angle_mae:.2f} degrees")
        print(f"Displacement MAE: {displacement_mae:.2f} cm")

    orientation_analysis(
        true_angles, predicted_angles, center_orientations, data_folder_path, model_type
    )
    displacement_analysis(
        true_displacements,
        predicted_displacements,
        center_displacements,
        data_folder_path,
        model_type,
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
