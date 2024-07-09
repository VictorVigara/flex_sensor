import os

import joblib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
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
class OrienDataset(Dataset):
    def __init__(self, X, angles, displacements, force_applied):
        self.X = torch.tensor(X, dtype=torch.float32).reshape(
            -1, 1, 2, 2
        )  # Reshape for CNN input
        self.angles = torch.tensor(angles, dtype=torch.float32)
        self.displacements = torch.tensor(displacements, dtype=torch.float32)
        self.force_applied = torch.tensor(force_applied, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            self.X[idx],
            self.angles[idx],
            self.displacements[idx],
            self.force_applied[idx],
        )


# Define the CNN for multi-task learning with smaller kernel size
class CNN_multi_task(nn.Module):
    def __init__(self):
        super(CNN_multi_task, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=1)  # Smaller kernel size
        self.conv2 = nn.Conv2d(16, 32, kernel_size=1)  # Smaller kernel size
        self.fc1 = nn.Linear(32 * 2 * 2, 128)  # Adjust input size
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 64)

        self.fc_force = nn.Linear(64, 1)  # Binary classification for force applied
        self.fc_angle = nn.Linear(64, 1)  # Continuous output for angle
        self.fc_displacement = nn.Linear(64, 1)  # Continuous output for displacement

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))

        force_applied = torch.sigmoid(self.fc_force(x))
        angle = self.fc_angle(x)
        displacement = self.fc_displacement(x)

        return force_applied, angle, displacement


if __name__ == "__main__":
    # Define the folder containing the CSV files
    model_type = "FFNN_CNN_raw"
    data_folder_path = (
        "/home/victor/ws_sensor_combined/src/flex_sensor/data/09-07-4orient-5pos/"
    )
    center_orientations = [0, 90, 180, 270]
    center_displacements = [0.5, 1.0, 1.5, 2.0, 2.5]

    model_output_path = os.path.join(data_folder_path, model_type)
    os.makedirs(model_output_path, exist_ok=True)

    ### LOAD TRAINING DATA ###
    all_data, all_orientations, all_positions, force_applied = load_data(
        data_folder_path
    )

    ### CREATE DATASET AND DATALOADER ###

    # Normalize the data
    scaler = StandardScaler()
    all_data = scaler.fit_transform(all_data)

    # Save the fitted scaler to a file
    joblib.dump(scaler, model_output_path + "/scaler.pkl")

    # Normalize angles to [0, 1] if necessary
    all_orientations = all_orientations / 360.0

    # Create dataset and split into training and validation sets
    dataset = OrienDataset(all_data, all_orientations, all_positions, force_applied)

    train_size = int(0.7 * len(dataset))
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
    if model_type == "FFNN_CNN_raw":
        model = CNN_multi_task()

    criterion = multi_task_loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Store the training and validation losses
    training_losses = []
    validation_losses = []

    # Track the best validation loss
    best_val_loss = float("inf")

    # Train the model
    num_epochs = 200
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for inputs, angle_labels, displacement_labels, force_labels in train_loader:
            optimizer.zero_grad()
            force_preds, angle_preds, displacement_preds = model(inputs)
            loss = criterion(
                force_preds.squeeze(),
                force_labels,
                angle_preds.squeeze(),
                angle_labels,
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
            for inputs, angle_labels, displacement_labels, force_labels in val_loader:
                force_preds, angle_preds, displacement_preds = model(inputs)
                loss = criterion(
                    force_preds.squeeze(),
                    force_labels,
                    angle_preds.squeeze(),
                    angle_labels,
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
            test_angle_labels,
            test_displacement_labels,
            test_force_labels,
        ) = next(iter(test_loader))
        test_force_preds, test_angle_preds, test_displacement_preds = model(test_inputs)

        predicted_force = test_force_preds.squeeze().round()
        true_force = test_force_labels
        predicted_angles = test_angle_preds.squeeze() * 360.0  # Convert back to degrees
        true_angles = test_angle_labels * 360.0  # Convert back to degrees
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
