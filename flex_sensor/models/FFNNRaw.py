import os

import matplotlib.pyplot as plt
import numpy as np
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


# Function to normalize data
def normalize_data(data, min_values=None, max_values=None):
    normalized_data = np.zeros_like(data)
    if min_values is None or max_values is None:
        min_values = np.min(data, axis=0)
        max_values = np.max(data, axis=0)
    for i in range(data.shape[1]):
        normalized_data[:, i] = (data[:, i] - min_values[i]) / (
            max_values[i] - min_values[i]
        )
    return normalized_data, min_values, max_values


# Define the PyTorch dataset
class FFNNRawDataset(Dataset):
    def __init__(self, X, angles, displacements, contact):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.angles = torch.tensor(angles, dtype=torch.float32)
        self.displacements = torch.tensor(displacements, dtype=torch.float32)
        self.contact = torch.tensor(contact, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            self.X[idx],
            self.angles[idx],
            self.displacements[idx],
            self.contact[idx],
        )


# Define the neural network for continuous output
class FFNNRaw(nn.Module):
    def __init__(self):
        super(FFNNRaw, self).__init__()
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc_force = nn.Linear(64, 1)
        self.fc_angle = nn.Linear(64, 1)
        self.fc_displacement = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        force_applied = torch.sigmoid(self.fc_force(x))
        angle = self.fc_angle(x)
        displacement = self.fc_displacement(x)
        return force_applied, angle, displacement


if __name__ == "__main__":
    model_type = "FFNNRaw"
    # Define the folder containing the CSV files
    data_folder_path = (
        "/home/victor/ws_sensor_combined/src/flex_sensor/data/09-07-4orient-5pos/"
    )
    center_orientations = [0, 90, 180, 270]
    center_displacements = [0.5, 1.0, 1.5, 2.0, 2.5]

    model_output_path = os.path.join(data_folder_path, model_type)
    os.makedirs(model_output_path, exist_ok=True)

    ### LOAD TRAINING DATA ###
    all_data, all_orientations, all_positions, all_contact = load_data(data_folder_path)

    ### NORMALIZE DATA ###
    """ all_data, min_values, max_values = normalize_data(all_data)

    # Save the min and max values
    normalization_params = {"min_values": min_values, "max_values": max_values}
    joblib.dump(
        normalization_params,
        os.path.join(model_output_path, "normalization_params.pkl"),
    ) """

    # Normalize the data
    scaler = StandardScaler()
    all_data = scaler.fit_transform(all_data)

    """ # Assuming all_data is already a numpy array
    all_data = np.array(all_data)

    # Extracting each sensor's data
    sensor_1 = all_data[:, 0].reshape(-1, 1)
    sensor_2 = all_data[:, 1].reshape(-1, 1)
    sensor_3 = all_data[:, 2].reshape(-1, 1)
    sensor_4 = all_data[:, 3].reshape(-1, 1)

    # Standardizing each sensor's data
    scaler_1 = StandardScaler()
    sensor_1_standardized = scaler_1.fit_transform(sensor_1.reshape(-1, 1))
    joblib.dump(scaler_1, model_output_path + "/scaler_1.pkl")

    scaler_2 = StandardScaler()
    sensor_2_standardized = scaler_2.fit_transform(sensor_2.reshape(-1, 1))
    joblib.dump(scaler_2, model_output_path + "/scaler_2.pkl")

    scaler_3 = StandardScaler()
    sensor_3_standardized = scaler_3.fit_transform(sensor_3.reshape(-1, 1))
    joblib.dump(scaler_3, model_output_path + "/scaler_3.pkl")

    scaler_4 = StandardScaler()
    sensor_4_standardized = scaler_4.fit_transform(sensor_4.reshape(-1, 1))
    joblib.dump(scaler_4, model_output_path + "/scaler_4.pkl")

    # Combining standardized data back into a single array
    all_data = np.hstack((sensor_1_standardized, sensor_2_standardized, sensor_3_standardized, sensor_4_standardized)) """

    # Normalize angles to [0, 1]
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
    model = FFNNRaw()
    criterion = multi_task_loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Store the training and validation losses
    training_losses = []
    validation_losses = []

    # Track the best validation loss
    best_val_loss = float("inf")

    # Train the model
    num_epochs = 10
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

        # Convert PyTorch tensor to NumPy array for compatibility with StandardScaler
        test_inputs_np = test_inputs.numpy()

        # Normalize the test_inputs using the fitted scaler
        test_inputs_normalized_np = scaler.transform(test_inputs_np)

        # Convert the normalized NumPy array back to a PyTorch tensor
        test_inputs_normalized = torch.tensor(
            test_inputs_normalized_np, dtype=torch.float32
        )

        """ # Convert PyTorch tensor to NumPy array for compatibility with StandardScaler
        test_inputs_np = test_inputs.numpy()

        # Extracting each sensor's data from test_inputs
        test_sensor_1 = test_inputs_np[:, 0].reshape(-1, 1)
        test_sensor_2 = test_inputs_np[:, 1].reshape(-1, 1)
        test_sensor_3 = test_inputs_np[:, 2].reshape(-1, 1)
        test_sensor_4 = test_inputs_np[:, 3].reshape(-1, 1)

        # Normalize each sensor's data using the fitted scalers
        test_sensor_1_normalized = scaler_1.transform(test_sensor_1)
        test_sensor_2_normalized = scaler_2.transform(test_sensor_2)
        test_sensor_3_normalized = scaler_3.transform(test_sensor_3)
        test_sensor_4_normalized = scaler_4.transform(test_sensor_4)

        # Combine normalized data back into a single array
        test_inputs_normalized_np = np.hstack((test_sensor_1_normalized, test_sensor_2_normalized, test_sensor_3_normalized, test_sensor_4_normalized))

        # Convert the normalized NumPy array back to a PyTorch tensor
        test_inputs_normalized = torch.tensor(test_inputs_normalized_np, dtype=torch.float32) """

        test_force_preds, test_angle_preds, test_displacement_preds = model(
            test_inputs_normalized
        )

        predicted_force = test_force_preds.squeeze().round()
        true_force = test_force_labels
        predicted_angles = test_angle_preds.squeeze() * 360.0  # Convert back to degrees
        true_angles = test_angle_labels * 360.0  # Convert back to degrees
        for idx, predicted_angle in enumerate(predicted_angles):
            if predicted_angle < 0:
                predicted_angles[idx] += 360.0
            if true_angles[idx] < 0:
                true_angles[idx] += 360.0
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
