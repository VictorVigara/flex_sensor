import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, random_split

if __name__ == "__main__":
    from common import load_dataset, load_test_dataset, angle_displacement_loss
    from result_analysis import displacement_analysis, orientation_analysis
else:
    from .common import load_dataset, load_test_dataset, angle_displacement_loss
    from .result_analysis import displacement_analysis, orientation_analysis

def split_dataset(dataset, batch_size=32):
    train_size = int(0.7 * len(dataset))
    val_test_size = len(dataset) - train_size
    train_dataset, val_test_dataset = random_split(dataset, [train_size, val_test_size])
    val_size = int(0.5 * len(val_test_dataset))
    test_size = len(val_test_dataset) - val_size
    val_dataset, test_dataset = random_split(val_test_dataset, [val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=test_size, shuffle=False)

    return train_loader, val_loader, test_loader

class TimeSeriesDataset(Dataset):
    def __init__(self, windows, angles, displacements):
        self.windows = torch.tensor(windows, dtype=torch.float32)
        self.sin_angles = (
            torch.tensor(np.sin(angles * np.pi/180), dtype=torch.float32)
            if angles is not None
            else None
        )
        self.cos_angles = (
            torch.tensor(np.cos(angles * np.pi/180), dtype=torch.float32)
            if angles is not None
            else None
        )
        self.displacements = (
            torch.tensor(displacements, dtype=torch.float32)
            if displacements is not None
            else None
        )

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return ( 
            self.windows[idx],
            self.sin_angles[idx], 
            self.cos_angles[idx], 
            self.displacements[idx], 
        )

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, windows_size):
        super(RNNModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc_sin_angle = nn.Linear(hidden_size*windows_size, hidden_size)
        self.fc_sin_angle2 = nn.Linear(hidden_size, 1)
        self.fc_cos_angle = nn.Linear(hidden_size*windows_size, hidden_size)
        self.fc_cos_angle2 = nn.Linear(hidden_size, 1)
        self.fc_displacement = nn.Linear(hidden_size*windows_size, hidden_size)
        self.fc_displacement2 = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(0.2)

        self.hidden_size = hidden_size

    def forward(self, x):
        x, _ = self.lstm(x)
        batch_size, sequence_length, hidden_size = x.size()
        x = x.contiguous().view(batch_size, -1)  # Flatten the LSTM output
        #x = x.contiguous().view(32, -1)
        #x = x[:, -1, :]  # Take the output of the last time step
        sin_angle = torch.relu(self.fc_sin_angle(x))
        sin_angle = (self.fc_sin_angle2(sin_angle))
        cos_angle = torch.relu(self.fc_cos_angle(x))
        cos_angle = (self.fc_cos_angle2(cos_angle))
        displacement = torch.relu(self.fc_displacement(x))
        displacement = (self.fc_displacement2(displacement))
        return sin_angle, cos_angle, displacement

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, model_output_folder):
    best_val_loss = float("inf")
    training_losses = []
    validation_losses = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        model.train()
        for windows, sin_angle_targets, cos_angle_targets, displacement_targets in train_loader:
            windows, sin_angle_targets, cos_angle_targets, displacement_targets = windows.to(device), sin_angle_targets.to(device), cos_angle_targets.to(device), displacement_targets.to(device)
            optimizer.zero_grad()
            sin_angle_preds, cos_angle_preds, displacement_preds = model(windows)
            loss = criterion(sin_angle_preds.squeeze(),
                        sin_angle_targets,
                        cos_angle_preds.squeeze(),
                        cos_angle_targets,
                        displacement_preds.squeeze(),
                        displacement_targets,)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_epoch_loss = epoch_loss / len(train_loader)
        training_losses.append(avg_epoch_loss)

        val_loss = evaluate_model(model, val_loader, criterion)
        validation_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_epoch_loss:.4f}, Validation Loss: {val_loss:.4f}"
            )

        #print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}')
    
    # Plot the training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(training_losses, label="Training Loss")
    plt.plot(validation_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Over Epochs Angle-displacement NN")
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig(model_output_folder + "/training_loss_angle-disp.png")

    return best_model

def evaluate_model(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for windows, sin_angle_targets, cos_angle_targets, displacement_targets in data_loader:
            windows, sin_angle_targets, cos_angle_targets, displacement_targets = windows.to(device), sin_angle_targets.to(device), cos_angle_targets.to(device), displacement_targets.to(device)
            sin_angle_preds, cos_angle_preds, displacement_preds = model(windows)
            loss = criterion(sin_angle_preds.squeeze(),
                        sin_angle_targets,
                        cos_angle_preds.squeeze(),
                        cos_angle_targets,
                        displacement_preds.squeeze(),
                        displacement_targets,)
            total_loss += loss.item()
    return total_loss / len(data_loader)

def test_model(model, data_loader, center_orientations, center_displacements, data_folder_path, model_type, contacts): 
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for windows, sin_angle_targets, cos_angle_targets, test_displacement_labels in data_loader:
            windows, sin_angle_targets, cos_angle_targets, test_displacement_labels = windows.to(device), sin_angle_targets.to(device), cos_angle_targets.to(device), test_displacement_labels.to(device)
            test_sin_angle_preds, test_cos_angle_preds, test_displacement_preds = model(windows)
    predicted_sin_angles = test_sin_angle_preds.cpu().squeeze()
    predicted_cos_angles = test_cos_angle_preds.cpu().squeeze()
    true_sin_angles = sin_angle_targets.cpu()
    true_cos_angles = cos_angle_targets.cpu()

    # Convert predicted sin/cos to angles
    predicted_angles = np.array(torch.atan2(predicted_sin_angles, predicted_cos_angles) * (360.0 / (2 * np.pi)))

    for idx, pred_angle in enumerate(predicted_angles):
        if pred_angle.item() < 0:
            predicted_angles[idx] = pred_angle + 360

    true_angles = torch.atan2(true_sin_angles, true_cos_angles) * (360.0 / (2 * np.pi))

    for idx, true_angle in enumerate(true_angles):
        if true_angle.item() < 0:
            true_angles[idx] = true_angle + 360

    predicted_displacements = test_displacement_preds.cpu().squeeze()
    true_displacements = test_displacement_labels.cpu()

    # Calculate Mean Absolute Error (MAE)
    angle_mae = mean_absolute_error(true_angles, predicted_angles)
    displacement_mae = mean_absolute_error(true_displacements, predicted_displacements)

    print(f"Angle MAE: {angle_mae:.2f} degrees")
    print(f"Displacement MAE: {displacement_mae:.2f} cm")

    test_inputs = np.array(windows[:, -1, :].cpu())
    test_force_labels = contacts

    orientation_analysis(
        true_angles, predicted_angles, center_orientations, data_folder_path, model_type, test_force_labels, test_inputs
    )
    displacement_analysis(
        true_displacements,
        predicted_displacements,
        center_displacements,
        data_folder_path,
        model_type,
        test_force_labels, 
        test_inputs
    )

def main():
    data_folder_path = "/media/victor/DATA/rosbag_asta_data/25-07-manual-flight-collisions"
    

    num_epochs = 350
    batch_size =32
    learning_rate = 0.001
    hidden_size = 32
    num_layers = 1

    model_type = f"RNN_w30_flight_data_{num_layers}layers_{hidden_size}hs_{batch_size}bs_{num_epochs}_ep"

    center_orientations = [0, 45, 90, 135, 180, 225, 270]
    center_displacements = [0.5, 1.0, 1.5, 2.0, 2.5]

    model_output_folder = data_folder_path + "/" + model_type
    os.makedirs(model_output_folder, exist_ok=True)

    # Load training dataset
    windows = np.load(os.path.join(data_folder_path, "windows.npy"), allow_pickle=True)
    labels = np.load(os.path.join(data_folder_path, "labels.npy"), allow_pickle=True)

    # Select only contact windows
    contact_mask = labels[:, 0] > 0  # Assuming the first label is contact
    windows = windows[contact_mask]
    angles = np.array(labels[contact_mask][:, 1]).astype(float)  # Select only angle and displacement
    displacements = np.array(labels[contact_mask][:, 2]).astype(float)

    # Scale the windows
    scaler = StandardScaler()
    windows_shape = windows.shape
    windows = windows.reshape(-1, windows_shape[-1])  # Reshape to 2D for scaling
    windows = scaler.fit_transform(windows)
    windows = windows.reshape(windows_shape)  # Reshape back to original shape
    joblib.dump(scaler, os.path.join(model_output_folder, "scaler.pkl"))

    dataset = TimeSeriesDataset(windows, angles, displacements)

    # Split dataset into training (70%), validation (15%), and test (15%)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=test_size, shuffle=False)

    model = RNNModel(input_size=windows.shape[2], hidden_size=hidden_size, num_layers=num_layers, windows_size=windows_shape[1]).to(device)
    criterion = angle_displacement_loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, model_output_folder)
    torch.save(best_model.state_dict(), os.path.join(model_output_folder, "angle_displacement_model.pth"))

    # Load test dataset
    test_windows = np.load(os.path.join(data_folder_path, "test_windows.npy"), allow_pickle=True)
    test_labels = np.load(os.path.join(data_folder_path, "test_labels.npy"), allow_pickle=True)

    # Select only contact windows
    test_windows = test_windows
    test_angles = np.array(test_labels[:, 1]).astype(float)  # Select only angle and displacement
    test_displacements = np.array(test_labels[:, 2]).astype(float)
    test_contacts = test_labels[:, 0]

    test_windows_shape = test_windows.shape
    test_windows = test_windows.reshape(-1, test_windows_shape[-1])  # Reshape to 2D for scaling
    test_windows = scaler.transform(test_windows)
    test_windows = test_windows.reshape(test_windows_shape)  # Reshape back to original shape

    # Create dataset and DataLoader for test set
    test_dataset = TimeSeriesDataset(test_windows, test_angles, test_displacements)
    test_loader = DataLoader(test_dataset, batch_size=test_windows_shape[0], shuffle=False, drop_last=True)

    # Evaluate on test dataset
    test_model(best_model, test_loader, center_orientations, center_displacements, data_folder_path, model_type, test_contacts)

if __name__ == "__main__":
    device = torch.device('cpu')
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main()
