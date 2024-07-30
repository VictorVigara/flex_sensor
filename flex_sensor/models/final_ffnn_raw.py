import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, mean_absolute_error, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, random_split

""" from common import load_dataset, load_test_dataset, angle_displacement_loss
from result_analysis import displacement_analysis, orientation_analysis """

if __name__ == "__main__":
    from common import load_dataset, load_test_dataset, angle_displacement_loss
    from result_analysis import displacement_analysis, orientation_analysis, contact_analysis
else:
    from common import load_dataset, load_test_dataset, angle_displacement_loss
    from result_analysis import displacement_analysis, orientation_analysis, contact_analysis
       
def split_dataset(dataset, combined_data, batch_size = 32, network_type=None, filtered = False):
    if combined_data == False: 
        train_size = int(0.7 * len(dataset))
        val_test_size = len(dataset) - train_size
        train_dataset, val_test_dataset = random_split(dataset, [train_size, val_test_size])
        val_size = int(0.5 * len(val_test_dataset))
        test_size = len(val_test_dataset) - val_size
        val_dataset, test_dataset = random_split(val_test_dataset, [val_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=test_size, shuffle=False)
    else: 
        train_size = int(0.7 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        all_data_test, all_orientations_test, all_positions_test, contact_test = load_test_dataset(data_folder_path, filtered)
        all_data_test = scaler.transform(all_data_test)

        # Create dataset and split into training and validation sets
        test_dataset = FFNNRawDataset(all_data_test, angles = all_orientations_test, displacements = all_positions_test, contact=contact_test)

        test_size = len(test_dataset)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=test_size, shuffle=False, drop_last=True)

    return train_loader, val_loader, test_loader

# Define the PyTorch dataset
class FFNNRawDataset(Dataset):
    def __init__(self, X, angles=None, displacements=None, contact=None):
        self.X = torch.tensor(X, dtype=torch.float32)
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
        self.contact = (
            torch.tensor(contact, dtype=torch.float32) if contact is not None else None
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # For angle/displacement NN training
        if self.sin_angles is not None and self.cos_angles is not None and self.displacements is not None and self.contact is None:
            return (
                self.X[idx],
                self.sin_angles[idx],
                self.cos_angles[idx],
                self.displacements[idx],
            )
        # For contact NN training
        elif self.sin_angles is None and self.cos_angles is None and self.displacements is None and self.contact is not None:
            return self.X[idx], self.contact[idx]
        
        # For flight evaluation
        elif self.sin_angles is not None and self.cos_angles is not None and self.displacements is not None and self.contact is not None:
            return (
                self.X[idx],
                self.contact[idx],
                self.sin_angles[idx],
                self.cos_angles[idx],
                self.displacements[idx],
            )

# Define the contact detection network
class ContactDetectionNN(nn.Module):
    def __init__(self):
        super(ContactDetectionNN, self).__init__()
        self.fc1 = nn.Linear(4, 16)
        #self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        #x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


# Define the angle and displacement prediction network
class AngleDisplacementNN(nn.Module):
    def __init__(self, input_size, hidden1=32, hidden2=64):
        super(AngleDisplacementNN, self).__init__()
        self.fc1 = nn.Linear(input_size,hidden1)
        #self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc_sin_angle = nn.Linear(hidden1, hidden2)
        self.fc_sin_angle2 = nn.Linear(hidden2, 1)
        self.fc_cos_angle = nn.Linear(hidden1, hidden2)
        self.fc_cos_angle2 = nn.Linear(hidden2, 1)
        self.fc_displacement = nn.Linear(hidden1, hidden2)
        self.fc_displacement2 = nn.Linear(hidden2, 1)
        # Define proportion or neurons to dropout
        self.dropout = nn.Dropout(0.1)

        """ self.fc1 = nn.Linear(4,hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc_sin_angle = nn.Linear(hidden2, 1)
        self.fc_cos_angle = nn.Linear(hidden2, 1)
        self.fc_displacement = nn.Linear(hidden2, 1)
        # Define proportion or neurons to dropout
        self.dropout = nn.Dropout(0.1) """



    def forward(self, x):
        #x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        sin_angle = torch.relu(self.fc_sin_angle(x))
        sin_angle = (self.fc_sin_angle2(sin_angle))
        cos_angle = torch.relu(self.fc_cos_angle(x))
        cos_angle = (self.fc_cos_angle2(cos_angle))
        displacement = torch.relu(self.fc_displacement(x))
        displacement = (self.fc_displacement2(displacement))
        
        return sin_angle, cos_angle, displacement

        """ x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        sin_angle = self.fc_sin_angle(x)
        cos_angle = self.fc_cos_angle(x)
        displacement = self.fc_displacement(x) """


def train_contact_detection_network(data, labels, model_output_path, combined_data, filtered = False):
    model = ContactDetectionNN()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    dataset = FFNNRawDataset(data, contact=labels)
    train_loader, val_loader, _ = split_dataset(dataset, combined_data=combined_data, batch_size= 32, filtered=filtered)
    #train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    all_data_test, all_orientations_test, all_positions_test, contact_test = load_test_dataset(test_data_folder_path)
    all_data_test = scaler.transform(all_data_test)
    test_set = FFNNRawDataset(all_data_test, contact=contact_test)
    test_size = len(test_set)
    test_loader = DataLoader(test_set, batch_size=test_size, shuffle=False, drop_last=True)

    training_losses = []
    validation_losses = []
    # Track the best validation loss
    best_val_loss = float("inf")
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for inputs, contact_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), contact_labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(train_loader)
        training_losses.append(avg_epoch_loss)


        # Evaluate on validation set
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, contact_labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), contact_labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        validation_losses.append(avg_val_loss)

        # Save the model if validation loss is the best we've seen so far
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = model
            torch.save(model.state_dict(), model_output_path + f"/contact_detection_model.pth")

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_epoch_loss:.4f}, Validation Loss: {avg_val_loss:.4f}"
            )


    # Test best model
    model.eval()
    with torch.no_grad():
        if not combined_data:
            inputs, contact_labels = next(iter(test_loader))
        else: 
            inputs, contact_labels, _, _, _ = next(iter(test_loader))
        predicted_contacts = best_model(inputs)

    contact_analysis(contact_labels, predicted_contacts, model_output_path)

    return best_model, test_loader


def train_angle_displacement_network(data, angles, displacements, model_output_path, combined_data, filtered = False, hidden1 = 16, hidden2 = 32):
    model = AngleDisplacementNN(hidden1 = hidden1, hidden2=hidden2, input_size=4)
    criterion = angle_displacement_loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    dataset = FFNNRawDataset(data, angles=angles, displacements=displacements, contact=None)
    train_loader, val_loader, test_loader = split_dataset(dataset, combined_data=combined_data, batch_size= 32, filtered=filtered)

    training_losses = []
    validation_losses = []
    # Track the best validation loss
    best_val_loss = float("inf")
    num_epochs = 300
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for inputs, sin_angle_targets, cos_angle_targets, displacement_targets in train_loader:
            optimizer.zero_grad()
            sin_angle_preds, cos_angle_preds, displacement_preds = model(inputs)
            loss = angle_displacement_loss(
                        sin_angle_preds.squeeze(),
                        sin_angle_targets,
                        cos_angle_preds.squeeze(),
                        cos_angle_targets,
                        displacement_preds.squeeze(),
                        displacement_targets,
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
            for inputs, sin_angle_targets, cos_angle_targets, displacement_targets in val_loader:
                sin_angle_preds, cos_angle_preds, displacement_preds = model(inputs)
                loss = angle_displacement_loss(
                            sin_angle_preds.squeeze(),
                            sin_angle_targets,
                            cos_angle_preds.squeeze(),
                            cos_angle_targets,
                            displacement_preds.squeeze(),
                            displacement_targets,
                ) 
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        validation_losses.append(avg_val_loss)

        # Save the model if validation loss is the best we've seen so far
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = model
            torch.save(model.state_dict(), model_output_path + f"/angle_displacement_model.pth")

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
    plt.title("Training and Validation Loss Over Epochs Angle-displacement NN")
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig(model_output_path + "/training_loss_angle-disp.png")

    return best_model, test_loader


def eval_model(model_output_path, combined_data, model, test_loader=None, test_with_flight_data = False):
    # Load the best model
    """ model = AngleDisplacementNN()
    model.load_state_dict(torch.load(model_output_path + "/model.pth")) """
    #scaler = joblib.load(model_output_path + "/scaler.pkl")

    # Test the model
    model.eval()
    
    with torch.no_grad():
        if not combined_data and not test_with_flight_data:
            inputs, sin_angle_targets, cos_angle_targets, test_displacement_labels = next(iter(test_loader))
        elif combined_data or test_with_flight_data: 
            inputs, test_contact_labels, sin_angle_targets, cos_angle_targets, test_displacement_labels = next(iter(test_loader))
        (
            test_sin_angle_preds,
            test_cos_angle_preds,
            test_displacement_preds,
        ) = model(inputs)

        predicted_sin_angles = test_sin_angle_preds.squeeze()
        predicted_cos_angles = test_cos_angle_preds.squeeze()
        true_sin_angles = sin_angle_targets
        true_cos_angles = cos_angle_targets

        # Convert predicted sin/cos to angles
        predicted_angles = np.array(torch.atan2(predicted_sin_angles, predicted_cos_angles) * (360.0 / (2 * np.pi)))

        for idx, pred_angle in enumerate(predicted_angles):
            if pred_angle.item() < 0:
                predicted_angles[idx] = pred_angle + 360

        true_angles = torch.atan2(true_sin_angles, true_cos_angles) * (360.0 / (2 * np.pi))

        for idx, true_angle in enumerate(true_angles):
            if true_angle.item() < 0:
                true_angles[idx] = true_angle + 360

        predicted_displacements = test_displacement_preds.squeeze()
        true_displacements = test_displacement_labels    
    
    if not combined_data and not test_with_flight_data:
        test_force_labels = np.full(np.array(true_angles).shape[0], True)
        test_inputs = None
    else: 
        test_force_labels = np.array(test_contact_labels)
        test_inputs = np.array(inputs)

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


if __name__ == "__main__":
    
    # Define the folder containing the CSV files
    data_folder_path = "/home/victor/ws_sensor_combined/src/flex_sensor/data/22-07-8orien-5pos"
    center_orientations = [0, 45, 90, 135, 180, 225, 270]
    center_displacements = [0.5, 1.0, 1.5, 2.0, 2.5]
    
    combined_data = False
    test_with_flight_data = True
    test_data_folder_path = "/media/victor/DATA/rosbag_asta_data/25-07-manual-flight-collisions"
    
    filtered = False
    
    train = True
    
    hidden1 = 16
    hidden2 = 32

    model_type = f"Contact_FFNNRaw_16hc_100ep" #f"Angle_Displacement_FFNN_Raw_bs32_{hidden1}h1_{hidden2}_h2_300_ep"

    model_output_path = os.path.join(data_folder_path, model_type)
    os.makedirs(model_output_path, exist_ok=True)

    if train:
        ### LOAD TRAINING DATA ###
        all_data, all_orientations, all_positions, all_contact = load_dataset(data_folder_path, combined_data, filtered)

        ### CREATE DATASET AND DATALOADER ###
        # Normalize the data
        scaler = StandardScaler()
        all_data = scaler.fit_transform(all_data)

        # Save the fitted scaler to a file
        joblib.dump(scaler, model_output_path + "/scaler.pkl")

        # Train Contact Detection Network
        contact_model, contact_test_loader = train_contact_detection_network(all_data, all_contact, model_output_path, combined_data=combined_data, filtered=filtered)

        # Filter data for contact events
        contact_indices = np.where(all_contact > 0)[0]
        contact_data = all_data[contact_indices]
        contact_orientations = all_orientations[contact_indices]
        contact_positions = all_positions[contact_indices]

        # Train Angle and Displacement Prediction Network
        angle_displacement_model, angle_displacement_test_loader= train_angle_displacement_network(contact_data, contact_orientations, contact_positions, model_output_path, combined_data, filtered=filtered, hidden1=hidden1, hidden2=hidden2)

    if not combined_data and test_with_flight_data: 
        scaler_path = f"{model_output_path}/scaler.pkl"
        scaler = joblib.load(scaler_path)
        all_data_test, all_orientations_test, all_positions_test, contact_test = load_test_dataset(test_data_folder_path)
        all_data_test = scaler.transform(all_data_test)

        # Create dataset and split into training and validation sets
        test_dataset = FFNNRawDataset(all_data_test, angles = all_orientations_test, displacements = all_positions_test, contact=contact_test)
        test_size = len(test_dataset)
        angle_displacement_test_loader = DataLoader(test_dataset, batch_size=test_size, shuffle=False, drop_last=True)

    if not train: 
        angle_displacement_model = AngleDisplacementNN(hidden1 = hidden1, hidden2=hidden2, input_size=4)
        angle_displacement_model.load_state_dict(torch.load(model_output_path + "/angle_displacement_model.pth"))
    
    if not train and combined_data:
        angle_displacement_model = AngleDisplacementNN(hidden1 = hidden1, hidden2=hidden2, input_size=4)
        angle_displacement_model.load_state_dict(torch.load(model_output_path + "/angle_displacement_model.pth"))
        scaler_path = f"{model_output_path}/scaler.pkl"
        scaler = joblib.load(scaler_path)
        all_data_test, all_orientations_test, all_positions_test, contact_test = load_test_dataset(data_folder_path)
        all_data_test = scaler.transform(all_data_test)

        # Create dataset and split into training and validation sets
        test_dataset = FFNNRawDataset(all_data_test, angles = all_orientations_test, displacements = all_positions_test, contact=contact_test)
        test_size = len(test_dataset)
        angle_displacement_test_loader = DataLoader(test_dataset, batch_size=test_size, shuffle=False, drop_last=True)

    eval_model(model_output_path, combined_data, angle_displacement_model, angle_displacement_test_loader, test_with_flight_data)