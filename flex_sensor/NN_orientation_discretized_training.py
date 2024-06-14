import joblib
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from NN_common_functions import load_data
from NN_orientation_discretized import (
    NN_orientation_discretized,
    OrienDataset,
    angular_distance_loss,
)
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, random_split

if __name__ == "__main__":

    # Define the folder containing the CSV files
    data_folder_path = "/home/victor/ws_sensor_combined/src/flex_sensor/data/14-06/"

    # Define the beam discretization parameter
    beam_discretization = 10  # Number of degrees per class, e.g., 1 degree interval
    num_orien_classes = int(360 / beam_discretization)  # Total number of classes

    ### LOAD TRAINING DATA ###
    all_data, all_orientations, all_positions = load_data(data_folder_path)

    ### CREATE DATASET AND DATALOADER ###

    # Normalize the data
    scaler = StandardScaler()
    all_data = scaler.fit_transform(all_data)

    # Save the fitted scaler to a file
    joblib.dump(scaler, data_folder_path + f"{beam_discretization}deg_scaler.pkl")

    # Convert angles to classes based on the beam discretization
    all_orien_classes = (all_orientations / beam_discretization).astype(
        int
    ) % num_orien_classes

    # Create dataset and split into training and validation sets
    dataset = OrienDataset(all_data, all_orien_classes)

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
    model = NN_orientation_discretized(num_classes=num_orien_classes)
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
            loss = criterion(outputs, labels, num_classes=num_orien_classes)
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
                loss = criterion(outputs, labels, num_classes=num_orien_classes)
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
                data_folder_path
                + f"best_model_classification_{beam_discretization}deg.pth",
            )

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_epoch_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}"
            )

    # Load the best model
    model.load_state_dict(
        torch.load(
            data_folder_path + f"best_model_classification_{beam_discretization}deg.pth"
        )
    )

    # Test the model

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        for test_X, labels in test_loader:
            predictions = model(test_X)
            predicted_classes = torch.argmax(predictions, dim=1).cpu().numpy()

    # Convert predicted classes back to angles
    predicted_angles = predicted_classes * beam_discretization
    true_angles = labels * beam_discretization
    # Calculate Mean Absolute Error (MAE)
    mae = mean_absolute_error(true_angles, predicted_angles)

    print("Predictions in degrees:")
    print(predicted_angles)
    print("Labels in degrees: ")
    print(true_angles)
    print(f"Mean Absolute Error (MAE): {mae:.2f} degrees")

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
