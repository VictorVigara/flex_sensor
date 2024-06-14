import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


# Define the PyTorch dataset
class OrienPosDataset(Dataset):
    def __init__(self, X, y_orientation, y_position):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_orientation = torch.tensor(y_orientation, dtype=torch.long)
        self.y_position = torch.tensor(y_position, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_orientation[idx], self.y_position[idx]


# Define the neural network for classification
class NN_orient_pos_discretized(nn.Module):
    def __init__(self, num_orientation_classes, num_position_classes):
        super(NN_orient_pos_discretized, self).__init__()
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc_orientation = nn.Linear(
            64, num_orientation_classes
        )  # Output classes based on beam discretization
        self.fc_position = nn.Linear(
            64, num_position_classes
        )  # Number of position classes

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        orientation_out = self.fc_orientation(x)
        position_out = self.fc_position(x)
        return orientation_out, position_out


# Custom Angular Distance Loss Function
def angular_pos_loss(
    pred_orientation, pred_position, target_orientation, target_position
):
    # Cross-Entropy Loss for classification
    ce_loss = F.cross_entropy(pred_orientation, target_orientation)

    # Cross-Entropy Loss for position classification
    ce_loss_position = F.cross_entropy(pred_position, target_position)

    # # Calculate the predicted and true angles in radians, ignoring "no force" class
    # mask = target_orientation != num_classes  # Mask for non-"no force" classes

    # if mask.sum() > 0:  # Only compute angular distance if there are non-"no force" samples
    #     pred_angles = torch.argmax(pred_orientation[:, :-1], dim=1).float() * (2 * np.pi / num_classes)
    #     true_angles = target_orientation.float() * (2 * np.pi / num_classes)

    #     # Compute the circular distance only for the angle classes (not the "no force" class)
    #     angular_distance = torch.min(torch.abs(pred_angles[mask] - true_angles[mask]), 2 * np.pi - torch.abs(pred_angles[mask] - true_angles[mask]))
    #     angular_distance_mean = torch.mean(angular_distance)
    # else:
    #     angular_distance_mean = 0.0

    distance_penalty = 0
    angular_distance_mean = 0

    # Combine losses with a penalty for the distance between position classes
    distance_penalty = torch.abs(
        target_position.float() - torch.argmax(pred_position, dim=1).float()
    )
    combined_loss = (
        ce_loss
        + 0.1 * angular_distance_mean
        + ce_loss_position
        + 0.1 * torch.mean(distance_penalty)
    )

    return combined_loss
