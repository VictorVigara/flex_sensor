import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


# Define the PyTorch dataset
class OrienDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Define the neural network for classification
class NN_orientation_discretized(nn.Module):
    def __init__(self, num_classes):
        super(NN_orientation_discretized, self).__init__()
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
def angular_distance_loss(preds, targets, num_classes):
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
