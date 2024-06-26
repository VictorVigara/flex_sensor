import joblib
import numpy as np
import rclpy
import torch
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

from .NN_orientation_pos_no_force_CNN import CNN_multi_task


# Define the fully connected neural network for multi-task learning
class NN_multi_task(torch.nn.Module):
    def __init__(self):
        super(NN_multi_task, self).__init__()
        self.fc1 = torch.nn.Linear(4, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, 64)

        self.fc_force = torch.nn.Linear(
            64, 1
        )  # Binary classification for force applied
        self.fc_angle = torch.nn.Linear(64, 1)  # Continuous output for angle
        self.fc_displacement = torch.nn.Linear(
            64, 1
        )  # Continuous output for displacement

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))

        force_applied = torch.sigmoid(self.fc_force(x))
        angle = self.fc_angle(x)
        displacement = self.fc_displacement(x)

        return force_applied, angle, displacement


# CollisionDetectorNode class to handle both models
class CollisionDetectorNode(Node):
    def __init__(self):
        super().__init__("collision_detector_node")

        self.model_type = "cnn"
        self.contact_threshold = (
            0.9  # Threshold from which a collision is detected [0-1]
        )

        # Load the scaler
        scaler_path = "/home/victor/ws_sensor_combined/src/flex_sensor/data/17-06-4positions/scaler.pkl"
        self.scaler = joblib.load(scaler_path)

        # Load the trained model
        if self.model_type == "cnn":
            model_path = "/home/victor/ws_sensor_combined/src/flex_sensor/data/17-06-4positions/best_model_multi_task_cnn.pth"
            self.model = CNN_multi_task()
        else:
            model_path = "/home/victor/ws_sensor_combined/src/flex_sensor/data/17-06-4positions/best_model_multi_task_ffnn.pth"
            self.model = NN_multi_task()

        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        # Initialize ROS2 subscribers and publishers
        self.subscription = self.create_subscription(
            Float32MultiArray,
            "collision_platform/raw_values",
            self.listener_callback,
            10,
        )

        self.collision_publisher = self.create_publisher(
            Float32MultiArray, "collision_detection", 10
        )

    def listener_callback(self, msg):
        raw_values = np.array(msg.data).reshape(1, -1)

        # Normalize the raw values using the loaded scaler
        normalized_values = self.scaler.transform(raw_values)

        # Convert to torch tensor
        input_tensor = torch.tensor(normalized_values, dtype=torch.float32)

        if isinstance(self.model, CNN_multi_task):
            input_tensor = input_tensor.reshape(-1, 1, 2, 2)  # Reshape for CNN input

        # Get predictions from the model
        with torch.no_grad():
            force_applied, angle, displacement = self.model(input_tensor)

        # Process the predictions
        contact = force_applied.item() > self.contact_threshold
        print(force_applied.item())
        if contact:
            contact_value = 1.0
        else:
            contact_value = 0.0

        angle_value = angle.item() * 360.0  # Convert angle back to degrees
        displacement_value = displacement.item()

        # Publish collision information
        self.collision_msg = Float32MultiArray()
        self.collision_msg.data = [contact_value, angle_value, displacement_value]
        self.collision_publisher.publish(self.collision_msg)


def main(args=None):
    rclpy.init(args=args)

    node = CollisionDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
