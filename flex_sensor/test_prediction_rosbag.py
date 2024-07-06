import joblib
import numpy as np
import rclpy
import torch
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

from .NN_orientation_pos_no_force_CNN import CNN_multi_task, CNN_multi_task_diverge


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


def calculate_differences(data):
    # Calcular diferencias entre pares de lecturas de sensores
    diff_data = []
    num_sensors = data.shape[1]
    for i in range(num_sensors):
        for j in range(i + 1, num_sensors):
            diff_data.append(data[:, i] - data[:, j])
    return np.array(diff_data).T


# CollisionDetectorNode class to handle both models
class CollisionDetectorNode(Node):
    def __init__(self):
        super().__init__("collision_detector_node")

        self.model_type = "CNN_FNN_continuous"  # Change to 'linear_raw', 'linear_differences', 'cnn', 'cnn_diverge', 'fnn', 'knn_raw', 'knn_differences', 'CNN_FNN_continuous'
        self.contact_threshold = (
            0.9  # Threshold from which a collision is detected [0-1]
        )

        data_folder = "/home/victor/ws_sensor_combined/src/flex_sensor/data"
        data_date = "04-07-8pos-5disp"

        model_folder = f"{data_folder}/{data_date}/{self.model_type}"

        # Load the scaler
        scaler_path = f"{model_folder}/scaler.pkl"
        self.scaler = joblib.load(scaler_path)

        self.model = None

        # Load the trained model
        if self.model_type == "CNN_FNN_continuous":
            model_path = f"{model_folder}/model.pth"
            self.model = CNN_multi_task()
        elif self.model_type == "cnn_diverge":
            model_path = f"{model_folder}/best_model_multi_task_cnn_diverge.pth"
            self.model = CNN_multi_task_diverge()
        elif self.model_type == "fnn":
            model_path = f"{model_folder}/best_model_multi_task_ffnn.pth"
            self.model = NN_multi_task()
        elif self.model_type in [
            "linear_raw",
            "linear_differences",
            "knn_raw",
            "knn_differences",
        ]:
            self.model_angle = joblib.load(f"{model_folder}/model_angle.pkl")
            self.model_disp = joblib.load(f"{model_folder}/model_disp.pkl")
            self.model_force = joblib.load(f"{model_folder}/model_force.pkl")
        else:
            raise ValueError(
                "Unsupported model type: use 'linear_raw', 'linear_differences', 'knn_raw', 'knn_differences', 'cnn', 'cnn_diverge', 'fnn', or 'CNN_FNN_continuous'"
            )

        if isinstance(
            self.model, (CNN_multi_task, CNN_multi_task_diverge, NN_multi_task)
        ):
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

        # Calculate differences if required
        if "differences" in self.model_type:
            raw_values = calculate_differences(raw_values)

        # Normalize the raw values using the loaded scaler
        normalized_values = self.scaler.transform(raw_values)

        # Convert to torch tensor if needed
        if isinstance(
            self.model, (CNN_multi_task, CNN_multi_task_diverge, NN_multi_task)
        ):
            input_tensor = torch.tensor(normalized_values, dtype=torch.float32)

            if isinstance(self.model, (CNN_multi_task, CNN_multi_task_diverge)):
                input_tensor = input_tensor.reshape(
                    -1, 1, 2, 2
                )  # Reshape for CNN input

            # Get predictions from the model
            with torch.no_grad():
                force_applied, angle, displacement = self.model(input_tensor)

            # Process the predictions
            contact = force_applied.item() > self.contact_threshold
            angle_value = angle.item() * 360.0  # Convert angle back to degrees
            displacement_value = displacement.item()
        else:
            # Get predictions from sklearn model
            force_applied = self.model_force.predict(normalized_values)
            angle_value = self.model_angle.predict(normalized_values)
            displacement_value = self.model_disp.predict(normalized_values)

            contact = force_applied[0] > self.contact_threshold
            angle_value = angle_value[0] * 360.0  # Convert angle back to degrees
            displacement_value = displacement_value[0]

        print(
            force_applied
            if isinstance(force_applied, torch.Tensor)
            else force_applied[0]
        )

        if contact:
            contact_value = 1.0
        else:
            contact_value = 0.0

        # Adjust angle
        """ angle_value = angle_value - 90
        if angle_value < 0:
            angle_value = 360 + angle_value """

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
