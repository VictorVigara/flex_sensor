import joblib
import numpy as np
import rclpy
import torch
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

from .models.CNN_FFNN_Raw_Diff import CNN_FNN_Raw_Diff
from .models.FFNN_CNN_Raw import CNN_multi_task
from .models.FFNN_Raw_Diff import FFNNRawDiff
from .models.FFNN_raw_sincon import FNNRaw_sincos
from .models.FFNNRaw import FFNNRaw


def calculate_differences(data):
    # Calculate differences between pairs of sensor readings
    diff_data = []
    num_sensors = data.shape[1]
    for i in range(num_sensors):
        for j in range(i + 1, num_sensors):
            diff_data.append(data[:, i] - data[:, j])
    return np.array(diff_data).T


def normalize_data(data, min_values, max_values):
    normalized_data = np.zeros_like(data)
    for i in range(data.shape[1]):
        normalized_data[:, i] = (data[:, i] - min_values[i]) / (
            max_values[i] - min_values[i]
        )
    return normalized_data


# CollisionDetectorNode class to handle both models
class CollisionDetectorNode(Node):
    def __init__(self):
        super().__init__("collision_detector_node")

        self.model_type = (
            "FFNN_CNN_raw"  # 'FFNNRaw', 'linear_raw', 'linear_differences',
        )
        # 'knn_raw', 'knn_differences', 'FFNN_CNN_raw',
        # 'FFNNRawDiff',

        self.contact_threshold = (
            0.9  # Threshold from which a collision is detected [0-1]
        )

        data_folder = "/home/blackbird/uav_forest_ws/src/flex_sensor/data"
        data_date = "04-07-8pos-5disp"

        self.NN_models = (
            CNN_multi_task,
            FFNNRaw,
            FFNNRawDiff,
            FNNRaw_sincos,
            CNN_FNN_Raw_Diff,
        )

        model_folder = f"{data_folder}/{data_date}/{self.model_type}"

        self.raw_scaler = None
        self.diff_scaler = None

        self.model = None

        # Load the trained model
        if self.model_type == "FFNN_CNN_raw":
            model_path = f"{model_folder}/model.pth"
            self.model = CNN_multi_task()
        elif self.model_type == "FFNNRaw":
            model_path = f"{model_folder}/model.pth"
            self.model = FFNNRaw()
        elif self.model_type == "FFNNRaw_sincos":
            model_path = f"{model_folder}/model.pth"
            self.model = FNNRaw_sincos()
        elif self.model_type == "FFNNRawDiff":
            model_path = f"{model_folder}/model.pth"
            self.model = FFNNRawDiff()
            self.raw_scaler = joblib.load(f"{model_folder}/scaler_raw.pkl")
            self.diff_scaler = joblib.load(f"{model_folder}/scaler_diff.pkl")
        elif self.model_type == "CNN_FNN_Raw_Diff":
            model_path = f"{model_folder}/model.pth"
            self.model = CNN_FNN_Raw_Diff()
            self.raw_scaler = joblib.load(f"{model_folder}/scaler_raw.pkl")
            self.diff_scaler = joblib.load(f"{model_folder}/scaler_diff.pkl")
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

        if (
            self.raw_scaler is None
            and self.diff_scaler is None
            and self.model_type != "FFNNRaw"
        ):
            # Load the scaler
            scaler_path = f"{model_folder}/scaler.pkl"
            self.scaler = joblib.load(scaler_path)

        # Load min and max values for normalization
        if self.model_type == "FFNNRaw":
            normalization_params = joblib.load(
                f"{model_folder}/normalization_params.pkl"
            )
            self.min_values = normalization_params["min_values"]
            self.max_values = normalization_params["max_values"]

        if isinstance(self.model, self.NN_models):
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
        print(f"RECIBIENDOOOOOOOO")
        raw_values = np.array(msg.data).reshape(1, -1)

        # Calculate differences if required
        if self.raw_scaler and self.diff_scaler:
            diff_values = calculate_differences(raw_values)
            diff_normalized_values = self.diff_scaler.transform(diff_values)
            raw_normalized_values = calculate_differences(raw_values)
            raw_normalized_values = self.raw_scaler.transform(raw_values)
        elif self.model_type == "FFNNRaw":
            print("Normalizing values")
            raw_normalized_values = normalize_data(
                raw_values, self.min_values, self.max_values
            )
        else:
            # Normalize the raw values using the loaded min and max values
            # raw_normalized_values = normalize_data(raw_values, self.min_values, self.max_values)
            raw_normalized_values = self.scaler.transform(raw_values)

        # Convert to torch tensor if needed
        if (
            isinstance(self.model, self.NN_models)
            and not self.raw_scaler
            and not self.diff_scaler
        ):
            input_tensor = torch.tensor(raw_normalized_values, dtype=torch.float32)

            if "CNN" in self.model_type:
                input_tensor = input_tensor.reshape(
                    -1, 1, 2, 2
                )  # Reshape for CNN input

            if "sincos" in self.model_type:
                with torch.no_grad():
                    (
                        force_applied,
                        angle_sin_preds,
                        angle_cos_preds,
                        displacement,
                    ) = self.model(input_tensor)

                # Process the predictions
                contact = force_applied.item() > self.contact_threshold
                angle_value = float(
                    torch.atan2(angle_sin_preds, angle_cos_preds) * 180 / np.pi
                )
                if angle_value < 0:
                    angle_value = angle_value + 360
                displacement_value = displacement.item()
            else:
                # Get predictions from the model
                with torch.no_grad():
                    force_applied, angle, displacement = self.model(input_tensor)

                # Process the predictions
                contact = force_applied.item() > self.contact_threshold
                angle_value = angle.item() * 360.0  # Convert angle back to degrees
                print(angle_value)
                """ angle_value -= 90 """
                if angle_value > 360:
                    angle_value -= 360

                if angle_value < 0:
                    angle_value = angle_value + 360
                displacement_value = displacement.item()

        elif (
            isinstance(self.model, self.NN_models)
            and self.raw_scaler
            and self.diff_scaler
        ):
            raw_input_tensor = torch.tensor(raw_normalized_values, dtype=torch.float32)
            diff_input_tensor = torch.tensor(
                diff_normalized_values, dtype=torch.float32
            )

            if "CNN" in self.model_type:
                raw_input_tensor = raw_input_tensor.reshape(
                    -1, 1, 2, 2
                )  # Reshape for CNN input

            # Get predictions from the model
            with torch.no_grad():
                force_applied, angle, displacement = self.model(
                    raw_input_tensor, diff_input_tensor
                )

            # Process the predictions
            contact = force_applied.item() > self.contact_threshold
            angle_value = angle.item() * 360.0  # Convert angle back to degrees
            print(f"original output: {angle_value}")
            """ angle_value -= 90 """
            if angle_value > 360:
                angle_value -= 360

            if angle_value < 0:
                angle_value = angle_value + 360
            print(f"Corrected angle: {angle_value}")
            displacement_value = displacement.item()

        else:
            # Get predictions from sklearn model
            force_applied = self.model_force.predict(raw_normalized_values)
            angle_value = self.model_angle.predict(raw_normalized_values)
            displacement_value = self.model_disp.predict(raw_normalized_values)

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
