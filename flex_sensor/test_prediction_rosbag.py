import joblib
import numpy as np
import rclpy
import torch
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

from .models.final_ffnn_raw import ContactDetectionNN
from .models.final_lstm_windows import RNNModel
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

        self.model_type = "FFNNRaw_sincos_nocont_0grad_64hidden_bs32"  # 'FFNNRaw', 'linear_raw', 'linear_differences',
        # 'knn_raw', 'knn_differences', 'FFNN_CNN_raw',
        # 'FFNNRawDiff',

        #RNN PARAMS 
        self.rnn_h_size = 16
        self.windows_size = 20

        self.s1_w = []
        self.s2_w = []
        self.s3_w = []
        self.s4_w = []
        self.raw_values_window = []

        self.contact_threshold = (
            0.9  # Threshold from which a collision is detected [0-1]
        )

        contact_model_folder =  "/home/blackbird/uav_forest_ws/src/flex_sensor/data/best_models/contact/FFNNRaw_16hc_100ep_16h1_32_h2_300_ep_bs32"
        ang_disp_model_folder = "/home/blackbird/uav_forest_ws/src/flex_sensor/data/best_models/ang_disp/RNN_w20_flight_data_1layers_16hs_32bs_300_ep"

        self.raw_scaler = None
        self.diff_scaler = None

        self.model = None

        # LOAD CONTACT MODEL 
        contact_model_path = f"{contact_model_folder}/contact_detection_model.pth"
        self.contact_model = ContactDetectionNN()
        self.contact_model.load_state_dict(torch.load(contact_model_path))
        self.contact_model.eval()

        contact_scaler_path = f"{contact_model_folder}/scaler.pkl"
        self.contact_scaler = joblib.load(contact_scaler_path)


        # LOAD ANGLE DISPLPACEMENT MODEL
        angle_disp_model_path = f"{ang_disp_model_folder}/angle_displacement_model.pth"
        self.ang_disp_model = RNNModel(input_size=4, hidden_size=self.rnn_h_size, num_layers=1, windows_size=self.windows_size)
        self.ang_disp_model.load_state_dict(torch.load(angle_disp_model_path))
        self.ang_disp_model.eval()
        
        # Load the scaler
        ang_disp_scaler_path = f"{ang_disp_model_folder}/scaler.pkl"
        self.ang_disp_scaler = joblib.load(ang_disp_scaler_path)
        

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

        # Scale values for contact model
        contact_values = self.contact_scaler.transform(raw_values)
        contact_input = torch.tensor(contact_values, dtype=torch.float32)

        # Prepare RNN values
        windows_ready = False
        if len(self.s1_w) == self.windows_size:
            self.s1_w.pop(0)
            self.s1_w.append(raw_values[0][0])
            self.s2_w.pop(0)
            self.s2_w.append(raw_values[0][1])
            self.s3_w.pop(0)
            self.s3_w.append(raw_values[0][2])
            self.s4_w.pop(0)
            self.s4_w.append(raw_values[0][3])

            self.raw_values_window.pop(0)
            self.raw_values_window.append([raw_values[0][0], raw_values[0][1], raw_values[0][2], raw_values[0][3]])

            raw_values_windows = [self.s1_w, self.s2_w, self.s3_w, self.s4_w]
            windows_array = np.array(self.raw_values_window)
            #windows_array = windows_array.reshape(windows_array.shape[1], windows_array.shape[0])
            windows_scaled = self.ang_disp_scaler.transform(windows_array)
            windows_tensor = torch.tensor(windows_scaled, dtype=torch.float32).unsqueeze(0)
            windows_ready = True

        else: 
            self.s1_w.append(raw_values[0][0])
            self.s2_w.append(raw_values[0][1])
            self.s3_w.append(raw_values[0][2])
            self.s4_w.append(raw_values[0][3])

            self.raw_values_window.append([raw_values[0][0], raw_values[0][1], raw_values[0][2], raw_values[0][3]])


        if windows_ready:
            with torch.no_grad():
                contact_predicted = self.contact_model(contact_input)
                sin_angle_preds, cos_angle_preds, displacement_pred = self.ang_disp_model(windows_tensor)

            contact = contact_predicted.item() > self.contact_threshold
            if contact:
                contact_value = 1.0
            else:
                contact_value = 0.0
            
            predicted_sin_angles = sin_angle_preds.squeeze()
            predicted_cos_angles = cos_angle_preds.squeeze()

            # Convert predicted sin/cos to angles
            predicted_angle = np.array(torch.atan2(predicted_sin_angles, predicted_cos_angles) * (360.0 / (2 * np.pi)))

            if predicted_angle.item() < 0:
                predicted_angle = predicted_angle.item() + 360
            else:
                predicted_angle = predicted_angle.item()

            displacement_pred = displacement_pred.item()

            if contact: 
                print(f"Contact angle:{predicted_angle} / disp: {displacement_pred}")

            # Publish collision information
            self.collision_msg = Float32MultiArray()
            self.collision_msg.data = [
                contact_value,
                predicted_angle,
                displacement_pred,
            ]
            self.collision_publisher.publish(self.collision_msg)
        
        """ # Process the predictions
        contact = force_applied.item() > self.contact_threshold
        angle_value = float(
            torch.atan2(angle_sin_preds, angle_cos_preds) * 180 / np.pi
        )
        if angle_value < 0:
            angle_value = angle_value + 360
        displacement_value = displacement.item() """
            

        """ if contact:
            contact_value = 1.0
        else:
            contact_value = 0.0

        # Publish collision information
        self.collision_msg = Float32MultiArray()
        self.collision_msg.data = [
            contact_value,
            angle_value,
            displacement_value,
            force_applied.item(),
        ]
        self.collision_publisher.publish(self.collision_msg)
 """

def main(args=None):
    rclpy.init(args=args)

    node = CollisionDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
