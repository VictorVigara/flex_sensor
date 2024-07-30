import csv
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from models.FFNN_CNN_Raw import CNN_multi_task
from save_bag_contact_data import (
    calculate_3d_distance_to_plane,
    calculate_angle_between_uav_yaw_and_glass_normal,
)
from models.final_ffnn_raw import ContactDetectionNN, AngleDisplacementNN


class createDataset:
    def __init__(self) -> None:
        # Data to convert
        self.bag_folder = "/media/victor/DATA/rosbag_asta_data/11-07-manual-collisions-for-dataset/rosbag2_2024_07_11-14_01_34"
        self.csv_file_data = self.bag_folder + "/flight_data.csv"

        # Model to predict collision
        self.contact_threshold = 0.99
        self.disp_threshold = 0.0
        self.avoid_beginning = 0
        self.avoid_end = float("inf")
        self.model_type = "FFNNRaw_calib_data_32hc_50ep_16h1_32_h2_200_ep"
        self.data_folder = "/home/victor/ws_sensor_combined/src/flex_sensor/data"
        self.data_date = "11-07-8orien-5pos"

        self.initial_contact = False

        self.samples_saved = 0

        # Load model to predict collision
        self.load_model()
        # Create dataset file
        self.create_dataset_csv()
        # Fill dataset file
        self.read_flight_info()
        # Create plots
        self.create_plots()

    def create_dataset_csv(self):
        self.csv_file_dataset = os.path.join(self.bag_folder, f"flight_dataset.csv")
        # print(f"csv path: {self.csv_file_dataset}")
        with open(self.csv_file_dataset, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "raw_value_1",
                    "raw_value_2",
                    "raw_value_3",
                    "raw_value_4",
                    "collision_predicted",
                    "angle_gt",
                    "displacement_gt",
                ]
            )

    def load_model(self):
        model_folder = f"{self.data_folder}/{self.data_date}/{self.model_type}"

        # Prepare model to get collisions
        model_path = f"{model_folder}/model.pth"

        if "CNN" in self.model_type:
            self.model = CNN_multi_task()
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()

        elif "FFNNRaw" in self.model_type:
            self.model_contact = ContactDetectionNN()
            self.model_angle_displacement = AngleDisplacementNN(input_size=4, hidden1=16, hidden2=32)
            self.model_contact.load_state_dict(torch.load(f"{model_folder}/contact_detection_model.pth"))
            self.model_angle_displacement.load_state_dict(torch.load(f"{model_folder}/angle_displacement_model.pth"))
        
        # Load the scaler
        scaler_path = f"{model_folder}/scaler.pkl"
        self.scaler = joblib.load(scaler_path)

    def read_flight_info(self):
        with open(self.csv_file_data, mode="r") as file:
            reader = csv.DictReader(file)
            for idx, row in enumerate(reader):
                raw_values = [
                    float(row["raw_value_1"]),
                    float(row["raw_value_2"]),
                    float(row["raw_value_3"]),
                    float(row["raw_value_4"]),
                ]
                uav_pose = [
                    float(row["uav_pose_x"]),
                    float(row["uav_pose_y"]),
                    float(row["uav_pose_z"]),
                ]
                uav_q = [
                    float(row["uav_q_x"]),
                    float(row["uav_q_y"]),
                    float(row["uav_q_z"]),
                    float(row["uav_q_w"]),
                ]
                glass0_pose = [
                    float(row["glass0_pose_x"]),
                    float(row["glass0_pose_y"]),
                    float(row["glass0_pose_z"]),
                ]
                glass1_pose = [
                    float(row["glass1_pose_x"]),
                    float(row["glass1_pose_y"]),
                    float(row["glass1_pose_z"]),
                ]

                # Get model contact prediction
                contact = self.get_collision_predicted(raw_values)

                # Get angle gt based on uav yaw and glass pose
                angle_gt = calculate_angle_between_uav_yaw_and_glass_normal(
                    uav_q, glass0_pose, glass1_pose
                )

                if idx > self.avoid_beginning and self.samples_saved < self.avoid_end:
                    self.samples_saved += 1
                # Get displacement gt
                    if contact:
                        # print(f"Angle gt: {angle_gt}")
                        # print(f"uavq: {uav_q}, g0: {glass0_pose}, g1: {glass1_pose}")
                        # print(f"Initial collision: {self.initial_contact}")
                        if not self.initial_contact:
                            # print(f"Initializing contact: ")
                            self.initial_contact = True
                            self.initial_uav_glass_distance = (
                                calculate_3d_distance_to_plane(
                                    uav_pose, glass0_pose, glass1_pose
                                )
                            )
                            # print(
                            #     f"initial_uav_glass_distance: {self.initial_uav_glass_distance}"
                            # )
                            disp_gt = 0.0
                        else:
                            # print(f"Calculating displacement: ")
                            # Calculate the actual displacement since initial contact
                            actual_uav_glass_distance = calculate_3d_distance_to_plane(
                                uav_pose, glass0_pose, glass1_pose
                            )
                            # print(f"actual distance: {actual_uav_glass_distance}")
                            disp_gt = (
                                abs(
                                    self.initial_uav_glass_distance
                                    - actual_uav_glass_distance
                                )
                                * 100
                            )  # meters to cm
                            # print(f"Current displacement: {disp_gt}")
                    else:
                        self.initial_contact = False
                        disp_gt = 0.0

                    if contact and disp_gt < self.disp_threshold:
                        contact = False

                    # if contact and disp_gt > 4:
                    #     disp_gt = 3.5

                    if not contact:
                        disp_gt = 0.0

                    # Save the data to the CSV file
                    with open(self.csv_file_dataset, mode="a", newline="") as file:
                        writer = csv.writer(file)
                        writer.writerow(
                            [
                                raw_values[0],
                                raw_values[1],
                                raw_values[2],
                                raw_values[3],
                                contact,
                                angle_gt,
                                disp_gt,
                            ]
                        )

    def create_plots(self):
        # Read the generated dataset
        data = pd.read_csv(self.csv_file_dataset)

        # Create subplots
        fig, axs = plt.subplots(4, 1, figsize=(15, 20), sharex=True)

        # Plot flex sensor raw values
        axs[0].plot(data.index, data["raw_value_1"], label="Sensor 1")
        axs[0].plot(data.index, data["raw_value_2"], label="Sensor 2")
        axs[0].plot(data.index, data["raw_value_3"], label="Sensor 3")
        axs[0].plot(data.index, data["raw_value_4"], label="Sensor 4")
        axs[0].set_ylabel("Raw Values")
        axs[0].set_title("Flex Sensor Raw Values")
        axs[0].legend()

        # Plot contact (0, 1)
        axs[1].plot(
            data.index, data["collision_predicted"], label="Contact", color="tab:orange"
        )
        axs[1].set_ylabel("Contact")
        axs[1].set_title("Contact Prediction (0, 1)")
        axs[1].legend()

        # Plot angle_gt
        axs[2].plot(data.index, data["angle_gt"], label="Angle GT", color="tab:green")
        axs[2].set_ylabel("Angle GT (degrees)")
        axs[2].set_title("Ground Truth Angle (degrees)")
        axs[2].legend()

        # Plot disp_gt
        axs[3].plot(
            data.index,
            data["displacement_gt"],
            label="Displacement GT",
            color="tab:red",
        )
        axs[3].set_ylabel("Displacement GT (cm)")
        axs[3].set_title("Ground Truth Displacement (cm)")
        axs[3].legend()

        # Set common x-label
        axs[-1].set_xlabel("Time Steps")

        # Adjust layout
        plt.tight_layout()

        # Save the plot to a file
        plot_path = os.path.join(self.bag_folder, "flight_dataset_plots.png")
        plt.savefig(plot_path)

        # Display the plot
        plt.show()

    def get_collision_predicted(self, raw_values):

        if "CNN" in self.model_type:
            raw_values = np.array(raw_values).reshape(1, -1)
            raw_normalized_values = self.scaler.transform(raw_values)
            input_tensor = torch.tensor(raw_normalized_values, dtype=torch.float32)
            input_tensor = input_tensor.reshape(-1, 1, 2, 2)  # Reshape for CNN input
        elif "FFNNRaw" in self.model_type:
            input_tensor = torch.tensor(self.scaler.transform(np.array(raw_values).reshape(1, -1)), dtype=torch.float32)
        
        with torch.no_grad():
            if "CNN" in self.model_type:
                force_applied, angle, displacement = self.model(input_tensor)
            elif "FFNNRaw" in self.model_type: 
                force_applied = self.model_contact(input_tensor)
                
                test_sin_angle_preds,test_cos_angle_preds, displacement= self.model_angle_displacement(input_tensor)
                predicted_sin_angles = test_sin_angle_preds.squeeze()
                predicted_cos_angles = test_cos_angle_preds.squeeze()
                # Convert predicted sin/cos to angles
                pred_angle = torch.atan2(predicted_sin_angles, predicted_cos_angles) * (360.0 / (2 * np.pi))
                pred_angle = pred_angle.item()

                if pred_angle < 0:
                    pred_angle = pred_angle + 360

        # Process the predictions
        contact = force_applied.item() > self.contact_threshold
        disp = displacement.item()

        """ if contact: 
            contact = True
        elif disp > 0.1:
            contact = True """

        return contact


if __name__ == "__main__":
    createDataset()
