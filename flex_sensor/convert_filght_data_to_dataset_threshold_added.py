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
        self.bag_folder = "/media/victor/DATA/rosbag_asta_data/11-07-manual-collisions-for-dataset/rosbag2_2024_07_11-13_43_23"
        self.csv_file_data = self.bag_folder + "/flight_data.csv"

        # Model to predict collision
        self.contact_threshold = 0.5
        self.disp_threshold = 0.0
        self.max_disp_gt = 5
        self.avoid_beginning = 0
        self.avoid_end = float("inf")
        self.model_type = "FFNNRaw_calib_data_32hc_50ep_16h1_32_h2_200_ep"
        self.data_folder = "/home/victor/ws_sensor_combined/src/flex_sensor/data"
        self.data_date = "11-07-8orien-5pos"
        self.previous_samples = 4  # Number of previous samples to include

        self.initial_contact = False
        self.raw_values_buffer = []
        self.angle_gt_buffer = []
        self.uav_pose_buffer = []

        self.values_contact = []
        self.angles_contact = []
        self.displacements_contact = []
        self.contact_idx = 0

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

                if idx == 0:
                    self.glass0_pose = [
                        float(row["glass0_pose_x"]),
                        float(row["glass0_pose_y"]),
                        float(row["glass0_pose_z"]),
                    ]
                    self.glass1_pose = [
                        float(row["glass1_pose_x"]),
                        float(row["glass1_pose_y"]),
                            float(row["glass1_pose_z"]),
                        ]

                if idx > self.avoid_beginning and self.samples_saved < self.avoid_end:
                    self.samples_saved += 1
                    # Get model contact prediction
                    contact = self.get_collision_predicted(raw_values)

                    # Get angle gt based on uav yaw and glass pose
                    angle_gt = calculate_angle_between_uav_yaw_and_glass_normal(
                        uav_q, self.glass0_pose, self.glass1_pose
                    )
                    # Get displacement gt
                    if contact or (self.initial_contact and disp_gt > 0):
                        contact = True
                        if not self.initial_contact:
                            self.initial_contact = True
                            for idx in range(len(self.raw_values_buffer)): 
                                # Delete buffer rows from csv
                                self.delete_last_row_from_csv()

                            for idx in range(len(self.raw_values_buffer)): 
                                if idx == 0: 
                                    self.initial_uav_glass_distance = calculate_3d_distance_to_plane(
                                        self.uav_pose_buffer[idx], self.glass0_pose, self.glass1_pose
                                    )
                                    disp_gt = 0.0
                                    self.save_to_csv(self.raw_values_buffer[idx], True, self.angle_gt_buffer[0], 0)
                                else: 
                                    actual_uav_glass_distance = calculate_3d_distance_to_plane(
                                        self.uav_pose_buffer[idx], self.glass0_pose, self.glass1_pose
                                    )
                                    disp_gt = (
                                        (
                                            self.initial_uav_glass_distance
                                            - actual_uav_glass_distance
                                        )
                                        * 100
                                    )  # meters to cm
                                    if disp_gt > self.max_disp_gt: 
                                        disp_gt = self.max_disp_gt
                                    self.save_to_csv(self.raw_values_buffer[idx], True, self.angle_gt_buffer[idx], disp_gt)

                                self.values_contact.append(self.raw_values_buffer[idx])
                                self.angles_contact.append(self.angle_gt_buffer[idx])
                                self.displacements_contact.append(disp_gt)

                            actual_uav_glass_distance = calculate_3d_distance_to_plane(
                                uav_pose, self.glass0_pose, self.glass1_pose
                            )
                            disp_gt = (
                                abs(
                                    self.initial_uav_glass_distance
                                    - actual_uav_glass_distance
                                )
                                * 100
                            )  # meters to cm

                            if disp_gt>self.max_disp_gt: 
                                disp_gt = self.max_disp_gt

                            self.values_contact.append(raw_values)
                            self.angles_contact.append(angle_gt)
                            self.displacements_contact.append(disp_gt)

                        else:
                            actual_uav_glass_distance = calculate_3d_distance_to_plane(
                                uav_pose, self.glass0_pose, self.glass1_pose
                            )
                            disp_gt = (
                                (
                                    self.initial_uav_glass_distance
                                    - actual_uav_glass_distance
                                )
                                * 100
                            )  # meters to cm

                            if disp_gt > self.max_disp_gt: 
                                disp_gt = self.max_disp_gt

                            if disp_gt >= 0: 
                                self.values_contact.append(raw_values)
                                self.angles_contact.append(angle_gt)
                                self.displacements_contact.append(disp_gt)
                    else:
                        if self.initial_contact: 
                            self.save_contact_event_plots(self.values_contact, self.angles_contact, self.displacements_contact, self.contact_idx)

                            self.contact_idx += 1
                            self.values_contact = []
                            self.angles_contact = []
                            self.displacements_contact = []

                        self.initial_contact = False
                        disp_gt = 0.0
                        self.during_contact = False

                    # If contact is detected, add previous samples from the buffer
                    """ if contact:
                        if len(self.contact_buffer) >= self.previous_samples:
                            for buffered_sample in self.contact_buffer:
                                self.save_to_csv(*buffered_sample)
                        self.contact_buffer = [] """

                    if disp_gt < 0: 
                        contact = False
                        disp_gt = 0
                        # self.initial_contact = False
                    # Save the data to the CSV file
                    self.save_to_csv(raw_values, contact, angle_gt, disp_gt)

                    # Update buffer
                    self.raw_values_buffer.append(raw_values)
                    self.angle_gt_buffer.append(angle_gt)
                    self.uav_pose_buffer.append(uav_pose)
                    if len(self.raw_values_buffer) > self.previous_samples:
                        self.raw_values_buffer.pop(0)
                        self.angle_gt_buffer.pop(0)
                        self.uav_pose_buffer.pop(0)

    def save_to_csv(self, raw_values, contact, angle_gt, disp_gt):
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
    def delete_last_row_from_csv(self): 
        df = pd.read_csv(self.csv_file_dataset)
        df = df.drop(df.index[-1])
        df.to_csv(self.csv_file_dataset, index=False)

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
    
    def save_contact_event_plots(self, raw_values, angle, displacement, contact_idx):
        contact_event_folder = os.path.join(self.bag_folder, "contact_events")
        os.makedirs(contact_event_folder, exist_ok=True)

        # Create the figure and axes
        fig, axs = plt.subplots(3, 1, figsize=(10, 12))
        raw_values = np.array(raw_values)
        axs[0].plot(raw_values[:,0], label="S1", linestyle="-")
        axs[0].plot(raw_values[:,1], label="S2", linestyle="-")
        axs[0].plot(raw_values[:,2], label="S3", linestyle="-")
        axs[0].plot(raw_values[:,3], label="S4", linestyle="-")
        axs[0].set_xlabel("Sample idx")
        axs[0].set_ylabel(f"ADC value")
        axs[0].set_title(f"Flex sensors output")
        axs[0].legend()
        axs[0].grid(True)

        # Angle
        axs[1].plot(angle, label="Angle GT", linestyle="-")
        axs[1].set_xlabel("Sample Index")
        axs[1].set_ylabel(f"Angle (degrees)")
        axs[1].set_title(f"Ground truth angle during contact event")
        axs[1].legend()
        axs[1].grid(True)

        # Displacements
        axs[2].plot(displacement, label="Displacement GT", linestyle="-")
        axs[2].set_xlabel("Sample Index")
        axs[2].set_ylabel(f"Displacement (centimeters)")
        axs[2].set_title(f"Ground truth displacement during contact event")
        axs[2].legend()
        axs[2].grid(True)

        # Save the figure
        plt.tight_layout()
        plt.savefig(os.path.join(contact_event_folder, f"Contact_{contact_idx}.png"))
        plt.close()




if __name__ == "__main__":
    createDataset()
