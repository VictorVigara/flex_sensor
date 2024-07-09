import csv
import os

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Float32MultiArray


def calculate_2d_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def quaternion_to_yaw(quaternion):
    r = R.from_quat(quaternion)
    euler = r.as_euler("zyx", degrees=True)  # Use 'zyx' convention to get yaw
    return euler[0]  # yaw angle in degrees


def calculate_angle_between_centers(uav_pose, uav_orientation, pipe_pose):
    # Calculate the angle of the vector from UAV to pipe in global frame
    vector = np.array(pipe_pose[:2]) - np.array(uav_pose[:2])
    global_angle = np.degrees(np.arctan2(vector[1], vector[0]))

    # Calculate the UAV's yaw angle
    uav_yaw = quaternion_to_yaw(uav_orientation)

    # Pipe yaw is static and assumed to be 0
    pipe_yaw = 0

    # Calculate the relative angle in the UAV's frame
    relative_angle = global_angle - uav_yaw + pipe_yaw
    return (relative_angle + 180) % 360 - 180  # Normalize to [-180, 180]


class SaveContactInfoNode(Node):
    def __init__(self):
        super().__init__("save_contact_info_node")

        data_folder = "/media/victor/DATA/rosbag_asta_data/07-07-recording-nets/asta_net_in_porteria"

        self.csv_file = os.path.join(data_folder, "contact_info.csv")

        # Create the CSV file and write the header
        with open(self.csv_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "raw_value_1",
                    "raw_value_2",
                    "raw_value_3",
                    "raw_value_4",
                    "collision_predicted",
                    "collision_value_predicted" "collision_orientation_predicted",
                    "collision_displacement_predicted",
                    "uav_pose_x",
                    "uav_pose_y",
                    "uav_pose_z",
                    "uav_q_w",
                    "uav_q_x",
                    "uav_q_y",
                    "uav_q_z",
                    "pipe_pose_x",
                    "pipe_pose_y",
                    "pipe_pose_z",
                    "pipe_q_w",
                    "pipe_q_x",
                    "pipe_q_y",
                    "pipe_q_z",
                ]
            )

        # Initialize ROS2 subscribers
        self.create_subscription(
            Float32MultiArray,
            "collision_platform/raw_values",
            self.flex_sensor_cb,
            1,
        )

        self.create_subscription(
            Float32MultiArray,
            "collision_detection",
            self.collision_detection_cb,
            1,
        )

        self.create_subscription(
            PoseStamped,
            "mocap_initial",
            self.drone_mocap_cb,
            1,
        )

        self.create_subscription(
            PoseStamped,
            "pipe_initial",
            self.pipe_mocap_cb,
            1,
        )

        # Initialize data storage variables
        self.raw_values = [None, None, None, None]
        self.uav_pose = [None, None, None]
        self.uav_q_wxyz = [None, None, None, None]
        self.pipe_pose = [None, None, None]
        self.pipe_q_wxyz = [None, None, None, None]
        self.initial_contact = False

    def flex_sensor_cb(self, msg):
        self.raw_values = msg.data[:4]
        print(f"Flex sensor values: {self.raw_values}")

    def collision_detection_cb(self, contact_msg):
        collision_predicted = contact_msg.data[0]
        collision_orientation_predicted = contact_msg.data[1]
        collision_displacement_predicted = contact_msg.data[2]
        collision_value_predicted = contact_msg.data[3]

        # Check if all data is available before saving
        if (
            None not in self.raw_values
            and None not in self.uav_pose
            and None not in self.uav_q_wxyz
            and None not in self.pipe_pose
            and None not in self.pipe_q_wxyz
        ):
            print(f"Predictions: {collision_predicted}")
            # Save the data to the CSV file
            with open(self.csv_file, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        self.raw_values[0],
                        self.raw_values[1],
                        self.raw_values[2],
                        self.raw_values[3],
                        collision_predicted,
                        collision_value_predicted,
                        collision_orientation_predicted,
                        collision_displacement_predicted,
                        self.uav_pose[0],
                        self.uav_pose[1],
                        self.uav_pose[2],
                        self.uav_q_wxyz[0],
                        self.uav_q_wxyz[1],
                        self.uav_q_wxyz[2],
                        self.uav_q_wxyz[3],
                        self.pipe_pose[0],
                        self.pipe_pose[1],
                        self.pipe_pose[2],
                        self.pipe_q_wxyz[0],
                        self.pipe_q_wxyz[1],
                        self.pipe_q_wxyz[2],
                        self.pipe_q_wxyz[3],
                    ]
                )

                """ # Calculate the angle between the UAV's x-axis and the line joining the centers of the UAV and pipe
                actual_angle = calculate_angle_between_centers(self.uav_pose, self.uav_q_wxyz, self.pipe_pose)
                angle_difference = abs(collision_orientation_predicted - actual_angle)
                
                if self.initial_contact is None:
                    self.initial_contact = True
                    initial_uav_pipe_distance = calculate_2d_distance(self.uav_pose, self.pipe_pose)
                else:
                    # Calculate the actual displacement since initial contact
                    actual_uav_pipe_distance = calculate_2d_distance(self.uav_pose, self.pipe_pose)
                    displacement_difference = initial_uav_pipe_distance - actual_uav_pipe_distance """

    def drone_mocap_cb(self, msg):
        self.uav_pose = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        self.uav_q_wxyz = [
            msg.pose.orientation.w,
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
        ]
        print(f"UAV p: {self.uav_pose}")

    def pipe_mocap_cb(self, msg):
        self.pipe_pose = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        self.pipe_q_wxyz = [
            msg.pose.orientation.w,
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
        ]
        print(f"pipe p: {self.pipe_pose}")


def main(args=None):
    rclpy.init(args=args)

    node = SaveContactInfoNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
