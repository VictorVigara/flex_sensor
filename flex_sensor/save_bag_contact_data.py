import csv
import os

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Float32, Float32MultiArray

if __name__ == "__main__": 
    from .models.FFNN_CNN_Raw import CNN_multi_task
else: 
    from .models.FFNN_CNN_Raw import CNN_multi_task



def calculate_3d_distance_to_plane(point, plane_point1, plane_point2):
    # Calculate the plane normal
    glass_vector = np.array(plane_point2) - np.array(plane_point1)
    plane_normal = np.cross(glass_vector, [0, 0, 1])
    plane_normal /= np.linalg.norm(plane_normal)

    # Calculate the distance from the point to the plane
    point_vector = np.array(point) - np.array(plane_point1)
    distance = np.dot(point_vector, plane_normal)

    return np.abs(distance)


def calculate_glass_normal(glass0_pose, glass1_pose):
    glass_vector = np.array(glass1_pose)[:2] - np.array(glass0_pose)[:2]  # x-y plane
    normal_vector = np.array(
        [-glass_vector[1], glass_vector[0]]
    )  # Perpendicular in the x-y plane
    normal_vector /= np.linalg.norm(normal_vector)  # Normalize the vector
    return normal_vector


def quaternion_to_yaw(quaternion):
    r = R.from_quat(quaternion)
    euler = r.as_euler("zyx", degrees=True)  # Use 'zyx' convention to get yaw
    return euler[0]  # yaw angle in degrees


def calculate_angle_between_uav_yaw_and_glass_normal(
    uav_orientation, glass0_pose, glass1_pose
):
    glass_normal = calculate_glass_normal(glass0_pose, glass1_pose)
    # print(f"Glass normal: {glass_normal}")

    uav_yaw = quaternion_to_yaw(uav_orientation)
    # print(f"UAV yaw: {uav_yaw}")

    uav_yaw_vector = np.array(
        [np.cos(np.radians(uav_yaw)), np.sin(np.radians(uav_yaw))]
    )

    # Calculate the angle between the glass normal and the UAV yaw vector
    angle = np.degrees(
        np.arctan2(glass_normal[1], glass_normal[0])
        - np.arctan2(uav_yaw_vector[1], uav_yaw_vector[0])
    )

    if angle < 0:
        angle += 360

    return angle


def calculate_2d_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def calculate_angle_error(angulo1, angulo2):
    # Convertir los ángulos a radianes
    angulo1 = np.deg2rad(angulo1)
    angulo2 = np.deg2rad(angulo2)

    # Calcular la diferencia en radianes
    diff = angulo1 - angulo2

    # Ajustar la diferencia para que esté en el rango [-pi, pi]
    diff = (diff + np.pi) % (2 * np.pi) - np.pi

    # Convertir la diferencia de vuelta a grados
    diff = np.rad2deg(diff)

    return np.abs(diff)


def angle_between_pipe_uav_centers(uav_pose, uav_orientation, pipe_pose):
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


class saveMode:
    DATASET: "dataset"
    MISSION: "mission"


class obstacle:
    PIPE: "pipe"
    GLASS: "glass"


class SaveContactInfoNode(Node):
    def __init__(self):
        super().__init__("save_contact_info_node")

        self.obstacle = "glass"  # 'glass' or 'pipe'
        self.mode = "dataset"  # 'dataset' or 'mission"

        data_folder = "/media/victor/DATA/rosbag_asta_data/07-07-recording-nets/asta_net_in_porteria"

        if self.obstacle == "glass" and self.mode == "dataset":

            bag_folder = "/media/victor/DATA/rosbag_asta_data/11-07-manual-collisions-for-dataset/rosbag2_2024_07_11-14_01_34"
            bag_folder.split("/")[-1]

            self.csv_file = os.path.join(bag_folder, f"flight_data.csv")
            print(f"csv path: {self.csv_file}")
            with open(self.csv_file, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        "raw_value_1",
                        "raw_value_2",
                        "raw_value_3",
                        "raw_value_4",
                        "uav_pose_x",
                        "uav_pose_y",
                        "uav_pose_z",
                        "uav_q_x",
                        "uav_q_y",
                        "uav_q_z",
                        "uav_q_w",
                        "glass0_pose_x",
                        "glass0_pose_y",
                        "glass0_pose_z",
                        "glass1_pose_x",
                        "glass1_pose_y",
                        "glass1_pose_z",
                    ]
                )

        if self.obstacle == "pipe" and self.mode == "mission":
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
                        "collision_value_predicted",
                        "collision_orientation_predicted",
                        "collision_displacement_predicted",
                        "uav_pose_x",
                        "uav_pose_y",
                        "uav_pose_z",
                        "uav_q_x",
                        "uav_q_y",
                        "uav_q_z",
                        "uav_q_w",
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

        self.create_subscription(
            PoseStamped,
            "glass_0_initial",
            self.glass0_cb,
            1,
        )

        self.create_subscription(
            PoseStamped,
            "glass_1_initial",
            self.glass1_cb,
            1,
        )

        self.collision_publisher_gt = self.create_publisher(
            Float32MultiArray, "collision_detection/ground_truth", 10
        )

        self.angle_error_publisher = self.create_publisher(
            Float32, "collision_detection/angle_error", 10
        )

        self.displacement_error_publisher = self.create_publisher(
            Float32, "collision_detection/displacement_error", 10
        )

        # Initialize data storage variables
        self.raw_values = [None, None, None, None]
        self.uav_pose = [None, None, None]
        self.uav_q_xyzw = [None, None, None, None]
        self.pipe_pose = [None, None, None]
        self.pipe_q_wxyz = [None, None, None, None]
        self.initial_contact = False
        self.glass0_pose = None
        self.glass1_pose = None

    def flex_sensor_cb(self, msg):
        self.raw_values = msg.data[:4]

        # Save glass dataset
        if self.obstacle == "glass" and self.mode == "dataset":
            if (
                None not in self.raw_values
                and None not in self.uav_pose
                and None not in self.uav_q_xyzw
                and self.glass0_pose != None
                and self.glass1_pose != None
            ):
                # Save the data to the CSV file
                with open(self.csv_file, mode="a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(
                        [
                            self.raw_values[0],
                            self.raw_values[1],
                            self.raw_values[2],
                            self.raw_values[3],
                            self.uav_pose[0],
                            self.uav_pose[1],
                            self.uav_pose[2],
                            self.uav_q_xyzw[0],
                            self.uav_q_xyzw[1],
                            self.uav_q_xyzw[2],
                            self.uav_q_xyzw[3],
                            self.glass0_pose[0],
                            self.glass0_pose[1],
                            self.glass0_pose[2],
                            self.glass1_pose[0],
                            self.glass1_pose[1],
                            self.glass1_pose[2],
                        ]
                    )

    def collision_detection_cb(self, contact_msg):
        self.collision_predicted = contact_msg.data[0]
        self.collision_orientation_predicted = contact_msg.data[1]
        self.collision_displacement_predicted = contact_msg.data[2]
        contact_msg.data[3]

        # Check if all data is available before saving
        """ if (
            None not in self.raw_values
            and None not in self.uav_pose
            and None not in self.uav_q_xyzw
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
                        self.uav_q_xyzw[0],
                        self.uav_q_xyzw[1],
                        self.uav_q_xyzw[2],
                        self.uav_q_xyzw[3],
                        self.pipe_pose[0],
                        self.pipe_pose[1],
                        self.pipe_pose[2],
                        self.pipe_q_wxyz[0],
                        self.pipe_q_wxyz[1],
                        self.pipe_q_wxyz[2],
                        self.pipe_q_wxyz[3],
                    ]
                ) """

        if self.mode == "mission":
            # Calculate the angle between the UAV's x-axis and the line joining the centers of the UAV and pipe
            if self.obstacle == "pipe":
                angle_gt, disp_gt, angle_err, disp_err = self.calculate_gt_error_pipe()
            elif self.obstacle == "glass":
                angle_gt, disp_gt, angle_err, disp_err = self.calculate_gt_error_glass()

            if angle_gt is not None:
                self.collision_msg = Float32MultiArray()
                self.collision_msg.data = [
                    self.collision_predicted,
                    angle_gt,
                    disp_gt,
                ]
                self.collision_publisher_gt.publish(self.collision_msg)

                angle_error_msg = Float32()
                angle_error_msg.data = angle_err
                self.angle_error_publisher.publish(angle_error_msg)

                displacement_error_msg = Float32()
                displacement_error_msg.data = disp_err
                self.displacement_error_publisher.publish(displacement_error_msg)

    def drone_mocap_cb(self, msg):
        self.uav_pose = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        self.uav_q_xyzw = [
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        ]
        # print(f"UAV p: {self.uav_pose}")

    def pipe_mocap_cb(self, msg):
        self.pipe_pose = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        self.pipe_q_wxyz = [
            msg.pose.orientation.w,
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
        ]
        # print(f"pipe p: {self.pipe_pose}")

    def glass0_cb(self, msg):
        self.glass0_pose = [
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ]

    def glass1_cb(self, msg):
        self.glass1_pose = [
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ]

    def calculate_gt_error_pipe(self):

        if self.pipe_pose[0] is not None and self.uav_pose[0] is not None:
            angle_gt = angle_between_pipe_uav_centers(
                self.uav_pose, self.uav_q_xyzw, self.pipe_pose
            )
            angle_err = calculate_angle_error(
                self.collision_orientation_predicted, angle_gt
            )

            if angle_gt < 0:
                angle_gt += 360

            if self.collision_predicted == 1.0:
                print(f"Initial collision: {self.initial_contact}")
                if not self.initial_contact:
                    print(f"Initializing contact: ")
                    self.initial_contact = True
                    self.initial_uav_pipe_distance = calculate_2d_distance(
                        self.uav_pose, self.pipe_pose
                    )
                    print(
                        f"initial_uav_pipe_distance: {self.initial_uav_pipe_distance}"
                    )
                    displacement_err = 0.0
                    disp_gt = 0.0
                else:
                    print(f"Calculating displacement: ")
                    # Calculate the actual displacement since initial contact
                    actual_uav_pipe_distance = calculate_2d_distance(
                        self.uav_pose, self.pipe_pose
                    )
                    print(f"actual distance: {actual_uav_pipe_distance}")
                    disp_gt = abs(
                        self.initial_uav_pipe_distance - actual_uav_pipe_distance
                    )
                    print(f"Current displacement: {disp_gt}")
                    displacement_err = disp_gt - self.collision_displacement_predicted
                    print(f"displacement calculated: {displacement_err}")
            else:
                self.initial_contact = False
                displacement_err = 0.0
                angle_err = 0.0
                disp_gt = 0.0

            return angle_gt, disp_gt, angle_err, displacement_err

        else:
            return None, None, None, None

    def calculate_gt_error_glass(self):
        if (
            self.glass0_pose is not None
            and self.glass1_pose is not None
            and self.uav_pose is not None
        ):
            angle_gt = calculate_angle_between_uav_yaw_and_glass_normal(
                self.uav_q_xyzw, self.glass0_pose, self.glass1_pose
            )

            angle_err = calculate_angle_error(
                self.collision_orientation_predicted, angle_gt
            )

            if angle_gt < 0:
                angle_gt += 360

            if self.collision_predicted == 1.0:
                print(f"Angle gt: {angle_gt}")
                print(
                    f"uavq: {self.uav_q_xyzw}, g0: {self.glass0_pose}, g1: {self.glass1_pose}"
                )
                print(f"Initial collision: {self.initial_contact}")
                if not self.initial_contact:
                    print(f"Initializing contact: ")
                    self.initial_contact = True
                    self.initial_uav_glass_distance = calculate_3d_distance_to_plane(
                        self.uav_pose, self.glass0_pose, self.glass1_pose
                    )
                    print(
                        f"initial_uav_glass_distance: {self.initial_uav_glass_distance}"
                    )
                    displacement_err = 0.0
                    disp_gt = 0.0
                else:
                    print(f"Calculating displacement: ")
                    # Calculate the actual displacement since initial contact
                    actual_uav_glass_distance = calculate_3d_distance_to_plane(
                        self.uav_pose, self.glass0_pose, self.glass1_pose
                    )
                    print(f"actual distance: {actual_uav_glass_distance}")
                    disp_gt = (
                        abs(self.initial_uav_glass_distance - actual_uav_glass_distance)
                        * 100
                    )  # meters to cm
                    print(f"Current displacement: {disp_gt}")
                    displacement_err = disp_gt - self.collision_displacement_predicted
                    print(f"displacement calculated: {displacement_err}")
            else:
                self.initial_contact = False
                displacement_err = 0.0
                angle_err = 0.0
                disp_gt = 0.0

            return angle_gt, disp_gt, angle_err, displacement_err

        else:
            return None, None, None, None


def main(args=None):
    rclpy.init(args=args)

    node = SaveContactInfoNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
