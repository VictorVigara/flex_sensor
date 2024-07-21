import csv
import os

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Float32, Float32MultiArray, Int16

from px4_msgs.msg import VehicleCommand, VehicleLocalPosition
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from nav_msgs.msg import Path

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection



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


class SaveMissionToPlot(Node):
    def __init__(self):
        super().__init__("save_mission_to_plot")

        self.simulation = False
        self.contact_map_resolution = 0.88
        self.bag_folder = "/media/victor/DATA/rosbag_asta_data/09-07-blind_tests_2_succesfull/rosbag2_2024_07_09-14_20_47"

        self.obstacle = "pipe"  # 'glass' or 'pipe'
        self.mode = "dataset"  # 'dataset' or 'mission"

        data_folder = "/media/victor/DATA/rosbag_asta_data/07-07-recording-nets/asta_net_in_porteria"

        # INITIALIZATION
        self.contact_obstacles = []
        self.in_contact_event = False
        self.vehicle_position_mission = []
        self.vehicle_position_mission_contact_mask = []
        self.number_of_contact_events = 0

        self.orien_contact_events = []
        self.disp_contact_events = []
        self.current_orien_pred = []
        self.current_disp_pred = []
        self.current_orien_gt = []
        self.current_disp_gt = []
        self.current_orien_error = []
        self.current_disp_error = []
        self.current_raw_values = []

        self.RRT_path_curr = []

        # SUBSCRIBERS
        # Configure QoS profile for publishing and subscribing
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # UAV pose
        self.create_subscription(
            VehicleLocalPosition,
            "/fmu/out/vehicle_local_position",
            self.vehicle_local_position_callback,
            qos_profile,
        )

        self.create_subscription(
            PoseStamped, 
            '/px4_ekf_pose', 
            self.ekf_px4_pose_callback, 
            1
        )

        # Collision prediction
        self.create_subscription(
            Float32MultiArray,
            "collision_detection",
            self.collision_detection_cb,
            1,
        )

        # Contact map
        self.create_subscription(
            PointCloud2, "/contact_obstacles", self.contact_map_callback, 1
        )

        # Flex sensors
        self.create_subscription(
            Float32MultiArray,
            "collision_platform/raw_values",
            self.flex_sensor_cb,
            1,
        )

        # # RRT path
        self.create_subscription(Path, "/RRT_path", self.RRT_path_cb, 1)

        # # Curr waypoint
        self.create_subscription(Int16, "/curr_waypoint", self.curr_wayp_cb, 1)

        # # Obstacles GT
        # self.create_subscription(
        #     PoseStamped,
        #     "glass_0_initial",
        #     self.glass0_cb,
        #     1,
        # )

        # self.create_subscription(
        #     PoseStamped,
        #     "glass_1_initial",
        #     self.glass1_cb,
        #     1,
        # )

        self.create_subscription(
            PoseStamped,
            "pipe_initial",
            self.pipe_mocap_cb,
            1,
        )

        # UAV GT pose
        self.create_subscription(
            PoseStamped,
            "mocap_initial",
            self.drone_mocap_cb,
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

    
    
    def vehicle_local_position_callback(self, vehicle_local_position):
        """Callback function for vehicle_local_position topic subscriber."""
        if self.simulation:
            x = vehicle_local_position.y
            y = vehicle_local_position.x
            z = -vehicle_local_position.z

            self.vehicle_position = [x, y, z]

            self.save_vehicle_position_and_mask()

            

    def ekf_px4_pose_callback(self, ekf_pose_msg: PoseStamped) -> None: 
        if not self.simulation:
            x = ekf_pose_msg.pose.position.x
            y = ekf_pose_msg.pose.position.y
            z = ekf_pose_msg.pose.position.z
            self.vehicle_position = [x, y,z]

            self.save_vehicle_position_and_mask()

    def collision_detection_cb(self, contact_msg):
        self.collision_predicted = contact_msg.data[0]
        self.collision_orientation_predicted = contact_msg.data[1]
        self.collision_displacement_predicted = contact_msg.data[2]
        

        if self.collision_predicted == 1.0: 
            if self.in_contact_event == False: 
                self.in_contact_map_received = False
                print(f"STARTING CONTACT")

                self.initial_uav_pipe_distance = calculate_2d_distance(
                    self.uav_pose, self.pipe_pose
                )
                print(
                    f"initial_uav_pipe_distance: {self.initial_uav_pipe_distance}"
                )
                displacement_err = 0.0
                disp_gt = 0.0

            
            self.current_orien_pred.append(self.collision_orientation_predicted)
            self.current_disp_pred.append(self.collision_displacement_predicted)

            # Calculate orientation and displacement ground truth
            if not self.simulation: 

                if self.obstacle == 'pipe':
                    angle_gt = angle_between_pipe_uav_centers(
                    self.uav_pose, self.uav_q_xyzw, self.pipe_pose
                    )
                    self.current_orien_gt.append(angle_gt)
                    angle_err = calculate_angle_error(
                        self.collision_orientation_predicted, angle_gt
                    )
                    self.current_orien_error.append(angle_err)

                # Calculate displacement once contact started
                if self.in_contact_event: 
                    if self.obstacle == 'pipe':
                        print(f"Calculating displacement: ")
                        # Calculate the actual displacement since initial contact
                        actual_uav_pipe_distance = calculate_2d_distance(
                            self.uav_pose, self.pipe_pose
                        )
                        print(f"actual distance: {actual_uav_pipe_distance}")
                        disp_gt = abs(
                            self.initial_uav_pipe_distance - actual_uav_pipe_distance
                        )
                        self.current_disp_gt.append(disp_gt*100)
                        print(f"Current displacement: {disp_gt}")
                        displacement_err = abs(disp_gt - self.collision_displacement_predicted)
                        print(f"displacement calculated: {displacement_err}")
                        self.current_disp_error.append(displacement_err)
                
            
            self.in_contact_event = True
            print(f"map flag to False")
            

            print(f"Disp: {self.collision_displacement_predicted}")
        else: 
            if self.in_contact_event:

                self.orien_contact_events.append(self.current_orien_pred)
                self.disp_contact_events.append(self.current_disp_pred)

                # If no contact map received during contact, append a None object to not plot that contact event
                if not self.in_contact_map_received: 
                    self.contact_obstacles.append([None, None, None])

                self.in_contact_event  = False
                print(f"CONTACT FINISHED")
                self.number_of_contact_events += 1
            
            #if len(self.RRT_path_curr) > 0:

                self.plot_contact_event()
                self.RRT_path_curr = []


                self.current_orien_pred = []
                self.current_disp_pred = []
                self.current_orien_gt = []
                self.current_disp_gt = []
                self.current_orien_error = []
                self.current_disp_error = []
                self.current_raw_values = []

        """ if self.mode == "mission":
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
                self.displacement_error_publisher.publish(displacement_error_msg) """


    def contact_map_callback(self, pointcloud_contacts: PointCloud2) -> None:
        for p in point_cloud2.read_points(
            pointcloud_contacts, field_names=("x", "y", "z"), skip_nans=True
        ):
            # Get XYZ coordinates to calculate vertical angle and filter by vertical scans
            x = p[0]
            y = p[1]
            z = p[2]

            contact_obstacle = [x, y, z]

            # Save contact obstacles in order as they are contacted
            if contact_obstacle not in self.contact_obstacles: 
                print(f"map flag to True")
                self.in_contact_map_received = True
                self.contact_obstacles.append(contact_obstacle)

        print(f"Contact map updated: {self.contact_obstacles}")

    def flex_sensor_cb(self, msg):
        self.raw_values = np.array(msg.data[:4])

        if self.in_contact_event: 
            self.current_raw_values.append(self.raw_values)

    def RRT_path_cb(self, msg): 
        #TODO: read path
        self.RRT_path_curr = []
        poses = msg.poses
        for wayp in poses: 
            curr_wayp = [wayp.pose.position.x, wayp.pose.position.y, wayp.pose.position.z]
            self.RRT_path_curr.append(curr_wayp)

    def curr_wayp_cb(self, msg): 
        self.curr_wayp = msg.data

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

    def pipe_mocap_cb(self, msg):
        self.pipe_pose = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        self.pipe_q_wxyz = [
            msg.pose.orientation.w,
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
        ]
        # print(f"pipe p: {self.pipe_pose}")

    def drone_mocap_cb(self, msg):
        self.uav_pose = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        self.uav_q_xyzw = [
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        ]
        # print(f"UAV p: {self.uav_pose}")


    def save_vehicle_position_and_mask(self): 
        self.vehicle_position_mission.append(self.vehicle_position)
            
        if self.in_contact_event: 
            self.vehicle_position_mission_contact_mask.append(True)
        else:
            self.vehicle_position_mission_contact_mask.append(False)



    def plot_contact_event(self): 
        # Create a figure with specific size
        fig = plt.figure(figsize=(12, 8))

        # Add a 3D subplot in the left part (1/3 of the figure width)
        ax1 = plt.subplot2grid((5, 3), (0, 0), rowspan=5, projection='3d')

        # Convert lists to numpy arrays
        vehicle_position_mission_np = np.array(self.vehicle_position_mission)
        vehicle_position_mission_contact_mask_np = np.array(self.vehicle_position_mission_contact_mask)

        # Initialize variables to track the start and end of the last set of True values
        start = end = prev_start = prev_end = None

        # Traverse the list in reverse to find the last block of True values
        for i in range(len(self.vehicle_position_mission_contact_mask) - 1, -1, -1):
            if self.vehicle_position_mission_contact_mask[i] and end is None:
                end = i
            elif not self.vehicle_position_mission_contact_mask[i] and end is not None:
                start = i + 1
                break

        # If no False was found after the last True block, the start is the beginning of the list
        if start is None:
            start = 0

        # Get the indices of the last consecutive set of True values
        true_indices = list(range(start, end + 1))

        # Find the previous False subset
        for i in range(start - 1, -1, -1):
            if not self.vehicle_position_mission_contact_mask[i] and prev_end is None:
                prev_end = i
            elif self.vehicle_position_mission_contact_mask[i] and prev_end is not None:
                prev_start = i + 1
                break

        # If no True was found before the False block, the prev_start is the beginning of the list
        if prev_start is None:
            prev_start = 0

        # Get the indices of the previous consecutive set of False values
        false_indices = list(range(prev_start, prev_end + 1))

        filtered_points_np = vehicle_position_mission_np[np.array(true_indices)]
        prev_path_np = vehicle_position_mission_np[np.array(false_indices)]
        rrt_path_np = np.array(self.RRT_path_curr)



        # Plot the trajectory
        ax1.plot(filtered_points_np[:, 0], filtered_points_np[:, 1], filtered_points_np[:, 2], label='UAV Contact Trajectory')
        ax1.plot(prev_path_np[:, 0], prev_path_np[:, 1], prev_path_np[:, 2], label='UAV  prev Contact Trajectory')
        #ax1.plot(rrt_path_np[self.curr_wayp:, 0], rrt_path_np[self.curr_wayp:, 1], rrt_path_np[self.curr_wayp:, 2], label='RRT path recalculated')

        # Plot the boxes
        for center in self.contact_obstacles:
            x, y, z = center
            r = self.contact_map_resolution / 2
            # Vertices of a cube
            vertices = [
                [x-r, y-r, z-r],
                [x+r, y-r, z-r],
                [x+r, y+r, z-r],
                [x-r, y+r, z-r],
                [x-r, y-r, z+r],
                [x+r, y-r, z+r],
                [x+r, y+r, z+r],
                [x-r, y+r, z+r]
            ]
            # List of self.contact_map_resolutions' polygons
            faces = [
                [vertices[j] for j in [0, 1, 2, 3]],
                [vertices[j] for j in [4, 5, 6, 7]], 
                [vertices[j] for j in [0, 3, 7, 4]], 
                [vertices[j] for j in [1, 2, 6, 5]], 
                [vertices[j] for j in [0, 1, 5, 4]],
                [vertices[j] for j in [2, 3, 7, 6]]
            ]
            ax1.add_collection3d(Poly3DCollection(faces, alpha=0.1, linewidths=1, edgecolors='r'))

        # Set the same scale for all axes with specific limits
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.set_zlabel("z")
        """ ax1.set_xlim([self.vehicle_position[0]-1, self.vehicle_position[0]+6])
        ax1.set_ylim([self.vehicle_position[1]-3, self.vehicle_position[1]+3])
        ax1.set_zlim([0, 3]) """

        # Adjust the view angle
        ax1.view_init(elev=45, azim=230)  # Elevation and azimuth angle for the view


        ax1.set_title('Left Plot: 3D Trajectory and Boxes')
        ax1.legend()

        # Add 5 vertically stacked subplots in the right part (2/3 of the figure width)
        if self.simulation: 
            ax2 = plt.subplot2grid((2, 3), (0, 1), colspan=2)
            ax3 = plt.subplot2grid((2, 3), (1, 1), colspan=2)
            ax2.set_title('GT angle')
            ax3.set_title('GT displacement')
            ax2.plot(self.current_orien_pred)
            ax3.plot(self.current_disp_pred)
        else:
            a = np.array(self.current_raw_values)
            ax2 = plt.subplot2grid((5, 3), (0, 1), colspan=2)
            ax2.set_title('Flex Sensor values')
            ax2.plot(a[:,0], label="Sensor1")
            ax2.plot(a[:,1], label="Sensor2")
            ax2.plot(a[:,2], label="Sensor3")
            ax2.plot(a[:,3], label="Sensor4")
            ax3 = plt.subplot2grid((5, 3), (1, 1), colspan=2)
            ax3.set_title('Predicted vs GT angle')
            ax3.plot(self.current_orien_pred, label="Predicted angle")
            ax3.plot(self.current_orien_gt, label="GT angle")
            ax4 = plt.subplot2grid((5, 3), (2, 1), colspan=2)
            ax4.set_title('Angle error')
            ax4.plot(self.current_orien_error)
            ax5 = plt.subplot2grid((5, 3), (3, 1), colspan=2)
            ax5.plot(self.current_disp_pred, label="Predicted displacement")
            ax5.plot(self.current_disp_gt, label="GT displacement")
            ax6 = plt.subplot2grid((5, 3), (4, 1), colspan=2)
            ax6.plot(self.current_disp_error)

            # Set titles for visualization
            
            
            
            ax5.set_title('Predicted vs GT displacement')
            ax6.set_title('Displacement error')

        plt.tight_layout()
        plt.savefig(self.bag_folder + f"/contact_event_{self.number_of_contact_events}.png")

    





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

    node = SaveMissionToPlot()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
