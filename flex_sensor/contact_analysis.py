import csv
import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R

data_folder = (
    "/media/victor/DATA/rosbag_asta_data/07-07-recording-nets/asta_net_in_porteria"
)
csv_file = os.path.join(data_folder, "contact_info.csv")


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


def read_contact_info(csv_file):
    with open(csv_file, mode="r") as file:
        reader = csv.DictReader(file)
        distances = []
        angle_differences = []
        displacement_differences = []

        initial_contact = None

        data = []

        for row in reader:
            collision_predicted = float(row["collision_predicted"])
            raw_values = [
                float(row["raw_value_1"]),
                float(row["raw_value_2"]),
                float(row["raw_value_3"]),
                float(row["raw_value_4"]),
            ]

            if not collision_predicted:
                data.append(
                    (uav_pose, uav_q_wxyz, pipe_pose, pipe_q_wxyz, actual_angle)
                )

            if collision_predicted:

                collision_orientation_predicted = float(
                    row["collision_orientation_predicted"]
                )
                float(row["collision_displacement_predicted"])

                uav_pose = [
                    float(row["uav_pose_x"]),
                    float(row["uav_pose_y"]),
                    float(row["uav_pose_z"]),
                ]
                uav_q_wxyz = [
                    float(row["uav_q_w"]),
                    float(row["uav_q_x"]),
                    float(row["uav_q_y"]),
                    float(row["uav_q_z"]),
                ]

                pipe_pose = [
                    float(row["pipe_pose_x"]),
                    float(row["pipe_pose_y"]),
                    float(row["pipe_pose_z"]),
                ]
                pipe_q_wxyz = [
                    float(row["pipe_q_w"]),
                    float(row["pipe_q_x"]),
                    float(row["pipe_q_y"]),
                    float(row["pipe_q_z"]),
                ]

                # Calculate the distance between UAV and pipe
                distance = calculate_2d_distance(uav_pose, pipe_pose)
                distances.append(distance)

                # Calculate the angle between the UAV's x-axis and the line joining the centers of the UAV and pipe
                actual_angle = calculate_angle_between_centers(
                    uav_pose, uav_q_wxyz, pipe_pose
                )
                angle_difference = abs(collision_orientation_predicted - actual_angle)
                angle_differences.append(angle_difference)

                if initial_contact is None:
                    initial_contact = True
                    initial_uav_pipe_distance = calculate_2d_distance(
                        uav_pose, pipe_pose
                    )
                else:
                    # Calculate the actual displacement since initial contact
                    actual_uav_pipe_distance = calculate_2d_distance(
                        uav_pose, pipe_pose
                    )
                    displacement_difference = (
                        initial_uav_pipe_distance - actual_uav_pipe_distance
                    )
                    displacement_differences.append(displacement_difference)

                data.append(
                    (uav_pose, uav_q_wxyz, pipe_pose, pipe_q_wxyz, actual_angle)
                )

        return distances, angle_differences, displacement_differences, data


def calculate_metrics(data):
    mean = np.mean(data)
    std_dev = np.std(data)
    return mean, std_dev


def update_plot(index):
    uav_pose, uav_q_wxyz, pipe_pose, pipe_q_wxyz, actual_angle = data[index]

    ax1.clear()
    ax1.set_title("UAV and Pipe Pose")
    ax1.set_xlim(-10, 10)
    ax1.set_ylim(-10, 10)

    ax1.plot(uav_pose[0], uav_pose[1], "bo", label="UAV")
    ax1.plot(pipe_pose[0], pipe_pose[1], "ro", label="Pipe")
    ax1.arrow(
        uav_pose[0],
        uav_pose[1],
        1,
        0,
        head_width=0.2,
        head_length=0.2,
        fc="blue",
        ec="blue",
    )
    ax1.arrow(
        pipe_pose[0],
        pipe_pose[1],
        1,
        0,
        head_width=0.2,
        head_length=0.2,
        fc="red",
        ec="red",
    )
    ax1.legend()

    ax2.clear()
    ax2.set_title("Actual Angle")
    ax2.set_xlim(-180, 180)
    ax2.set_ylim(0, 1)
    ax2.axvline(actual_angle, color="green", label=f"Actual Angle: {actual_angle:.2f}°")
    ax2.legend()


if __name__ == "__main__":
    distances, angle_differences, displacement_differences, data = read_contact_info(
        csv_file
    )

    distance_mean, distance_std_dev = calculate_metrics(distances)
    angle_diff_mean, angle_diff_std_dev = calculate_metrics(angle_differences)
    displacement_diff_mean, displacement_diff_std_dev = calculate_metrics(
        displacement_differences
    )

    print(
        f"Distance between UAV and Pipe during contact - Mean: {distance_mean}, Std Dev: {distance_std_dev}"
    )
    print(f"Angle Difference - Mean: {angle_diff_mean}, Std Dev: {angle_diff_std_dev}")
    print(
        f"Displacement Difference - Mean: {displacement_diff_mean}, Std Dev: {displacement_diff_std_dev}"
    )

    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    plt.subplots_adjust(left=0.1, bottom=0.25)

    def update_frame(i):
        update_plot(i)
        return (fig,)

    ani = animation.FuncAnimation(
        fig, update_frame, frames=len(data), interval=100, blit=False
    )

    ani.save(data_folder + "/uav_pipe_contact.gif", writer="pillow")

    plt.show()
