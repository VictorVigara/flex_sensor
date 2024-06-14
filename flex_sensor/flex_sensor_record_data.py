import csv
import os


class FlexSensorDataRecorder:
    def __init__(self, record_time, timer_period, logger) -> None:
        self.record_time = record_time
        self.frequency = timer_period
        self.logger = logger

        self.recording_positions = [
            7,
        ]  # [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]
        self.recording_orientations = [0, 45, 90, 135, 180, 225, 270, 315]

        self.data_path = "/home/victor/ws_sensor_combined/src/flex_sensor/data/"

        # Initialization
        self.n_total_recordings = int(self.record_time / self.frequency)
        self.n_recorded = 0
        self.recording_finished = False

        self.position_idx = 0
        self.orientation_idx = 0

        self.recorded_values = []

        folder_name = input("Enter recording date and name: ")

        # Create folder
        self.folder_path = self.data_path + folder_name
        os.mkdir(self.folder_path)

        input(
            f"Press enter to record position: {self.recording_positions[self.position_idx]} / orientation: {self.recording_orientations[self.orientation_idx]}"
        )

    def record_data(self, values):
        """
        Record data and save it into a CSV file.
        args:
            values: list of flex sensor values
        """

        # If recording completed, save the data. Otherwise, continue recording.
        if self.n_recorded < self.n_total_recordings and not self.recording_finished:
            self.recorded_values.append(values)
            self.n_recorded += 1

        else:

            if not self.recording_finished:
                # Specify the file name
                filename = f"{self.folder_path}/orientation_{self.recording_orientations[self.orientation_idx]}_pos_{self.recording_positions[self.position_idx]}.csv"

                # Open the file in write mode
                with open(filename, mode="w", newline="") as file:
                    writer = csv.writer(file)

                    # Write each sublist as a row
                    writer.writerows(self.recorded_values)

                print(f"Data saved to {filename}")

                # Update recording stage
                if (self.position_idx + 1) == len(self.recording_positions):
                    self.position_idx = 0
                    self.orientation_idx += 1
                else:
                    self.position_idx += 1

                # check if recording has finished
                if (self.orientation_idx) == len(self.recording_orientations):
                    self.logger().info("Recording finished!!!!!!!!!!!!!")
                    self.recording_finished = True
                    input("Press enter to finish the recording!")
                else:
                    input(
                        f"Press enter to record position: {self.recording_positions[self.position_idx]} / orientation: {self.recording_orientations[self.orientation_idx]}"
                    )
                    self.n_recorded = 0
                    self.recorded_values = []
