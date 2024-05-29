import rclpy
from rclpy.node import Node

from .flex_sensor_connnection import FlexSensorConnection
from .flex_sensor_plot import FlexSensorPlot
from .flex_sensor_record_data import FlexSensorDataRecorder


class flexDriver(Node):
    """Flex sensor driver to read analogic arduino inputs"""

    def __init__(self):
        super().__init__("flex_sensor")

        ##################
        ### PARAMETERS ###
        ##################

        self.timer_period = 0.05  # Sensor reading frequency

        self.n_sensors = 4
        self.sensor_locations = [
            0,
            90,
            180,
            270,
        ]  # List containing sensor angle location (0deg right - counterclockwise)

        self.beam_discretization = 45

        self.VCC = 5  # Voltage at arduino 5V line
        self.R_DIV = 82000  # Resistor

        self.radial_plot = True
        self.linear_plot = False

        self.record_data = False
        self.record_time = 1

        self.calibration = False

        # Connection parameters
        self.port = "/dev/ttyACM0"

        # Arduino board analog output range
        self.ADC_max = 1023
        self.ADC_min = 0

        # Plot parameters
        self.max_time_plot = 5  # Seconds to display in timeline plot

        ######################
        ### INITIALIZATION ###
        ######################
        self.logger = self.get_logger

        # Initialize flex sensor connection
        self.flex_conn = FlexSensorConnection(
            self.n_sensors,
            self.VCC,
            self.R_DIV,
            self.port,
            self.logger,
            self.sensor_locations,
            self.beam_discretization,
        )
        self.flex_conn.flex_sensors_initialization()

        if self.calibration:
            (
                self.min_sensor_range,
                self.max_sensor_range,
            ) = self.flex_conn.get_sensor_range()
            self.flex_conn.initialize_steady_values()
            self.steady_values = False
        else:
            self.flex_conn.read_sensor_range_from_json()
            self.flex_conn.initialize_steady_values()

        # Initialize flex sensor plot
        self.flex_plot_analog = FlexSensorPlot(
            self.flex_conn,
            self.radial_plot,
            self.linear_plot,
            self.timer_period,
            self.n_sensors,
            self.sensor_locations,
            self.max_time_plot,
            [self.ADC_min, self.ADC_max],
        )
        self.flex_plot_percent = FlexSensorPlot(
            self.flex_conn,
            self.radial_plot,
            self.linear_plot,
            self.timer_period,
            self.n_sensors,
            self.sensor_locations,
            self.max_time_plot,
            [0, 100],
        )

        # Plots
        self.flex_plot_analog.plot_initialization(title="Analog output")
        self.flex_plot_percent.plot_initialization(title="NOrmalized output (%)")

        # Initialize flex sensor data recorder
        if self.record_data:
            self.data_recorder = FlexSensorDataRecorder(
                self.record_time, self.timer_period, self.logger
            )

        # Timer
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

    def timer_callback(self):
        """Main loop reading and plotting flex sensor values"""

        # Analog measurements
        ADC_values = self.flex_conn.read_sensor()
        force_direction = self.flex_conn.get_force_direction(ADC_values)
        # self.flex_plot_analog.plot_flex_value(ADC_values, [0, 1023])

        # Normalized measurements
        sensor_percent = self.flex_conn.get_sensor_percentage()
        self.flex_plot_percent.plot_flex_value(
            sensor_percent, [0, 100], angle=force_direction
        )

        if self.record_data:
            self.data_recorder.record_data(ADC_values)


def main(args=None):
    rclpy.init()

    flex_sensor = flexDriver()
    rclpy.spin(flex_sensor)
    flex_sensor.destroy_node()


if __name__ == "__main__":
    main()
