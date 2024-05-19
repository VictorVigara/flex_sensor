import rclpy
from rclpy.node import Node

from .flex_sensor_connnection import FlexSensorConnection
from .flex_sensor_plot import FlexSensorPlot


class flexDriver(Node):
    """Flex sensor driver to read analogic arduino inputs"""

    def __init__(self):
        super().__init__("flex_sensor")

        ##################
        ### PARAMETERS ###
        ##################

        self.timer_period = 0.05  # Sensor reading frequency

        self.n_sensors = 1
        self.sensor_locations = [
            0
        ]  # List containing sensor angle location (0deg right - counterclockwise)

        self.VCC = 5  # Voltage at arduino 5V line
        self.R_DIV = 82000  # Resistor

        self.radial_plot = True
        self.linear_plot = True

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

        self.flex_conn = FlexSensorConnection(
            self.n_sensors, self.VCC, self.R_DIV, self.port, self.logger
        )
        self.flex_conn.flex_sensors_initialization()

        self.flex_plot = FlexSensorPlot(
            self.flex_conn,
            self.radial_plot,
            self.linear_plot,
            self.timer_period,
            self.n_sensors,
            self.sensor_locations,
            self.max_time_plot,
            [self.ADC_min, self.ADC_max],
        )

        # Plots
        self.flex_plot.plot_initialization()

        # Timer
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

    def timer_callback(self):
        """Main loop reading and plotting flex sensor values"""

        for i in range(self.n_sensors):
            ADC_flex, V_flex, R_flex = self.flex_conn.get_flex_sensor_output(i)
            # Continue if no sensor reading
            if ADC_flex == 0:
                continue

            # Plots
            self.flex_plot.plot_flex_value(ADC_flex, i)


def main(args=None):
    rclpy.init()

    flex_sensor = flexDriver()
    rclpy.spin(flex_sensor)
    flex_sensor.destroy_node()


if __name__ == "__main__":
    main()
