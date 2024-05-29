import matplotlib.pyplot as plt
import numpy as np


class FlexSensorPlot:
    def __init__(
        self,
        flex_conn,
        radial_plot,
        linear_plot,
        freq,
        n_sensors,
        sensor_locations,
        plot_time,
        arduino_analog_range,
    ) -> None:

        self.flex_conn = flex_conn
        self.radial_plot = radial_plot
        self.linear_plot = linear_plot
        self.timer_period = freq
        self.n_sensors = n_sensors
        self.sensor_locations = sensor_locations[: self.n_sensors]
        self.max_time_plot = plot_time
        self.ADC_min, self.ADC_max = arduino_analog_range

    def init_linear_plot(self, title):
        ### Time line plot ###

        if self.linear_plot:
            self.fig_time, self.ax_time = plt.subplots(4)
            self.fig_time.suptitle(title)

            self.max_recorded_values = int(self.max_time_plot / self.timer_period)

            self.x_axis_time = np.linspace(
                0, self.max_time_plot, self.max_recorded_values
            )

            self.sensor_values = []
            self.plot_lines = []
            for i in range(self.n_sensors):
                self.sensor_values.append(np.zeros(self.max_recorded_values))
                (line,) = self.ax_time[i].plot(self.x_axis_time, self.sensor_values[i])
                self.plot_lines.append(line)

    def init_radial_plot(self, title):
        ### Radial plot ###
        if self.radial_plot:
            self.fig_rad, self.ax_radial = plt.subplots(
                subplot_kw={"projection": "polar"}
            )
            self.ax_radial.set_rmax(self.ADC_max)
            rticks = list(
                np.array(np.linspace(self.ADC_min, self.ADC_max, 10)).astype(int)
            )
            self.ax_radial.set_rticks(rticks)  # Less radial ticks
            self.ax_radial.set_rlabel_position(
                -45
            )  # Move radial labels away from plotted line
            self.ax_radial.grid(True)
            self.fig_rad.suptitle(title)

            self.theta = np.array(self.sensor_locations) * (np.pi / 180)
            self.values = []
            self.width = np.ones(4) * (np.pi / 4)
            # self.colors = plt.cm.viridis(self.values / 10.)

    def plot_initialization(self, title):

        if self.linear_plot or self.radial_plot:
            # Enable interactive mode
            plt.ion()
            # plt.show()

        self.init_linear_plot(title)
        self.init_radial_plot(title)

    def plot_arrow(self, angle, length=1.0, color="r"):
        """
        Plot an arrow in the radial plot pointing to the center at a given angle.

        args:
            angle: The angle in degrees (0 degrees is to the right, counterclockwise)
            length: The length of the arrow (default is 1.0)
            color: The color of the arrow (default is 'r')
        """
        if self.radial_plot:
            angle_rad = np.deg2rad(angle)  # Convert angle to radians
            self.ax_radial.arrow(
                angle_rad,
                0,
                0,
                length,
                head_width=0.1,
                head_length=0.2,
                fc=color,
                ec=color,
            )

    def plot_flex_value(self, value, limits=[0, 100], angle=0):
        """
        Plot sensor value in the required location

        args:
            value: sensor output value (ADC [0-1023])
            sensor_location: sensor angle location (0deg right - counterclockwise)
        """

        ### RADIAL ###
        # Clear the previous scatter plot while keeping the axis and labels intact

        if self.radial_plot:
            self.ax_radial.cla()
            self.radial_values = np.array(value)
            self.ax_radial.bar(
                self.theta,
                self.radial_values,
                width=self.width[: self.n_sensors],
                bottom=0.0,
                alpha=0.5,
            )
            angle_rad = np.deg2rad(angle)  # Convert angle to radians
            self.ax_radial.arrow(
                angle_rad, 0, 0, 50, head_width=0.1, head_length=0.2, fc="r", ec="r"
            )
            self.ax_radial.set_rmax(150)
            rticks = list(np.array(np.linspace(limits[0], limits[1], 10)).astype(int))
            self.ax_radial.set_rticks(rticks)  # Less radial ticks
        ### LINEAR ###
        if self.linear_plot:
            for sensor_idx in range(self.n_sensors):
                self.sensor_values[sensor_idx] = np.append(
                    self.sensor_values[sensor_idx], value[sensor_idx]
                )
                if len(self.sensor_values[sensor_idx]) > self.max_recorded_values:
                    self.sensor_values[sensor_idx] = self.sensor_values[sensor_idx][
                        -self.max_recorded_values :
                    ]

                self.plot_lines[sensor_idx].set_ydata(self.sensor_values[sensor_idx])
                self.plot_lines[sensor_idx].set_xdata(self.x_axis_time)
                self.ax_time[sensor_idx].set_ylim([limits[0], limits[1]])

        if self.linear_plot or self.radial_plot:
            plt.draw()
            plt.pause(0.0001)
