import matplotlib.pyplot as plt
import numpy as np

class FlexSensorPlot(): 
    def __init__(self, flex_conn, radial_plot, linear_plot, freq, n_sensors, sensor_locations, plot_time, arduino_analog_range) -> None:
        
        self.flex_conn = flex_conn
        self.radial_plot = radial_plot
        self.linear_plot = linear_plot
        self.timer_period = freq
        self.n_sensors = n_sensors
        self.sensor_locations = sensor_locations
        self.max_time_plot = plot_time
        self.ADC_min, self.ADC_max = arduino_analog_range

    def plot_initialization(self): 

        ### Radial plot ###
        if self.radial_plot: 
            self.fig, self.ax = plt.subplots(subplot_kw={'projection': 'polar'})
            self.ax.set_rmax(self.ADC_max)
            rticks = list(np.array(np.linspace(self.ADC_min, self.ADC_max, 10)).astype(int))
            self.ax.set_rticks(rticks)  # Less radial ticks
            self.ax.set_rlabel_position(-45)  # Move radial labels away from plotted line
            self.ax.grid(True)
            self.ax.set_title("Flex sensor output", va='bottom')

            # Enable interactive mode
            plt.ion()
            plt.show()

        ### Time line plot ###
        if self.linear_plot: 
            self.fig_time, self.ax_time = plt.subplots(4)
            self.fig_time.suptitle('Flex sensor output timeline')
            # Remove outer labels for inner plots
            for ax in self.ax_time.flat:
                ax.label_outer()
            
            self.max_recorded_values = int(self.max_time_plot/self.timer_period)

            self.x_axis_time = np.linspace(0, self.max_time_plot, self.max_recorded_values)
            self.y_axis = np.zeros(self.max_recorded_values)

            self.plot_lines = []
            for i in range(self.n_sensors): 
                line, = self.ax_time[i].plot(self.x_axis_time, self.y_axis)
                self.plot_lines.append(line)

            self.recorded_values = []
            self.x_axis_time = []
            # Append as many lists as sensors initialized with 0
            for i in range(self.n_sensors): 
                self.recorded_values.append([])
                self.ax_time[i].set_ylim([100, 600])
                for j in range(self.max_recorded_values):
                    self.recorded_values[i].append(0)
                    if i==0:
                        self.x_axis_time.append(self.timer_period*j)

    def plot_flex_value(self, value, sensor_idx):
        """
        Plot sensor value in the required location

        args: 
            value: sensor output value (ADC [0-1023])
            sensor_location: sensor angle location (0deg right - counterclockwise)
        """

        ### RADIAL ###
        # Clear the previous scatter plot while keeping the axis and labels intact
        if self.radial_plot:
            self.ax.cla()

            location = self.sensor_locations[sensor_idx]*360/np.pi
            self.ax.scatter(location, value)

            # Reapply labels and settings since we cleared the plot
            self.ax.set_rmax(self.ADC_max)
            rticks = list(np.array(np.linspace(self.ADC_min, self.ADC_max, 10)).astype(int))
            self.ax.set_rticks(rticks)
            self.ax.set_rlabel_position(-45)
            self.ax.grid(True)
            self.ax.set_title("Flex sensor output", va='bottom')

            # Redraw the canvas and process GUI events
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

        ### LINEAR ###
        if self.linear_plot: 
            # Delete last measurement
            self.recorded_values[sensor_idx].pop(0)
            self.recorded_values[sensor_idx].append(value)

            self.plot_lines[sensor_idx].set_ydata(np.array(self.recorded_values))
            
            self.ax_time[sensor_idx].set_ylim([100, 600])

            #plt.show()

            self.fig_time.canvas.draw()
            self.fig_time.canvas.flush_events()