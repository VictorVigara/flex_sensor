import rclpy
from rclpy.node import Node

import time 
from pyfirmata import Arduino, util

import matplotlib.pyplot as plt
import numpy as np


class flexDriver(Node): 
    '''Flex sensor driver to read analogic arduino inputs'''
    def __init__(self): 
        super().__init__('flex_sensor')

        ##################
        ### PARAMETERS ###
        ##################

        self.timer_period = 0.05    # Sensor reading frequency 

        self.n_sensors = 1
        self.sensor_locations = [0] # List containing sensor angle location (0deg right - counterclockwise)

        self.VCC = 5                # Voltage at arduino 5V line
        self.R_DIV = 82000          # Resistor  

        self.radial_plot = True
        self.linear_plot = False

        # Connection parameters
        self.port = '/dev/ttyACM0'
        
        # Plot parameters
        self.max_time_plot = 5  # Seconds to display in timeline plot
        
        ######################
        ### INITIALIZATION ###
        ######################

        self.pin_list = []  # Sensor pin access list
        self.flex_sensors_initialization()
        
        # Arduino board analog output range
        self.ADC_max = 1023         
        self.ADC_min = 0
        
        # Variables to save max and min values read
        self.v_max = 0
        self.v_min = 9999

        # Plots
        self.plot_initialization()

        # Timer
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

        
    def timer_callback(self):
        '''Method executed at a fixed rate self.timer_period'''

        for i in range(self.n_sensors):
            ADC_flex, V_flex, R_flex = self.get_flex_sensor_output(i)
            # Continue if no sensor reading
            if ADC_flex == 0: 
                continue

            # Plots
            self.plot_flex_value(ADC_flex, i)

    def get_flex_sensor_output(self, i): 
        """
        Read flex sensor output and calculate R and V

        args:
            i (int): flex sensor
        returns: 

        """
        ADC_flex = self.pin_list[i].read()*1000
        if ADC_flex == 0: 
            self._logger.error(f"Sensor {i} not connected properly")
            return
        
        V_flex = ADC_flex * self.VCC / 1023
        if V_flex > self.v_max: 
            self.v_max = V_flex
        if V_flex < self.v_min: 
            self.v_min = V_flex
        R_flex = self.R_DIV * (self.VCC / V_flex -1)
        self.get_logger().info(f"A{i} V: {V_flex}V / R:{R_flex} ohms")
        self.get_logger().info(f"V_max: {self.v_max} / V_min: {self.v_min}")

        return ADC_flex, V_flex, R_flex

    def flex_sensors_initialization(self): 
        """Initialize which pins to read"""

        self.board = Arduino(self.port)
        for i in range(self.n_sensors): 
            self.pin_list.append(self.board.get_pin(f'a:{i}:i'))

        self.it = util.Iterator(self.board)
        self.it.start()
    
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

        if self.linear_plot or self.radial_plot:
            # Enable interactive mode
            plt.ion()
            plt.show()

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

def main(args=None): 
    rclpy.init()

    flex_sensor = flexDriver()
    rclpy.spin(flex_sensor)
    flex_sensor.destroy_node()


if __name__ == '__main__': 
    main()