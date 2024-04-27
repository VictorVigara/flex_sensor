import rclpy
from rclpy.node import Node

import time 
from pyfirmata import Arduino, util

class flexDriver(Node): 
    '''Flex sensor driver to read analogic arduino inputs'''
    def __init__(self): 
        super().__init__('flex_sensor')

        self.timer_period = 0.01

        # Set-up parameters
        self.n_sensors = 1
        self.VCC = 5 # Voltage at arduino 5V line
        self.R_DIV = 82000

        # Connection parameters
        self.port = '/dev/ttyACM0'
        self.board = Arduino(self.port)
        
        # Initialize variables
        self.pin_list = []
        self.v_max = 0
        self.v_min = 9999

        self.pin_initialization()

        self.it = util.Iterator(self.board)
        self.it.start()

        # Timer
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

        
    def timer_callback(self):
        '''Method executed at a fixed rate self.timer_period'''

        for i in range(self.n_sensors):
            ADC_flex = self.pin_list[i].read()*1000
            V_flex = ADC_flex * self.VCC / 1023

            if V_flex > self.v_max: 
                self.v_max = V_flex
            if V_flex < self.v_min: 
                self.v_min = V_flex
            R_flex = self.R_DIV * (self.VCC / V_flex -1)
            print(f"A{i} V: {V_flex}V / R:{R_flex} ohms")
            print(f"V_max: {self.v_max} / V_min: {self.v_min}")

    def pin_initialization(self): 
        """Initialize which pins to read"""
        for i in range(self.n_sensors): 
            self.pin_list.append(self.board.get_pin(f'a:{i}:i'))

def main(args=None): 
    rclpy.init()

    flex_sensor = flexDriver()
    rclpy.spin(flex_sensor)
    flex_sensor.destroy_node()


if __name__ == '__main__': 
    main()