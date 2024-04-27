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
        self.n_sensors = 3

        # Connection parameters
        self.port = '/dev/ttyACM0'
        self.board = Arduino(self.port)
        
        # Initialize variables
        self.pin_list = []

        self.pin_initialization()

        self.it = util.Iterator(self.board)
        self.it.start()

        # Timer
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

        
    def timer_callback(self):
        '''Method executed at a fixed rate self.timer_period'''

        for i in range(self.n_sensors):
            value = self.pin_list[i].read()
            print(f"A{i}: {value}")

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