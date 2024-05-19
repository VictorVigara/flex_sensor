from pyfirmata import Arduino, util


class FlexSensorConnection:
    def __init__(self, n_sensors, VCC, R_DIV, port, logger) -> None:

        self.n_sensors = n_sensors
        self.VCC = VCC
        self.R_DIV = R_DIV
        self.port = port
        self.logger = logger

        ### INITIALIZATION ###
        self.pin_list = []  # Sensor pin access list

        # Variables to save max and min values read
        self.v_max = 0
        self.v_min = 9999

    def flex_sensors_initialization(self):
        """Initialize which pins to read"""

        self.board = Arduino(self.port)
        for i in range(self.n_sensors):
            self.pin_list.append(self.board.get_pin(f"a:{i}:i"))

        self.it = util.Iterator(self.board)
        self.it.start()

    def get_flex_sensor_output(self, i):
        """
        Read flex sensor output and calculate R and V

        args:
            i (int): flex sensor
        returns:

        """
        ADC_flex = self.pin_list[i].read() * 1000
        if ADC_flex == 0:
            self.logger().error(f"Sensor {i} not connected properly")
            V_flex = None
            R_flex = None
            return ADC_flex, V_flex, R_flex

        V_flex = ADC_flex * self.VCC / 1023
        if V_flex > self.v_max:
            self.v_max = V_flex
        if V_flex < self.v_min:
            self.v_min = V_flex
        R_flex = self.R_DIV * (self.VCC / V_flex - 1)
        self.logger().info(f"A{i} V: {V_flex}V / R:{R_flex} ohms")
        self.logger().info(f"V_max: {self.v_max} / V_min: {self.v_min}")

        return ADC_flex, V_flex, R_flex
