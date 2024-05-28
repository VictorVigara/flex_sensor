import json
import time

from pyfirmata import Arduino, util


class FlexSensorConnection:
    def __init__(self, n_sensors, VCC, R_DIV, port, logger, sensor_locations) -> None:

        self.n_sensors = n_sensors
        self.VCC = VCC
        self.R_DIV = R_DIV
        self.port = port
        self.logger = logger
        self.sensor_locations = sensor_locations

        ### INITIALIZATION ###
        self.pin_list = []  # Sensor pin access list

        # Variables to save max and min values read
        self.v_max = 0
        self.v_min = 9999

        # Calibration variables
        self.calibration_time = 2

        # Lists to save sensor range
        self.min_limits = []
        self.max_limits = []

        self.sensor_ADC = [0, 0, 0, 0]
        self.steady_values = [0, 0, 0, 0]
        self.sensor_percent = [0, 0, 0, 0]

    def flex_sensors_initialization(self):
        """Initialize which pins to read"""

        self.board = Arduino(self.port)
        for i in range(self.n_sensors):
            self.pin_list.append(self.board.get_pin(f"a:{i}:i"))

        self.it = util.Iterator(self.board)
        self.it.start()

    def read_sensor(self):
        """
        Read flex sensor output and calculate R and V

        args:
            i (int): flex sensor
        returns:

        """
        sensor_ADC = []
        for i in range(self.n_sensors):
            ADC_flex = self.pin_list[i].read() * 1000
            if ADC_flex == 0:
                self.logger().error(f"Sensor {i} not connected properly")
                ADC_flex = None
            """ V_flex = ADC_flex * self.VCC / 1023
            if ADC_flex > self.v_max:
                self.v_max = ADC_flex
            if ADC_flex < self.v_min:
                self.v_min = ADC_flex
            R_flex = self.R_DIV * (self.VCC / V_flex - 1) """
            # self.logger().info(f"A{i} V: {V_flex}V / R:{R_flex} ohms")

            self.logger().info(
                f" Sensor {i} | ADC: {ADC_flex} / ADC_max: {self.v_max} / ADC_min: {self.v_min}"
            )

            sensor_ADC.append(ADC_flex)
        return sensor_ADC

    def get_sensor_range(self):
        """
        Record max min value of each sensor
        """

        for i in range(self.n_sensors):

            # Get min value
            input(f"Bend sensor {i} and press Enter to get min range")
            self.logger().info(f"Reading sensor {i} ...")
            duration = 0
            ADC_min = None
            init_time = time.time()

            while duration < self.calibration_time:
                # Read sensor
                sensor_values = self.read_sensor()
                if None not in sensor_values:
                    sensor_value_i = sensor_values[i]

                    if ADC_min == None or sensor_value_i < ADC_min:
                        ADC_min = sensor_value_i

                    duration = time.time() - init_time
                else:
                    continue

            self.min_limits.append(ADC_min)
            self.logger().info(f"Sensor {i} min value: {ADC_min}")

            # Get max value
            input(f"Stretch sensor {i} and press Enter to get max range")
            self.logger().info(f"Reading sensor {i} ...")
            duration = 0
            ADC_max = None
            init_time = time.time()

            while duration < self.calibration_time:
                # Read sensor
                sensor_values = self.read_sensor()
                if None not in sensor_values:
                    sensor_value_i = sensor_values[i]

                    if ADC_max == None or sensor_value_i > ADC_max:
                        ADC_max = sensor_value_i

                    duration = time.time() - init_time
                else:
                    continue

            self.max_limits.append(ADC_max)
            self.logger().info(f"Sensor {i} max value: {ADC_max}")

            self.sensor_ranges = {
                "min_limits": self.min_limits,
                "max_limits": self.max_limits,
            }

        with open("src/flex_sensor/data/sensor_ranges.json", "w") as json_file:
            json.dump(self.sensor_ranges, json_file, indent=4)

        return self.min_limits, self.max_limits

    def read_sensor_range_from_json(self):

        with open("src/flex_sensor/data/sensor_ranges.json", "r") as json_file:
            loaded_dict = json.load(json_file)

        self.min_limits = loaded_dict["min_limits"]
        self.max_limits = loaded_dict["max_limits"]

    def initialize_steady_values(self):
        "Read sensor values in steady position"

        input("Leave the platform in the center and press Enter")

        self.get_steady_values()

    def get_steady_values(self):
        "Read values in steady position"
        steady_updated = False

        while not steady_updated:
            sensor_values = self.read_sensor()

            if None not in sensor_values:
                self.steady_values = sensor_values
                steady_updated = True
            else:
                self.logger().error(
                    "Sensor not connected properly when getting steady values"
                )

    def get_sensor_percentage(self):
        """
        Convert sensor measurement into normalized percentage
        """
        sensor_values = self.read_sensor()
        if None not in sensor_values:
            for i in range(self.n_sensors):
                # Read raw measurement
                ADC_i = sensor_values[i]

                # Convert it into a percentage
                if ADC_i > self.steady_values[i]:
                    self.max_limits[i] - self.steady_values[i]
                    ADC_i_percentage = (
                        50
                        + (
                            (ADC_i - self.steady_values[i])
                            / (self.max_limits[i] - self.steady_values[i])
                        )
                        * 100
                        / 2
                    )
                elif ADC_i < self.steady_values[i]:
                    ADC_i_percentage = (
                        50
                        - (
                            (self.steady_values[i] - ADC_i)
                            / (self.steady_values[i] - self.min_limits[i])
                        )
                        * 100
                        / 2
                    )
                else:
                    ADC_i_percentage = 50
                self.sensor_percent[i] = ADC_i_percentage
        else:
            self.sensor_percent = [None, None, None, None]

        return self.sensor_percent
