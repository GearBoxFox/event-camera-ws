import rclpy
from rclpy.node import Node
import cv2

from std_msgs.msg import String


class BiasTuner(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # List of default params used to check for updates
        self.params = [
            # bias_diff is read only
            221,  # bias_diff_off
            384,  # bias_diff_on
            1477, # bias_fo
            1499, # bias_hpf
            1250, # bias_pr
            1500 # bias_refr
        ]
        self.param_names = [
            "bias_diff_off",
            "bias_diff_on",
            "bias_fo",
            "bias_hpf",
            "bias_pr",
            "bias_refr"
        ]

    def timer_callback(self):
        # check for changes in parameters
        new_params = self.read_tuning_values()

        if not self.checkEqual(new_params, self.params):
            for i in range(len(new_params)):
                r

    def create_tuning_window(self):
        cv2.namedWindow("Tuning", 0)
        # all maximum values are pulled from https://docs.prophesee.ai/stable/hw/manuals/biases.html for a Gen 3.1 sensor
        cv2.createTrackbar("bias_diff_off", "Tuning", self.params[0], 499)
        cv2.createTrackbar("bias_diff_on", "Tuning", self.params[1], 214)
        cv2.createTrackbar("bias_fo", "Tuning", self.params[2], 1800)
        cv2.createTrackbar("bias_hpf", "Tuning", self.params[3], 1800)
        cv2.createTrackbar("bias_pr", "Tuning", self.params[4], 1800)

    def read_tuning_values(self):
        return [cv2.getTrackbarPos(key, "Tuning") for key in self.param_names]
    
    def checkEqual(a, b):  
        # If lengths of array are not equal means
        # array are not equal
        if len(a) != len(b):
            return False
        return sorted(a) == sorted(b)




def main(args=None):
    rclpy.init(args=args)

    bias_tuner = BiasTuner()

    rclpy.spin(bias_tuner)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    bias_tuner.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
