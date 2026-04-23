#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


class ResizeNode(Node):
    def __init__(self):
        super().__init__('resize_images')

        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)

        w = int(self.get_parameter('width').value)
        h = int(self.get_parameter('height').value)

        self.bridge = CvBridge()

        self.pub_left = self.create_publisher(Image, '/camera_left/image_resized', qos_profile_sensor_data)
        self.pub_right = self.create_publisher(Image, '/camera_right/image_resized', qos_profile_sensor_data)

        self.sub_left = self.create_subscription(
            Image, '/camera_left/image_raw', self.cb_left, qos_profile_sensor_data
        )
        self.sub_right = self.create_subscription(
            Image, '/camera_right/image_raw', self.cb_right, qos_profile_sensor_data
        )

        self.w = w
        self.h = h
        self.get_logger().info(f"Resizing to {self.w}x{self.h}")

    def _resize_and_publish(self, msg: Image, pub):
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        resized = cv2.resize(cv_img, (self.w, self.h), interpolation=cv2.INTER_AREA)
        out = self.bridge.cv2_to_imgmsg(resized, encoding='bgr8')
        out.header = msg.header
        pub.publish(out)

    def cb_left(self, msg: Image):
        self._resize_and_publish(msg, self.pub_left)

    def cb_right(self, msg: Image):
        self._resize_and_publish(msg, self.pub_right)


def main():
    rclpy.init()
    node = ResizeNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()