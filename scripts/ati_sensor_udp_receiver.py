#!/usr/bin/env python3
"""
ATI force/torque sensor — UDP receiver, ROS2 publisher, and CSV logger.

Listens for 28-byte UDP packets from the ATI sensor sender:
    <i6f>  device_idx (int32) + fx, fy, fz, tx, ty, tz (6 × float32)

Publishes to /ati_sensor/wrench (geometry_msgs/WrenchStamped) and logs
every sample to a timestamped CSV file.

Usage:
    python3 ati_sensor_udp_receiver.py
    python3 ati_sensor_udp_receiver.py --ros-args -p udp_port:=5006
    python3 ati_sensor_udp_receiver.py --ros-args -p udp_ip:=192.168.1.10 -p udp_port:=5006
"""

import csv
import socket
import struct
import time
from datetime import datetime

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import WrenchStamped

ATI_PACKET_SIZE = 28  # 4-byte int + 6 × 4-byte float


class AtiSensorUdpReceiver(Node):
    def __init__(self):
        super().__init__("ati_sensor_udp_receiver")

        self.declare_parameter("udp_ip",   "0.0.0.0")
        self.declare_parameter("udp_port", 5006)
        self.declare_parameter("frame_id", "ati_sensor")

        udp_ip   = self.get_parameter("udp_ip").get_parameter_value().string_value
        udp_port = self.get_parameter("udp_port").get_parameter_value().integer_value
        self.frame_id = self.get_parameter("frame_id").get_parameter_value().string_value

        self.wrench_pub = self.create_publisher(WrenchStamped, "/ati_sensor/wrench", 10)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((udp_ip, udp_port))
        self.sock.settimeout(0.1)

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_filename = f"ati_sensor_{timestamp_str}.csv"
        self.csv_file = open(self.csv_filename, mode="w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["timestamp_s", "fx", "fy", "fz", "tx", "ty", "tz"])

        self.get_logger().info(
            f"Listening on {udp_ip}:{udp_port} | logging to {self.csv_filename}"
        )

        self.create_timer(0.001, self._read_and_publish)

    def _read_and_publish(self):
        try:
            data, _ = self.sock.recvfrom(1024)
        except socket.timeout:
            return
        except OSError as exc:
            self.get_logger().error(f"Socket error: {exc}")
            return

        if len(data) != ATI_PACKET_SIZE:
            self.get_logger().warning(f"Unexpected packet length {len(data)}, expected {ATI_PACKET_SIZE}")
            return

        device_idx, fx, fy, fz, tx, ty, tz = struct.unpack("<i6f", data)

        if device_idx != -1:
            self.get_logger().warning(f"Unexpected device_idx {device_idx} (expected -1 for ATI sensor)")
            return

        now = time.time()

        msg = WrenchStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.wrench.force.x  = fx
        msg.wrench.force.y  = fy
        msg.wrench.force.z  = fz
        msg.wrench.torque.x = tx
        msg.wrench.torque.y = ty
        msg.wrench.torque.z = tz
        self.wrench_pub.publish(msg)

        self.csv_writer.writerow([now, fx, fy, fz, tx, ty, tz])

    def destroy_node(self):
        try:
            if not self.csv_file.closed:
                self.csv_file.close()
                print(f"Saved {self.csv_filename}")
            self.sock.close()
        finally:
            super().destroy_node()


def main():
    rclpy.init()
    node = AtiSensorUdpReceiver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
