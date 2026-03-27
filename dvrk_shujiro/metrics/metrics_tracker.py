"""Metrics tracking for position and orientation"""
import math
from ..utils.quaternion_math import quaternion_conjugate, quaternion_multiply, quaternion_to_angle


class MetricsTracker:
    """Tracks path length and orientation metrics for a single PSM"""
    
    def __init__(self, name="PSM"):
        self.name = name
        
        # Path tracking
        self.path_length = 0.0  # meters
        self.last_position = None
        self.path_sample_count = 0
        
        # Orientation tracking
        self.angular_displacement = 0.0  # radians
        self.angle_time_sum = 0.0
        self.last_orientation = None
        self.last_timestamp = None
        self.orientation_sample_count = 0
    
    def update_position(self, position):
        """Update path length with new position [x, y, z]"""
        if self.last_position is not None:
            dx = position[0] - self.last_position[0]
            dy = position[1] - self.last_position[1]
            dz = position[2] - self.last_position[2]
            distance = math.sqrt(dx*dx + dy*dy + dz*dz)
            
            self.path_length += distance
            self.path_sample_count += 1
        
        self.last_position = position
    
    def update_orientation(self, orientation, timestamp):
        """Update orientation metrics with new quaternion [x, y, z, w]"""
        if self.last_orientation is not None and self.last_timestamp is not None:
            q_j_inv = quaternion_conjugate(self.last_orientation)
            q_diff = quaternion_multiply(orientation, q_j_inv)
            theta = quaternion_to_angle(q_diff)
            dt = timestamp - self.last_timestamp
            
            if dt > 0:
                self.angular_displacement += theta
                self.angle_time_sum += theta / dt
                self.orientation_sample_count += 1
        
        self.last_orientation = orientation
        self.last_timestamp = timestamp
    
    def get_path_mm(self):
        """Get path length in millimeters"""
        return self.path_length * 1000.0
    
    def get_angular_displacement_rad(self):
        """Get angular displacement in radians"""
        return self.angular_displacement
    
    def get_angular_displacement_deg(self):
        """Get angular displacement in degrees"""
        return math.degrees(self.angular_displacement)
    
    def get_orientation_rate_rad(self):
        """Get average orientation rate in rad/s"""
        if self.orientation_sample_count == 0:
            return 0.0
        return self.angle_time_sum / self.orientation_sample_count
    
    def get_orientation_rate_deg(self):
        """Get average orientation rate in °/s"""
        return math.degrees(self.get_orientation_rate_rad())
    
    def reset(self):
        """Reset all metrics"""
        self.path_length = 0.0
        self.last_position = None
        self.path_sample_count = 0
        
        self.angular_displacement = 0.0
        self.angle_time_sum = 0.0
        self.last_orientation = None
        self.last_timestamp = None
        self.orientation_sample_count = 0