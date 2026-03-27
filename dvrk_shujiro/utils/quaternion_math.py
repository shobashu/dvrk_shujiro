"""Quaternion mathematics utilities"""
import math


def quaternion_conjugate(q):
    """Return conjugate of quaternion [x, y, z, w]"""
    return [-q[0], -q[1], -q[2], q[3]]


def quaternion_multiply(q1, q2):
    """Multiply two quaternions: q1 * q2"""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    
    return [
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ]


def quaternion_to_angle(q):
    """Extract rotation angle θ from quaternion [x, y, z, w]"""
    w = q[3]
    w = max(-1.0, min(1.0, w))
    theta = 2.0 * math.acos(w)
    return theta