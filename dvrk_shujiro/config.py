"""Configuration constants for dVRK timer"""

# Timer settings
TIMER_RATE_HZ = 200
TIMER_INTERVAL = 1.0 / TIMER_RATE_HZ
MAX_TIME_SEC = 120

# GUI settings
WINDOW_ALPHA = 0.1
WINDOW_WIDTH = 340
WINDOW_HEIGHT = 95

# Display settings
FONT_STATUS = ("Arial", 8)
FONT_TIME = ("Arial", 18, "bold")
FONT_PATH = ("Arial", 10)

# Colors
COLOR_GREEN = "green"
COLOR_ORANGE = "orange"
COLOR_RED = "red"
COLOR_PSM1 = "blue"
COLOR_PSM2 = "purple"

# Progress thresholds (%)
PROGRESS_YELLOW_THRESHOLD = 70
PROGRESS_RED_THRESHOLD = 90

# ROS topics
TOPIC_TELEOP_ENABLED = '/console/teleop/enabled'
TOPIC_OPERATOR_PRESENT = '/console/operator_present'
TOPIC_PSM1_POSE = '/PSM1/measured_cp'
TOPIC_PSM2_POSE = '/PSM2/measured_cp'
