"""
RealSense D405 tuner — keyboard controls in the preview window:
  e / E  : exposure  -100 / +100 µs
  g / G  : gain      -1   / +1
  q      : quit and print final values
"""
import cv2
import numpy as np
import pyrealsense2 as rs

WIDTH, HEIGHT, FPS = 640, 480, 30

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.infrared, 1, WIDTH, HEIGHT, rs.format.y8, FPS)
config.enable_stream(rs.stream.depth,       WIDTH, HEIGHT, rs.format.z16, FPS)
profile = pipeline.start(config)

sensor = profile.get_device().query_sensors()[0]
sensor.set_option(rs.option.enable_auto_exposure, 0)

EXP_MIN,  EXP_MAX  = 1,  165000
GAIN_MIN, GAIN_MAX = 16, 248

exposure = 8500
gain     = 16
sensor.set_option(rs.option.exposure, exposure)
sensor.set_option(rs.option.gain,     gain)

print("Controls (click the preview window first):")
print("  e / r  ->  exposure  -100 / +100 us")
print("  g / h  ->  gain      -1   / +1")
print("  q      ->  quit")
print(f"\nStarting:  exposure={exposure}  gain={gain}\n")

WIN = "D405 Tuner"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
colorizer = rs.colorizer()
colorizer.set_option(rs.option.color_scheme, 2)

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

try:
    while True:
        frames      = pipeline.wait_for_frames(timeout_ms=5000)
        ir_frame    = frames.get_infrared_frame(1)
        depth_frame = frames.get_depth_frame()
        if not ir_frame or not depth_frame:
            continue

        ir_img    = np.asanyarray(ir_frame.get_data())
        depth_img = np.asanyarray(colorizer.colorize(depth_frame).get_data())

        ir_norm = cv2.normalize(ir_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        ir_bgr  = cv2.cvtColor(ir_norm, cv2.COLOR_GRAY2BGR)

        cv2.putText(ir_bgr, f"Exposure: {exposure} us  (e=down  r=up)",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(ir_bgr, f"Gain:     {gain}  (g=down  h=up)",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow(WIN, np.hstack([ir_bgr, depth_img]))

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('e'):
            exposure = clamp(exposure - 100, EXP_MIN, EXP_MAX)
            sensor.set_option(rs.option.exposure, exposure)
            print(f"exposure={exposure}  gain={gain}")
        elif key == ord('r'):
            exposure = clamp(exposure + 100, EXP_MIN, EXP_MAX)
            sensor.set_option(rs.option.exposure, exposure)
            print(f"exposure={exposure}  gain={gain}")
        elif key == ord('g'):
            gain = clamp(gain - 1, GAIN_MIN, GAIN_MAX)
            sensor.set_option(rs.option.gain, gain)
            print(f"exposure={exposure}  gain={gain}")
        elif key == ord('h'):
            gain = clamp(gain + 1, GAIN_MIN, GAIN_MAX)
            sensor.set_option(rs.option.gain, gain)
            print(f"exposure={exposure}  gain={gain}")
finally:
    print(f"\nFinal values:  exposure={exposure}  gain={gain}")
    pipeline.stop()
    cv2.destroyAllWindows()
