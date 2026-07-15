"""
01_check_camera.py — Verify Intel RealSense D405 connection and basic stream.

Streams color + aligned depth at 848x480 @ 30 fps for 5 seconds, then prints:
  - Depth scale (metres per depth unit)
  - Color intrinsics
  - Depth intrinsics
  - Median depth of the 50x50 window at the image centre

Run from the repo root:
    python3 scripts/depth_camera/01_check_camera.py
"""

import time
import numpy as np
import pyrealsense2 as rs


STREAM_WIDTH  = 640
STREAM_HEIGHT = 480
FPS           = 30
DURATION_SEC  = 5
WINDOW_HALF   = 25   # half-side of the centre sample window (50x50 total)


def fmt_intrinsics(intr):
    return (
        f"  size   : {intr.width} x {intr.height}\n"
        f"  fx, fy : {intr.fx:.4f}, {intr.fy:.4f}\n"
        f"  cx, cy : {intr.ppx:.4f}, {intr.ppy:.4f}\n"
        f"  distortion model : {intr.model}\n"
        f"  coeffs : {[round(c, 6) for c in intr.coeffs]}"
    )


def main():
    pipeline = rs.pipeline()
    config   = rs.config()
    config.enable_stream(rs.stream.color, STREAM_WIDTH, STREAM_HEIGHT,
                         rs.format.bgr8, FPS)
    config.enable_stream(rs.stream.depth, STREAM_WIDTH, STREAM_HEIGHT,
                         rs.format.z16,  FPS)

    print("Starting RealSense pipeline…")
    profile = pipeline.start(config)

    # Depth scale: converts raw uint16 depth units → metres
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale  = depth_sensor.get_depth_scale()
    print(f"\nDepth scale : {depth_scale:.6f} m/unit  "
          f"({1.0/depth_scale:.0f} units/m)\n")

    # Intrinsics
    color_intr = (profile.get_stream(rs.stream.color)
                          .as_video_stream_profile()
                          .get_intrinsics())
    depth_intr = (profile.get_stream(rs.stream.depth)
                          .as_video_stream_profile()
                          .get_intrinsics())

    print("Color intrinsics:")
    print(fmt_intrinsics(color_intr))
    print("\nDepth intrinsics:")
    print(fmt_intrinsics(depth_intr))

    align = rs.align(rs.stream.color)

    cx = STREAM_WIDTH  // 2
    cy = STREAM_HEIGHT // 2
    x0, x1 = cx - WINDOW_HALF, cx + WINDOW_HALF
    y0, y1 = cy - WINDOW_HALF, cy + WINDOW_HALF

    print(f"\nStreaming for {DURATION_SEC} s — sampling depth at "
          f"centre window [{x0}:{x1}, {y0}:{y1}]…\n")

    t_start = time.time()
    frame_count = 0

    try:
        while time.time() - t_start < DURATION_SEC:
            frames  = pipeline.wait_for_frames(timeout_ms=5000)
            aligned = align.process(frames)

            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            frame_count += 1

            # Sample centre window — depth_frame.get_distance returns metres
            depth_arr = np.asanyarray(depth_frame.get_data())  # uint16 raw units
            window    = depth_arr[y0:y1, x0:x1].astype(float) * depth_scale
            valid     = window[window > 0]
            median_m  = float(np.median(valid)) if valid.size else float("nan")

            elapsed = time.time() - t_start
            print(f"  t={elapsed:5.2f}s  frame={frame_count:3d}  "
                  f"centre median depth = {median_m:.4f} m  "
                  f"({valid.size}/{window.size} valid pixels)")

    finally:
        pipeline.stop()
        print(f"\nStopped after {frame_count} frames "
              f"({frame_count / DURATION_SEC:.1f} fps). Done.")


if __name__ == "__main__":
    main()
