"""
02_live_viewer.py — Live color + depth viewer for the Intel RealSense D405.

Displays color and colorized depth side by side.
Left-click anywhere in the color window to print the pixel coordinates,
depth in metres, and the deprojected 3D point in the camera frame.

Controls:
  left-click   print (u, v), depth, and (X, Y, Z)
  q            quit

Run:
    python3 scripts/depth_camera/02_live_viewer.py
"""

import cv2
import numpy as np
import pyrealsense2 as rs


STREAM_WIDTH  = 640
STREAM_HEIGHT = 480
FPS           = 30
DEPTH_SAMPLE_RADIUS = 3   # median over (2r+1)^2 patch to reduce noise


# ── shared state (filled once pipeline starts) ────────────────────────────────
state = {
    "depth_frame": None,
    "intrinsics":  None,
}


def get_median_depth(depth_frame, u, v, radius=DEPTH_SAMPLE_RADIUS):
    h, w = depth_frame.get_height(), depth_frame.get_width()
    u0, u1 = max(0, u - radius), min(w - 1, u + radius)
    v0, v1 = max(0, v - radius), min(h - 1, v + radius)
    depths = [
        depth_frame.get_distance(col, row)
        for row in range(v0, v1 + 1)
        for col in range(u0, u1 + 1)
        if depth_frame.get_distance(col, row) > 0
    ]
    return float(np.median(depths)) if depths else 0.0


def on_mouse(event, u, v, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    depth_frame = state["depth_frame"]
    intrinsics  = state["intrinsics"]
    if depth_frame is None or intrinsics is None:
        return

    depth_m = get_median_depth(depth_frame, u, v)
    if depth_m <= 0:
        print(f"  click ({u:4d}, {v:4d})  →  no valid depth (too close / reflective)")
        return

    xyz = rs.rs2_deproject_pixel_to_point(intrinsics, [float(u), float(v)], depth_m)
    X, Y, Z = xyz
    print(f"  click ({u:4d}, {v:4d})  →  depth={depth_m:.4f} m  |  "
          f"X={X:+.4f}  Y={Y:+.4f}  Z={Z:.4f} m  (camera frame)")


def main():
    pipeline = rs.pipeline()
    config   = rs.config()
    config.enable_stream(rs.stream.color, STREAM_WIDTH, STREAM_HEIGHT, rs.format.bgr8, FPS)
    config.enable_stream(rs.stream.depth, STREAM_WIDTH, STREAM_HEIGHT, rs.format.z16,  FPS)

    print("Starting RealSense pipeline…")
    profile = pipeline.start(config)

    state["intrinsics"] = (
        profile.get_stream(rs.stream.color)
               .as_video_stream_profile()
               .get_intrinsics()
    )

    align     = rs.align(rs.stream.color)
    colorizer = rs.colorizer()
    colorizer.set_option(rs.option.color_scheme, 2)   # white-to-black

    cv2.namedWindow("Color", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Depth", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Color", STREAM_WIDTH, STREAM_HEIGHT)
    cv2.resizeWindow("Depth", STREAM_WIDTH, STREAM_HEIGHT)
    cv2.setMouseCallback("Color", on_mouse)

    print("Streaming. Left-click in 'Color' window to probe depth. 'q' to quit.\n")

    try:
        while True:
            frames  = pipeline.wait_for_frames(timeout_ms=5000)
            aligned = align.process(frames)

            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            state["depth_frame"] = depth_frame

            color_img = np.asanyarray(color_frame.get_data())
            depth_vis = np.asanyarray(colorizer.colorize(depth_frame).get_data())

            cv2.imshow("Color", color_img)
            cv2.imshow("Depth", depth_vis)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("Stopped.")


if __name__ == "__main__":
    main()
