"""
02b_tune_exposure.py — Interactive exposure / laser tuner for the D405.

Three-panel live view (single OpenCV window):
  [Color image] | [Colorized depth] | [Validity mask — white=valid, black=invalid]

Numeric settings are overlaid on the color panel every frame.

Controls:
  e / r   depth-sensor exposure  - / +  (µs)
  g / h   depth-sensor gain      - / +
  l / ;   laser projector power  - / +  (mW)
  c / v   color-sensor exposure  - / +  (µs)
  w / q   color white balance    - / +  (K)
  p       print all current settings to terminal
  s       save settings to camera_settings.json (load in later scripts)
  Esc     quit

Run:
    python3 scripts/depth_camera/02b_tune_exposure.py
"""

import json
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

# ── stream config ─────────────────────────────────────────────────────────────
# 848x480 is not supported simultaneously for color+depth on this setup;
# 640x480 confirmed working in 01_check_camera.py.
STREAM_WIDTH  = 640
STREAM_HEIGHT = 480
FPS           = 30

# ── step sizes per keypress ───────────────────────────────────────────────────
STEP = {
    "depth_exposure": 100,   # µs
    "depth_gain":       1,
    "laser_power":     30,   # mW
    "color_exposure": 100,   # µs
    "white_balance":  100,   # K
}

SETTINGS_PATH = Path("camera_settings.json")


# ── sensor helpers ────────────────────────────────────────────────────────────
def find_color_sensor(device):
    """Return the RGB sensor — identified by carrying the white-balance option."""
    for s in device.query_sensors():
        if s.supports(rs.option.enable_auto_white_balance):
            return s
    return None


def adjust(sensor, option, delta):
    """Nudge an option by delta, snapping to the device's min/max/step."""
    try:
        r   = sensor.get_option_range(option)
        cur = sensor.get_option(option)
        stp = r.step if r.step > 0 else 1
        new = max(r.min, min(r.max, cur + delta))
        new = r.min + round((new - r.min) / stp) * stp
        sensor.set_option(option, new)
        return new
    except Exception as exc:
        print(f"  [!] Could not set option: {exc}")
        return None


def read_all(depth_sensor, color_sensor):
    """Return a dict of the six tunable settings (None if unsupported)."""
    pairs = [
        ("depth_exposure", depth_sensor, rs.option.exposure),
        ("depth_gain",     depth_sensor, rs.option.gain),
        ("laser_power",    depth_sensor, rs.option.laser_power),
        ("color_exposure", color_sensor, rs.option.exposure),
        ("white_balance",  color_sensor, rs.option.white_balance),
    ]
    out = {}
    for name, sensor, opt in pairs:
        try:
            out[name] = sensor.get_option(opt)
        except Exception:
            out[name] = None
    return out


def print_settings(depth_sensor, color_sensor):
    s = read_all(depth_sensor, color_sensor)
    print("\n── Current settings ───────────────────────────────")
    print(f"  depth_exposure : {_fmt(s['depth_exposure'], ' µs')}")
    print(f"  depth_gain     : {_fmt(s['depth_gain'])}")
    print(f"  laser_power    : {_fmt(s['laser_power'], ' mW')}  (N/A = not supported)")
    print(f"  color_exposure : {_fmt(s['color_exposure'], ' µs')}")
    print(f"  white_balance  : {_fmt(s['white_balance'], ' K')}")
    print("───────────────────────────────────────────────────\n")


def save_settings(depth_sensor, color_sensor):
    s = read_all(depth_sensor, color_sensor)
    SETTINGS_PATH.write_text(json.dumps(s, indent=2))
    print(f"  Saved → {SETTINGS_PATH.resolve()}")


def load_settings(depth_sensor, color_sensor):
    """Apply settings from camera_settings.json if it exists. Skip null values."""
    if not SETTINGS_PATH.exists():
        print("  No camera_settings.json found — using camera defaults.")
        return
    s = json.loads(SETTINGS_PATH.read_text())
    mapping = [
        ("depth_exposure", depth_sensor, rs.option.exposure),
        ("depth_gain",     depth_sensor, rs.option.gain),
        ("laser_power",    depth_sensor, rs.option.laser_power),
        ("color_exposure", color_sensor, rs.option.exposure),
        ("white_balance",  color_sensor, rs.option.white_balance),
    ]
    for key, sensor, opt in mapping:
        val = s.get(key)
        if val is None:
            continue
        try:
            sensor.set_option(opt, val)
            print(f"  Loaded {key} = {val}")
        except Exception as exc:
            print(f"  [!] Could not set {key}: {exc}")
    print()


# ── overlay ───────────────────────────────────────────────────────────────────
def _fmt(val, unit=""):
    return f"{val:.0f}{unit}" if val is not None else "N/A"


def draw_overlay(img, s):
    """Burn current settings into the top-left of img (in-place)."""
    lines = [
        f"D.exp  {_fmt(s['depth_exposure'], ' us')}   [e/r]",
        f"D.gain {_fmt(s['depth_gain'])}              [g/h]",
        f"Laser  {_fmt(s['laser_power'], ' mW')}  [l/;]",
        f"C.exp  {_fmt(s['color_exposure'], ' us')}   [c/v]",
        f"WB     {_fmt(s['white_balance'], ' K')}   [w/q]",
        f"  p=print  s=save  Esc=quit",
    ]
    font  = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.42
    lh    = 17
    pad   = 5
    box_h = len(lines) * lh + pad * 2
    box_w = 220

    roi     = img[2:2 + box_h, 2:2 + box_w]
    bg      = np.zeros_like(roi)
    img[2:2 + box_h, 2:2 + box_w] = cv2.addWeighted(roi, 0.35, bg, 0.65, 0)

    for i, line in enumerate(lines):
        color = (180, 255, 180) if i < 5 else (140, 140, 140)
        y = 2 + pad + (i + 1) * lh
        cv2.putText(img, line, (6, y), font, scale, color, 1, cv2.LINE_AA)


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    pipeline = rs.pipeline()
    cfg      = rs.config()
    cfg.enable_stream(rs.stream.color, STREAM_WIDTH, STREAM_HEIGHT, rs.format.bgr8, FPS)
    cfg.enable_stream(rs.stream.depth, STREAM_WIDTH, STREAM_HEIGHT, rs.format.z16,  FPS)

    print("Starting RealSense pipeline…")
    profile = pipeline.start(cfg)
    device  = profile.get_device()

    depth_sensor = device.first_depth_sensor()
    color_sensor = find_color_sensor(device)
    if color_sensor is None:
        print("[!] Color sensor not found — quitting.")
        pipeline.stop()
        return

    print(f"  Depth sensor : {depth_sensor.get_info(rs.camera_info.name)}")
    print(f"  Color sensor : {color_sensor.get_info(rs.camera_info.name)}")

    # Disable all automatic controls
    depth_sensor.set_option(rs.option.enable_auto_exposure,      0)
    color_sensor.set_option(rs.option.enable_auto_exposure,      0)
    color_sensor.set_option(rs.option.enable_auto_white_balance, 0)
    print("  Auto-exposure OFF (depth + color).  Auto-white-balance OFF.\n")

    print(f"  Loading settings from {SETTINGS_PATH}…")
    load_settings(depth_sensor, color_sensor)
    print_settings(depth_sensor, color_sensor)

    align     = rs.align(rs.stream.color)
    colorizer = rs.colorizer()
    colorizer.set_option(rs.option.color_scheme, 2)   # white-to-black

    WIN = "Color | Depth | Validity"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, STREAM_WIDTH * 3, STREAM_HEIGHT)

    try:
        while True:
            frames  = pipeline.wait_for_frames(timeout_ms=5000)
            aligned = align.process(frames)

            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color_img = np.asanyarray(color_frame.get_data())
            depth_raw = np.asanyarray(depth_frame.get_data())          # uint16
            depth_vis = np.asanyarray(colorizer.colorize(depth_frame).get_data())

            # validity mask: white where depth is non-zero, black elsewhere
            validity     = (depth_raw > 0).astype(np.uint8) * 255
            validity_bgr = cv2.cvtColor(validity, cv2.COLOR_GRAY2BGR)

            draw_overlay(color_img, read_all(depth_sensor, color_sensor))

            # label each panel
            for panel, label in [(color_img, "Color"),
                                  (depth_vis, "Depth"),
                                  (validity_bgr, "Validity (white=valid)")]:
                cv2.putText(panel, label,
                            (8, STREAM_HEIGHT - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

            combined = np.concatenate([color_img, depth_vis, validity_bgr], axis=1)
            cv2.imshow(WIN, combined)

            key = cv2.waitKey(1) & 0xFF
            if   key == 27:           break                                                              # Esc
            elif key == ord("e"): adjust(depth_sensor, rs.option.exposure,      -STEP["depth_exposure"])
            elif key == ord("r"): adjust(depth_sensor, rs.option.exposure,      +STEP["depth_exposure"])
            elif key == ord("g"): adjust(depth_sensor, rs.option.gain,          -STEP["depth_gain"])
            elif key == ord("h"): adjust(depth_sensor, rs.option.gain,          +STEP["depth_gain"])
            elif key == ord("l"): adjust(depth_sensor, rs.option.laser_power,   -STEP["laser_power"])
            elif key == ord(";"): adjust(depth_sensor, rs.option.laser_power,   +STEP["laser_power"])
            elif key == ord("c"): adjust(color_sensor, rs.option.exposure,      -STEP["color_exposure"])
            elif key == ord("v"): adjust(color_sensor, rs.option.exposure,      +STEP["color_exposure"])
            elif key == ord("w"): adjust(color_sensor, rs.option.white_balance, -STEP["white_balance"])
            elif key == ord("q"): adjust(color_sensor, rs.option.white_balance, +STEP["white_balance"])
            elif key == ord("p"): print_settings(depth_sensor, color_sensor)
            elif key == ord("s"): save_settings(depth_sensor, color_sensor)

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("Stopped.")


if __name__ == "__main__":
    main()