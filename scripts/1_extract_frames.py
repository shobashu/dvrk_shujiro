#!/usr/bin/env python3
"""
Extract frames from ROS 2 bag (.mcap) for YOLO training
python 1_extract_frames.py --bag compressed/test6 --out ~/dvrk_shujiro_ws/data/frames/trial_001 --fps 5
"""

import argparse
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore
import yaml

def resolve_bag_path(bag_input: str) -> str:
    """Convert relative bag name to absolute path."""
    bag_base = Path.home() / "dvrk_recordings"
    bag_path = Path(bag_input)

    if bag_path.is_absolute():
        return str(bag_path)

    resolved = bag_base / bag_path
    return str(resolved)


def decode_image(msg, is_compressed):
    if is_compressed:
        np_arr = np.frombuffer(msg.data, np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    else:
        height, width = msg.height, msg.width
        cv_image = np.frombuffer(msg.data, dtype=np.uint8).reshape(height, width, -1)
        if msg.encoding == 'rgb8':
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
        return cv_image


def extract_frames(bag_path, output_dir, target_fps=5, left_topic=None, right_topic=None):
    """Extract frames and kinematics from ROS 2 bag"""

    bag_path = Path(bag_path)
    output_dir = Path(output_dir)

    print(f"Reading bag: {bag_path}")
    print(f"Output directory: {output_dir}")
    print(f"Target FPS: {target_fps}")
    print()

    typestore = get_typestore(Stores.ROS2_HUMBLE)

    kinematics_data = []
    kinematics_matched = []
    last_frame_time_kinematic1 = None
    last_frame_time_kinematic2 = None
    frame_interval = 1.0 / target_fps

    default_left_topics = ['/camera_left/image_raw', '/camera/left/image_raw', '/camera/left/compressed']
    default_right_topics = ['/camera_right/image_raw', '/camera/right/image_raw', '/camera/right/compressed']

    with Reader(bag_path) as reader:
        available_topics = {conn.topic for conn in reader.connections}
        print("Available topics in bag:")
        for topic in sorted(available_topics):
            print(f"   {topic}")
        print()

        # Resolve left topic
        if left_topic:
            if left_topic not in available_topics:
                print(f"ERROR: Left topic '{left_topic}' not found!")
                return
            final_left = left_topic
        else:
            final_left = next((t for t in default_left_topics if t in available_topics), None)

        # Resolve right topic
        if right_topic:
            if right_topic not in available_topics:
                print(f"ERROR: Right topic '{right_topic}' not found!")
                return
            final_right = right_topic
        else:
            final_right = next((t for t in default_right_topics if t in available_topics), None)

        if not final_left and not final_right:
            print("ERROR: No camera topics found! Use --left-topic / --right-topic to specify manually.")
            return

        # Build per-topic state
        camera_topics = {}
        for side, topic in [('left', final_left), ('right', final_right)]:
            if topic is None:
                continue
            images_dir = output_dir / "images" / side
            images_dir.mkdir(parents=True, exist_ok=True)
            is_compressed = any('CompressedImage' in conn.msgtype for conn in reader.connections if conn.topic == topic)
            msgtype = next(conn.msgtype for conn in reader.connections if conn.topic == topic)
            camera_topics[topic] = {
                'side': side,
                'images_dir': images_dir,
                'is_compressed': is_compressed,
                'frame_count': 0,
                'last_frame_time': None,
            }
            print(f"Using {side} camera topic: {topic}  ({msgtype})")
        print()

        print("Processing messages...")
        print()

        for connection, timestamp, rawdata in reader.messages():
            msg = typestore.deserialize_cdr(rawdata, connection.msgtype)

            if connection.topic in camera_topics:
                state = camera_topics[connection.topic]
                msg_time = timestamp / 1e9

                if state['last_frame_time'] is None or (msg_time - state['last_frame_time']) >= frame_interval:
                    try:
                        cv_image = decode_image(msg, state['is_compressed'])
                        frame_count = state['frame_count']
                        frame_filename = f"frame_{frame_count:04d}.jpg"
                        frame_path = state['images_dir'] / frame_filename
                        cv2.imwrite(str(frame_path), cv_image)

                        if frame_count % 10 == 0:
                            print(f"   [{state['side']}] frame {frame_count}: {frame_filename}")

                        state['frame_count'] += 1
                        state['last_frame_time'] = msg_time

                    except Exception as e:
                        print(f"Warning: Failed to decode [{state['side']}] frame at {msg_time:.2f}s: {e}")
            
            elif connection.topic == '/PSM1/measured_cp':
                if last_frame_time_kinematic1 is None or ((timestamp / 1e9) - last_frame_time_kinematic1) >= frame_interval:
                    kinematics_matched.append({
                        'timestamp': timestamp / 1e9,
                        'topic': 'measured_cp',
                        'x': msg.pose.position.x,      # Changed from msg.transform.translation.x
                        'y': msg.pose.position.y,
                        'z': msg.pose.position.z,
                        'qx': msg.pose.orientation.x,  # Changed from msg.transform.rotation.x
                        'qy': msg.pose.orientation.y,
                        'qz': msg.pose.orientation.z,
                        'qw': msg.pose.orientation.w,
                    })
                    last_frame_time_kinematic1 = timestamp / 1e9

                kinematics_data.append({
                    'timestamp': timestamp / 1e9,
                    'topic': 'measured_cp',
                    'x': msg.pose.position.x,      # Changed from msg.transform.translation.x
                    'y': msg.pose.position.y,
                    'z': msg.pose.position.z,
                    'qx': msg.pose.orientation.x,  # Changed from msg.transform.rotation.x
                    'qy': msg.pose.orientation.y,
                    'qz': msg.pose.orientation.z,
                    'qw': msg.pose.orientation.w,
                })

            elif connection.topic == '/PSM2/measured_cp':
                if last_frame_time_kinematic2 is None or ((timestamp / 1e9) - last_frame_time_kinematic2) >= frame_interval:
                    kinematics_matched.append({
                        'timestamp': timestamp / 1e9,
                        'topic': 'measured_cp_psm2',
                        'x': msg.pose.position.x,      # Changed from msg.transform.translation.x
                        'y': msg.pose.position.y,
                        'z': msg.pose.position.z,
                        'qx': msg.pose.orientation.x,  # Changed from msg.transform.rotation.x
                        'qy': msg.pose.orientation.y,
                        'qz': msg.pose.orientation.z,
                        'qw': msg.pose.orientation.w,
                    })
                    last_frame_time_kinematic2 = timestamp / 1e9

                kinematics_data.append({
                    'timestamp': timestamp / 1e9,
                    'topic': 'measured_cp_psm2',
                    'x': msg.pose.position.x,
                    'y': msg.pose.position.y,
                    'z': msg.pose.position.z,
                    'qx': msg.pose.orientation.x,
                    'qy': msg.pose.orientation.y,
                    'qz': msg.pose.orientation.z,
                    'qw': msg.pose.orientation.w,
                })
            
            elif connection.topic == '/PSM1/jaw/measured_js':
                kinematics_data.append({
                    'timestamp': timestamp / 1e9,
                    'topic': 'jaw',
                    'jaw_angle': msg.position[0] if len(msg.position) > 0 else 0.0,
                    'jaw_velocity': msg.velocity[0] if len(msg.velocity) > 0 else 0.0,
                })
            
            elif connection.topic == '/console/clutch':
                kinematics_data.append({
                    'timestamp': timestamp / 1e9,
                    'topic': 'clutch',
                    'clutch_pressed': msg.data,
                })
    
    print()
    for state in camera_topics.values():
        print(f"Extracted {state['frame_count']} frames [{state['side']}]  -> {state['images_dir']}")

    if kinematics_data:
        df = pd.DataFrame(kinematics_data)
        csv_path = output_dir / "kinematics.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved kinematics data: {csv_path}  ({len(df)} samples)")

    if kinematics_matched:
        df = pd.DataFrame(kinematics_matched)
        csv_path = output_dir / "kinematics_matched.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved matched kinematics data: {csv_path}  ({len(df)} samples)")

    metadata = {
        'bag_file': str(bag_path),
        'extraction_date': pd.Timestamp.now().isoformat(),
        'target_fps': target_fps,
        'camera_topics': {state['side']: topic for topic, state in camera_topics.items()},
        'frames_extracted': {state['side']: state['frame_count'] for state in camera_topics.values()},
    }

    metadata_path = output_dir / "metadata.yaml"
    with open(metadata_path, 'w') as f:
        yaml.dump(metadata, f)
    print(f"Saved metadata: {metadata_path}")

    print()
    print("Extraction complete!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract frames from ROS 2 bag')
    parser.add_argument('--bag', required=True, help='Bag name or path relative to ~/dvrk_recordings')
    parser.add_argument('--out', required=True, help='Output directory')
    parser.add_argument('--fps', type=float, default=2.0, help='Target frames per second (default: 2)')
    parser.add_argument('--left-topic', type=str, default=None, help='Left camera topic (auto-detect if not specified)')
    parser.add_argument('--right-topic', type=str, default=None, help='Right camera topic (auto-detect if not specified)')

    args = parser.parse_args()
    bag_path = resolve_bag_path(args.bag)
    extract_frames(bag_path, args.out, args.fps, args.left_topic, args.right_topic)