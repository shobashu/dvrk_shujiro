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


def extract_frames(bag_path, output_dir, target_fps=5, camera_topic=None):
    """Extract frames and kinematics from ROS 2 bag"""
    
    bag_path = Path(bag_path)
    output_dir = Path(output_dir)
    
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📂 Reading bag: {bag_path}")
    print(f"💾 Output directory: {output_dir}")
    print(f"🎞️  Target FPS: {target_fps}")
    if camera_topic:
        print(f"📹 Specified camera topic: {camera_topic}")
    print()
    
    # Get typestore for deserialization
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    
    kinematics_data = []
    kinematics_matched = []  # kinematic data with matching pictures
    frame_count = 0
    last_frame_time = None
    last_frame_time_kinematic1 = None
    last_frame_time_kinematic2 = None
    frame_interval = 1.0 / target_fps
    
    default_camera_topics = [
        # '/camera/left/compressed',
        # '/camera_left/compressed',
        # '/camera/left/image_raw',
        '/camera_right/compressed',
        '/camera_right/image_raw'
        # '/endoscope/left/compressed',
        # '/stereo/left/compressed',
        # '/jhu_crsus/left/image_raw',
    ]
    
    with Reader(bag_path) as reader:
        available_topics = {conn.topic for conn in reader.connections}
        print("📋 Available topics in bag:")
        for topic in sorted(available_topics):
            print(f"   {topic}")
        print()
        
        if camera_topic:
            if camera_topic not in available_topics:
                print(f"❌ ERROR: Specified topic '{camera_topic}' not found!")
                return
            final_camera_topic = camera_topic
        else:
            final_camera_topic = None
            for topic in default_camera_topics:
                if topic in available_topics:
                    final_camera_topic = topic
                    break
            
            if not final_camera_topic:
                print("❌ ERROR: No camera topic found!")
                print(f"   Use --topic to specify manually")
                return
        
        print(f"✅ Using camera topic: {final_camera_topic}")
        
        is_compressed = False
        for conn in reader.connections:
            if conn.topic == final_camera_topic:
                print(f"   Message type: {conn.msgtype}")
                is_compressed = 'CompressedImage' in conn.msgtype
                break
        print()
        
        print("🔄 Processing messages...")
        print()
        
        for connection, timestamp, rawdata in reader.messages():
            # Deserialize using typestore
            msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
            
            if connection.topic == final_camera_topic:
                msg_time = timestamp / 1e9
                
                if last_frame_time is None or (msg_time - last_frame_time) >= frame_interval:
                    try:
                        if is_compressed:
                            np_arr = np.frombuffer(msg.data, np.uint8)
                            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                        else:
                            height, width = msg.height, msg.width
                            cv_image = np.frombuffer(msg.data, dtype=np.uint8).reshape(height, width, -1)
                            if msg.encoding == 'rgb8':
                                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
                        
                        frame_filename = f"frame_{frame_count:04d}.jpg"
                        frame_path = images_dir / frame_filename
                        cv2.imwrite(str(frame_path), cv_image)
                        
                        if frame_count % 10 == 0:
                            print(f"   Extracted frame {frame_count}: {frame_filename}")
                        
                        frame_count += 1
                        last_frame_time = msg_time
                        
                    except Exception as e:
                        print(f"⚠️  Warning: Failed to decode frame at {msg_time:.2f}s: {e}")
            
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
    print(f"✅ Extracted {frame_count} frames")
    
    if kinematics_data:
        df = pd.DataFrame(kinematics_data)
        csv_path = output_dir / "kinematics.csv"
        df.to_csv(csv_path, index=False)
        print(f"✅ Saved kinematics data: {csv_path}")
        print(f"   Total kinematics samples: {len(df)}")
    
    if kinematics_matched:
        df = pd.DataFrame(kinematics_matched)
        csv_path = output_dir / "kinematics_matched.csv"
        df.to_csv(csv_path, index=False)
        print(f"✅ Saved matched kinematics data: {csv_path}")
        print(f"   Total matched kinematics samples: {len(df)}")
    
    metadata = {
        'bag_file': str(bag_path),
        'extraction_date': pd.Timestamp.now().isoformat(),
        'target_fps': target_fps,
        'camera_topic': final_camera_topic,
        'frames_extracted': frame_count,
    }
    
    metadata_path = output_dir / "metadata.yaml"
    with open(metadata_path, 'w') as f:
        yaml.dump(metadata, f)
    print(f"✅ Saved metadata: {metadata_path}")
    
    print()
    print("🎉 Extraction complete!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract frames from ROS 2 bag')
    parser.add_argument('--bag', required=True, help='Bag name or path relative to ~/dvrk_recordings')
    parser.add_argument('--out', required=True, help='Output directory')
    parser.add_argument('--fps', type=float, default=2.0, help='Target frames per second (default: 2)')
    parser.add_argument('--topic', type=str, default=None, help='Camera topic name (auto-detect if not specified)')
    
    args = parser.parse_args()
    bag_path = resolve_bag_path(args.bag)
    extract_frames(bag_path, args.out, args.fps, args.topic)