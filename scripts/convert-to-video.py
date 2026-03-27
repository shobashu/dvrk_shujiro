#!/usr/bin/env python3

import sys
import cv2
from pathlib import Path
from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr
import numpy as np

def bag_to_video(bag_path, output_video):
    """Convert ROS2 bag to MP4 video"""
    
    print(f"Converting: {bag_path}")
    print(f"Output: {output_video}")
    
    writer = None
    frame_count = 0
    
    with Reader(bag_path) as reader:
        # Get connections for left camera
        connections = [x for x in reader.connections if x.topic == '/camera/left/image_raw']
        
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            # Deserialize image message
            msg = deserialize_cdr(rawdata, connection.msgtype)
            
            # Convert to numpy array
            img_array = np.frombuffer(msg.data, dtype=np.uint8)
            img = img_array.reshape((msg.height, msg.width, 3))
            
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            # Initialize video writer on first frame
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(output_video, fourcc, 30.0, 
                                        (msg.width, msg.height))
            
            # Write frame
            writer.write(img_bgr)
            frame_count += 1
            
            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames...")
    
    if writer:
        writer.release()
    
    print(f"\nDone! {frame_count} frames written to {output_video}")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: ./convert-to-video.py <recording_name>")
        print("Example: ./convert-to-video.py trial_01")
        sys.exit(1)
    
    recording_name = sys.argv[1]
    bag_dir = Path.home() / 'dvrk_recordings' / recording_name
    output_file = str(bag_dir.parent / f"{recording_name}_left.mp4")
    
    if not bag_dir.exists():
        print(f"Error: Recording not found: {bag_dir}")
        sys.exit(1)
    
    bag_to_video(str(bag_dir), output_file)

