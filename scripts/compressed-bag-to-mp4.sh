#!/usr/bin/env bash

if [ -z "$1" ]; then
    echo "Usage: ./compressed-bag-to-mp4.sh <recording_name> [left|right|both]"
    echo "Example: ./compressed-bag-to-mp4.sh trial_01_compressed both"
    exit 1
fi

RECORDING=$1
CAMERA=${2:-both}

BAG_DIR=~/dvrk_recordings/compressed/$RECORDING
OUTPUT_DIR=~/dvrk_recordings/compressed
TEMP_DIR=/tmp/bag_to_mp4_$$

if [ ! -d "$BAG_DIR" ]; then
    echo "Error: Recording not found: $BAG_DIR"
    exit 1
fi

source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash

echo "Converting compressed bag to MP4..."
echo ""

# Create Python script to extract compressed images
cat > /tmp/extract_compressed.py << 'PYEOF'
#!/usr/bin/env python3

import sys
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import os

class CompressedImageExtractor(Node):
    def __init__(self, topic, output_dir):
        super().__init__('image_extractor')
        self.output_dir = output_dir
        self.frame_count = 0
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.subscription = self.create_subscription(
            CompressedImage,
            topic,
            self.callback,
            10
        )
        
        print(f"Extracting from {topic} to {output_dir}")
    
    def callback(self, msg):
        filename = os.path.join(self.output_dir, f"frame_{self.frame_count:06d}.jpg")
        
        # Write JPEG data directly
        with open(filename, 'wb') as f:
            f.write(msg.data)
        
        self.frame_count += 1
        
        if self.frame_count % 100 == 0:
            print(f"Extracted {self.frame_count} frames")

def main():
    rclpy.init()
    
    if len(sys.argv) < 3:
        print("Usage: extract_compressed.py <topic> <output_dir>")
        sys.exit(1)
    
    topic = sys.argv[1]
    output_dir = sys.argv[2]
    
    extractor = CompressedImageExtractor(topic, output_dir)
    
    try:
        rclpy.spin(extractor)
    except KeyboardInterrupt:
        pass
    finally:
        print(f"\nTotal frames extracted: {extractor.frame_count}")
        try:
            extractor.destroy_node()
        except:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except:
            pass

if __name__ == '__main__':
    main()
PYEOF

chmod +x /tmp/extract_compressed.py

convert_camera() {
    local CAMERA_NAME=$1
    local TOPIC="/camera_${CAMERA_NAME}/compressed"
    local OUTPUT_VIDEO="${OUTPUT_DIR}/${RECORDING}_${CAMERA_NAME}.mp4"
    local FRAME_DIR="${TEMP_DIR}/${CAMERA_NAME}"
    
    echo "═══════════════════════════════════════"
    echo "  Converting ${CAMERA_NAME} camera"
    echo "═══════════════════════════════════════"
    echo ""
    
    mkdir -p $FRAME_DIR
    
    # Play bag in background
    echo "Playing bag and extracting frames..."
    ros2 bag play $BAG_DIR --topics $TOPIC --rate 2.0 &
    BAG_PID=$!
    
    sleep 2
    
    # Extract compressed images
    python3 /tmp/extract_compressed.py $TOPIC $FRAME_DIR &
    EXTRACT_PID=$!
    
    # Wait for bag to finish
    wait $BAG_PID
    
    sleep 2
    kill $EXTRACT_PID 2>/dev/null
    
    # Count frames
    FRAME_COUNT=$(ls $FRAME_DIR/*.jpg 2>/dev/null | wc -l)
    
    if [ $FRAME_COUNT -eq 0 ]; then
        echo "⚠ Warning: No frames extracted from ${CAMERA_NAME} camera"
        echo "   Check if topic $TOPIC exists in bag"
        return 1
    fi
    
    echo "✓ Extracted $FRAME_COUNT frames"
    echo ""
    echo "Creating MP4 video..."
    
    # Create video
    cd $FRAME_DIR
    ffmpeg -loglevel error -stats \
      -framerate 30 -pattern_type glob -i "frame_*.jpg" \
      -c:v libx264 -pix_fmt yuv420p -preset medium -crf 23 \
      -y $OUTPUT_VIDEO
    
    if [ -f "$OUTPUT_VIDEO" ]; then
        FILE_SIZE=$(du -h $OUTPUT_VIDEO | cut -f1)
        echo ""
        echo "✓ ${CAMERA_NAME^} camera video created:"
        echo "  File: $OUTPUT_VIDEO"
        echo "  Size: $FILE_SIZE"
        echo "  Frames: $FRAME_COUNT"
    else
        echo "✗ Failed to create video"
        return 1
    fi
    
    return 0
}

# Main execution
mkdir -p $TEMP_DIR

case $CAMERA in
    left)
        convert_camera "left"
        ;;
    right)
        convert_camera "right"
        ;;
    both)
        convert_camera "left"
        echo ""
        sleep 2
        convert_camera "right"
        ;;
    *)
        echo "Error: Invalid camera '$CAMERA'. Use 'left', 'right', or 'both'"
        rm -rf $TEMP_DIR
        exit 1
        ;;
esac

# Cleanup
cd ~
rm -rf $TEMP_DIR
rm -f /tmp/extract_compressed.py

echo ""
echo "═══════════════════════════════════════"
echo "✓ Conversion complete!"
echo "═══════════════════════════════════════"
echo ""
ls -lh ${OUTPUT_DIR}/${RECORDING}_*.mp4 2>/dev/null
echo ""