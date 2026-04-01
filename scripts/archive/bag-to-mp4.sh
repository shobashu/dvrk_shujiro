#!/usr/bin/env bash

if [ -z "$1" ]; then
    echo "Usage: ./bag-to-mp4.sh <recording_name> [left|right|both]"
    echo "Example: ./bag-to-mp4.sh test1_shu both"
    echo ""
    echo "Options:"
    echo "  left  - Convert only left camera (default)"
    echo "  right - Convert only right camera"
    echo "  both  - Convert both cameras (sequential)"
    exit 1
fi

RECORDING=$1
MODE=${2:-left}

BAG_DIR=~/dvrk_recordings/$RECORDING
TEMP_DIR=/tmp/bag_images_$$

if [ ! -d "$BAG_DIR" ]; then
    echo "Error: Recording not found: $BAG_DIR"
    exit 1
fi

source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash

convert_camera() {
    local CAMERA=$1
    local TOPIC="/camera/${CAMERA}/image_raw"
    local OUTPUT_VIDEO=~/dvrk_recordings/${RECORDING}_${CAMERA}.mp4
    local FRAME_DIR="${TEMP_DIR}/${CAMERA}"
    
    echo ""
    echo "═══════════════════════════════════════"
    echo "  Converting ${CAMERA} camera"
    echo "═══════════════════════════════════════"
    echo ""
    
    mkdir -p $FRAME_DIR
    cd $FRAME_DIR
    
    echo "Step 1: Extracting ${CAMERA} camera frames..."
    echo "Topic: $TOPIC"
    
    # Play ONLY this camera's topic (critical fix!)
    ros2 bag play $BAG_DIR \
        --topics $TOPIC \
        --rate 2.0 &
    BAG_PID=$!
    
    sleep 3
    
    # Save images
    ros2 run image_view image_saver --ros-args \
        -r image:=$TOPIC \
        -p filename_format:="${CAMERA}_%06d.jpg" &
    SAVER_PID=$!
    
    # Wait for playback to finish
    wait $BAG_PID
    
    sleep 2
    kill $SAVER_PID 2>/dev/null
    
    FRAME_COUNT=$(ls ${CAMERA}_*.jpg 2>/dev/null | wc -l)
    
    if [ $FRAME_COUNT -eq 0 ]; then
        echo "⚠ Warning: No frames extracted from ${CAMERA} camera!"
        echo "   Topic $TOPIC may not exist in the bag"
        echo ""
        echo "   Available topics in bag:"
        ros2 bag info $BAG_DIR | grep "/camera"
        return 1
    fi
    
    echo "✓ Extracted $FRAME_COUNT frames"
    echo ""
    echo "Step 2: Creating MP4 video..."
    
    # Create video
    ffmpeg -framerate 30 -pattern_type glob -i "${CAMERA}_*.jpg" \
        -c:v libx264 -pix_fmt yuv420p -preset medium -crf 23 \
        -y $OUTPUT_VIDEO 2>&1 | grep -E "frame=|Duration|video:"
    
    if [ -f "$OUTPUT_VIDEO" ]; then
        FILE_SIZE=$(du -h $OUTPUT_VIDEO | cut -f1)
        echo ""
        echo "✓ ${CAMERA^} camera video created:"
        echo "  File: $OUTPUT_VIDEO"
        echo "  Size: $FILE_SIZE"
        echo "  Frames: $FRAME_COUNT"
    else
        echo "✗ Failed to create video for ${CAMERA} camera"
        return 1
    fi
    
    return 0
}

# Main execution
mkdir -p $TEMP_DIR

case $MODE in
    left)
        convert_camera "left"
        ;;
    right)
        convert_camera "right"
        ;;
    both)
        echo "Converting both cameras (sequential processing)..."
        convert_camera "left"
        
        # Wait between conversions to avoid conflicts
        echo ""
        echo "Waiting 3 seconds before processing right camera..."
        sleep 3
        
        convert_camera "right"
        ;;
    *)
        echo "Error: Invalid mode '$MODE'. Use 'left', 'right', or 'both'"
        rm -rf $TEMP_DIR
        exit 1
        ;;
esac

# Cleanup
cd ~
rm -rf $TEMP_DIR

echo ""
echo "═══════════════════════════════════════"
echo "✓ Conversion complete!"
echo "═══════════════════════════════════════"
echo ""
echo "Videos saved to: ~/dvrk_recordings/"
ls -lh ~/dvrk_recordings/${RECORDING}_*.mp4 2>/dev/null
echo ""