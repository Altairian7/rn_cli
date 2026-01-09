#!/bin/bash
# Dog Nose Recognition - IP Camera Stream Test
# Test the nose recognition system using remote IP camera

echo "🚀 Dog Nose Recognition - IP Camera Stream Test"
echo "=================================================="
echo ""
echo "Testing camera at: http://10.83.99.99:8080/video"
echo ""
echo "Activating venv..."
source /home/sean/Documents/codeBASE/useless_pro/rn_cli/backend/venv/bin/activate

cd /home/sean/Documents/codeBASE/useless_pro/rn_cli/backend

echo "✅ Running recognition test..."
echo ""
echo "Options:"
echo "  --source URL         : Camera stream URL"
echo "  --threshold VALUE    : Similarity threshold (default: 0.25)"
echo "  --capture-interval MS: Capture interval (default: 500ms)"
echo "  --no-display        : Disable display window"
echo ""
echo "Running with threshold=0.25 and no-display for better performance..."
echo ""

python test_camera_nose_recognition.py \
    --source http://10.83.99.99:8080/video \
    --threshold 0.25 \
    --no-display
