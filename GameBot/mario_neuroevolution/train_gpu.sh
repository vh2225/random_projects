#!/bin/bash
# GPU-accelerated training script for Mario Neuroevolution

echo "Starting GPU-accelerated Mario Neuroevolution training..."
echo "RTX 4070 GPU will be utilized for preprocessing and inference"
echo ""

# Activate virtual environment
source venv/bin/activate

# Check GPU availability
python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None"

# Set display for headless mode
export DISPLAY=:0

# Run training with original main.py (more stable)
python main.py \
    --mode train \
    --generations 200 \
    --checkpoint-freq 10 \
    --level 1-1

echo ""
echo "Training complete! Check the outputs/ directory for results."