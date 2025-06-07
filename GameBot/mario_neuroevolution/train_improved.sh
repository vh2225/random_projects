#!/bin/bash
# Improved training script with better fitness function

echo "Starting improved Mario Neuroevolution training..."
echo "New fitness function penalizes jumping in place and rewards forward progress"
echo ""

# Activate virtual environment
source venv/bin/activate

# Clear previous runs
echo "Creating fresh output directory..."
timestamp=$(date +%Y%m%d_%H%M%S)
output_dir="outputs/run_improved_${timestamp}"
mkdir -p $output_dir

# Run training with improved fitness
echo "Training with improved fitness function..."
python main.py \
    --mode train \
    --generations 300 \
    --checkpoint-freq 10 \
    --level 1-1

echo ""
echo "Training complete! Check the outputs/ directory for results."
echo "The new fitness function should produce agents that move forward instead of jumping in place."