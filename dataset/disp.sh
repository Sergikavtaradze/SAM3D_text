#!/bin/bash -l

#specify the required resources
#$ -l tmem=20G
#$ -l gpu=1
#$ -l gpu_type=a6000

# Set the job name, output file paths
#$ -N Endonasal_Visualization
#$ -o /cluster/project7/SAMed/samed_codes/3DSAM-adapter/3DSAM-adapter/job_info
#$ -e /cluster/project7/SAMed/samed_codes/3DSAM-adapter/3DSAM-adapter/job_info
#$ -wd /home/mobislam

# Activate the virtual environment
# Initialize Conda 
eval "$(/SAN/medic/CARES/mobarak/venvs/anaconda3/bin/conda shell.bash hook)"
conda activate 3DSAM-adapter

# Add CUDA blocking for better error reporting
export CUDA_LAUNCH_BLOCKING=1

# Exporting CUDA Paths. cuDNN included in cuda paths. 
# Add the CUDA Path 
export PATH=/share/apps/cuda-11.8/bin:/usr/local/cuda-11.8/bin:${PATH}
export LD_LIBRARY_PATH=/share/apps/cuda-11.8/lib64:/usr/local/cuda-11.8/lib:/lib64:${LD_LIBRARY_PATH}
export CUDA_INC_DIR=/share/apps/cuda-11.8/include
export LIBRARY_PATH=/share/apps/cuda-11.8/lib64:/usr/local/cuda-11.8/lib:/lib64:${LIBRARY_PATH}

########################################################
# Finding available GPUs
nvidia-smi

# Get the number of available GPUs
NUM_GPUS=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)

# Get the first available GPU ID
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk '$2 < 100 {print $1}' | head -n 1 | tr -d ',')

# Check if no GPU is available
if [ -z "$CUDA_VISIBLE_DEVICES" ] || [ "$CUDA_VISIBLE_DEVICES" -ge "$NUM_GPUS" ]; then
    echo "No available GPU found. Exiting..." # Throw an error and exit
    exit 1
fi

echo "Number of GPUs: $NUM_GPUS" # Print the number of GPUs
echo "Using GPU: $CUDA_VISIBLE_DEVICES" # Print the GPU ID

########################################################

# Navigate to the directory containing the scripts
cd /cluster/project7/SAMed/samed_codes/3DSAM-adapter/3DSAM-adapter

# Create a temporary Python script to run the visualization
cat > run_visualization.py << EOF
import sys
import os
from dataset.datasets import load_data_volume
from argparse import Namespace

# Parse arguments
output_dir = "${1:-visualizations}"  # Use first argument or default to "visualizations"
start_idx = int("${2:-0}")  # Start index (default: 0)
end_idx = int("${3:-5}")    # End index (default: 5)

# Create args namespace similar to what would be passed during training
args = Namespace(
    data="endonasal",
    data_prefix="/cluster/project7/SAMed/datasets/endonasal_mri_batch2",
    batch_size=1,
    rand_crop_size=(128, 128, 128),  # Use default or adjust as needed
    num_worker=1
)

# Load the dataset using the same function as in training
train_loader = load_data_volume(
    data=args.data,
    path_prefix=args.data_prefix,
    batch_size=1,
    augmentation=False,  # Set to False for visualization
    split="train",
    rand_crop_spatial_size=args.rand_crop_size,
    num_worker=args.num_worker
)

# Extract the dataset from the loader
dataset = train_loader.dataset

# Create visualizations for each index
for idx in range(start_idx, end_idx):
    if idx < len(dataset):
        print(f"Generating visualization for index {idx}...")
        from dataset.disp import visualize_masks
        visualize_masks(dataset, idx, output_dir=output_dir)
    else:
        print(f"Index {idx} out of range. Dataset has {len(dataset)} items.")

print(f"All visualizations saved to {output_dir}")
EOF

# Run the visualization script
# Arguments: output_dir start_idx end_idx
# Example: ./disp.sh visualizations/test 0 10
python3 run_visualization.py ${1:-visualizations} ${2:-0} ${3:-5}

# Clean up the temporary script
rm run_visualization.py

echo "Visualization job completed."
