#!/bin/bash -l

#specify the required resources
#$ -l tmem=20G
#$ -l gpu=1

# Set the job name, output file paths
#$ -N Test_tumor_192crop_seed24_lr0.0004_GPT2_ckptApr19_Aug15
#$ -o /cluster/project7/SAMed/samed_codes/3DSAM-adapter/3DSAM-adapter/job_info/Aug/Test_192_lr0.0004
#$ -e /cluster/project7/SAMed/samed_codes/3DSAM-adapter/3DSAM-adapter/job_info/Aug/Test_192_lr0.0004

# Activate the virtual environment
# Initialize Conda
# source /share/apps/source_files/python/python-3.9.16.source
# Activate the specific environment 
eval "$(/SAN/medic/CARES/mobarak/venvs/anaconda3/bin/conda shell.bash hook)"
conda activate 3DSAM-adapter

##################################
##### CUDA Environment Setup #####
##################################

# Add CUDA binary directories to PATH - enables system to find and execute CUDA tools (nvidia-smi, nvcc, etc.)
export PATH=/share/apps/cuda-11.8/bin:/usr/local/cuda-11.8/bin:${PATH}

# Set runtime library path - tells system where to find CUDA shared libraries during program execution
# This includes both shared (/share/apps) and local (/usr/local) CUDA installations
export LD_LIBRARY_PATH=/share/apps/cuda-11.8/lib64:/usr/local/cuda-11.8/lib:/lib64:${LD_LIBRARY_PATH}

# Set CUDA include directory - specifies location of CUDA header files
# Used during compilation of CUDA programs (if needed)
export CUDA_INC_DIR=/share/apps/cuda-11.8/include

# Set compile-time library path - tells compiler where to find libraries during linking
# Similar to LD_LIBRARY_PATH but used at build time instead of runtime
export LIBRARY_PATH=/share/apps/cuda-11.8/lib64:/usr/local/cuda-11.8/lib:/lib64:${LIBRARY_PATH}

##################################
##### Finding available GPUs #####
##################################
nvidia-smi

# Get the first available GPU ID
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk '$2 < 100 {print $1}' | head -n 1| tr -d ',')

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    echo "No available GPU found. Exiting..."
    exit 1
fi

echo "Using GPU: $CUDA_VISIBLE_DEVICES"

# Add CUDA blocking for better error reporting
export CUDA_LAUNCH_BLOCKING=1

########################################################

# Navigate to the directory containing the scripts
cd /cluster/project7/SAMed/samed_codes/3DSAM-adapter/3DSAM-adapter

# MaskWise_Excluded_IDs_lr_0.004crop(128, 128, 128)_Apr17, epoch 149, Y
# MaskWise_Excluded_IDs_lr_0.004crop(192, 192, 192)_Apr17, epoch 147, Y
# MaskWise_Excluded_IDs_lr_0.004crop(256, 256, 256)_Apr17, epoch 144, N

# MaskWise_Excluded_IDs_lr_0.0004crop(128, 128, 128)ie_lr*0.1_Apr14, epoch 130, N
# MaskWise_Excluded_IDs_lr_0.0004crop(192, 192, 192)_Apr19, epoch 110, N
# MaskWise_Excluded_IDs_lr_0.0004crop(256, 256, 256)_Apr19, epoch 139, N

# Use other shell script for LoRa Model
# MaskWise_Excluded_IDs_lr_0.0004crop(128, 128, 128)_Apr22, epoch 16


python3 test_GPT_maskwise.py --data maskwise_endonasal --checkpoint best \
--snapshot_path "/cluster/project7/SAMed/samed_codes/3DSAM-adapter/3DSAM-adapter/ckpt/maskwise_endonasal/maskwise_endonasal/MaskWise_Excluded_IDs_lr_0.0004crop(192, 192, 192)_Apr19" \
--data_prefix "/cluster/project7/SAMed/datasets/endonasal_mri_batch2" \
--target_class ICA \
--rand_crop_size 192 192 192 \
--seed '24_192_lr0.0004'
