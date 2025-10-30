#!/bin/bash -l

#specify the required resources
#$ -l tmem=80G
#$ -l gpu=1
#$ -l gpu_type=a100_80

# Set the job name, output file paths
#$ -N Train_MaskWise_192crop_lr4e-4_Oct30
#$ -o /cluster/project7/SAMed/samed_codes/3DSAM-adapter/3DSAM-adapter/job_info/Oct/
#$ -e /cluster/project7/SAMed/samed_codes/3DSAM-adapter/3DSAM-adapter/job_info/Oct/
#$ -wd /home/mobislam

#set -euo pipefail

# Keep Python isolated to the conda env
export PYTHONNOUSERSITE=1

# Disable hugging face parallelism
export TOKENIZERS_PARALLELISM=false
#unset PYTHONPATH

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
export LD_LIBRARY_PATH=/SAN/medic/CARES/mobarak/venvs/anaconda3/envs/3DSAM-adapter/lib/python3.9/site-packages/torch/lib:${LD_LIBRARY_PATH}

########################################################

# 4) Optional debug to confirm imports come from the env, not ~/.local
python -s - <<'PY'
import sys, site, importlib_metadata, setuptools, torch, torchvision
print("[debug] exe:", sys.executable)
print("[debug] user-site:", site.getusersitepackages())
print("[debug] importlib_metadata:", importlib_metadata.__file__)
print("[debug] setuptools:", setuptools.__version__)
print("[debug] torch:", torch.__version__, "build cuda:", torch.version.cuda, "cuda_available:", torch.cuda.is_available())
print("[debug] torchvision:", torchvision.__version__)
PY

########################################################
# Finding available GPUs
nvidia-smi

# Get the number of available GPUs
NUM_GPUS=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)

# Get the first available GPU ID
export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk '$2 < 100 {print $1}' | head -n 1 | tr -d ',')

# Check if no GPU is available
if [ -z "$CUDA_VISIBLE_DEVICES" ] || [ "$CUDA_VISIBLE_DEVICES" -ge "$NUM_GPUS" ]; then
    echo "No available GPU found. Exiting..." 
    exit 1
fi

echo "Number of GPUs: $NUM_GPUS" 
echo "Using GPU: $CUDA_VISIBLE_DEVICES"

########################################################


# Navigate to the directory containing the scripts
cd /cluster/project7/SAMed/samed_codes/3DSAM-adapter/3DSAM-adapter


# The ckpt folder should be the best from the 128, 192, 256 crop training,
# Because the image encoder will need to be good at the specific patch size
python3 train_mobarak_LLAMA.py --data maskwise_endonasal \
--snapshot_path '/cluster/project7/SAMed/samed_codes/3DSAM-adapter/3DSAM-adapter/ckpt' \
--data_prefix '/cluster/project7/SAMed/datasets/endonasal_mri_batch2' \
--max_epoch 200 \
--date '_Oct30' \
--lr 0.0004 \
--rand_crop_size 192 192 192 \