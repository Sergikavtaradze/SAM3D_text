#!/bin/bash -l

#specify the required resources
#$ -l tmem=48G
#$ -l gpu=1
#$ -l gpu_type=a6000|a100|a100_80

# Set the job name, output file paths
#$ -N Train_192crop_lr4e-4_Mobarak_Apr15
#$ -o /cluster/project7/SAMed/samed_codes/3DSAM-adapter/3DSAM-adapter/job_info/Apr15/multiclass_hyperparameter_tuning
#$ -e /cluster/project7/SAMed/samed_codes/3DSAM-adapter/3DSAM-adapter/job_info/Apr15/multiclass_hyperparameter_tuning
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

# Add PyTorch CUDA libraries to path
export LD_LIBRARY_PATH=/SAN/medic/CARES/mobarak/venvs/anaconda3/envs/3DSAM-adapter/lib/python3.9/site-packages/torch/lib:${LD_LIBRARY_PATH}

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

python3 train_mobarak.py --data maskwise_endonasal \
--snapshot_path '/cluster/project7/SAMed/samed_codes/3DSAM-adapter/3DSAM-adapter/ckpt' \
--data_prefix '/cluster/project7/SAMed/datasets/endonasal_mri_batch2' \
--date '_Apr15' \
--rand_crop_size 192
