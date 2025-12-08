#!/bin/bash
#SBATCH --job-name=continuous_gridsearch
#SBATCH --output={{LOG_DIR}}/continuous_gridsearch_%A_%a.out
#SBATCH --error={{LOG_DIR}}/continuous_gridsearch_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --array=1-{{NUM_JOBS}}%8

# Load necessary modules (adjust based on your cluster)
# module load python/3.10
# module load cuda/11.8

# Activate virtual environment
source /mnt/polished-lake/home/annabelma/other/.venv/bin/activate

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

# Get job parameters from job list file
JOB_LIST_FILE="{{JOB_LIST_FILE}}"
OUTPUT_DIR="{{OUTPUT_DIR}}"

# Read the line corresponding to this array task
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$JOB_LIST_FILE")

# Parse the line (format: lr prompt_length num_epochs [--adversarial])
lr=$(echo "$LINE" | awk '{print $1}')
prompt_length=$(echo "$LINE" | awk '{print $2}')
num_epochs=$(echo "$LINE" | awk '{print $3}')
adversarial_flag=$(echo "$LINE" | awk '{print $4}')

# Run the training script
cd /mnt/polished-lake/home/annabelma/other
python train_continuous_soft_prompt.py \
    --lr "$lr" \
    --prompt_length "$prompt_length" \
    --num_epochs "$num_epochs" \
    --output_dir "$OUTPUT_DIR" \
    $adversarial_flag \
    --batch_size 16 \
    --model_name t5-large

echo "Job ${SLURM_ARRAY_TASK_ID} completed: lr=$lr, prompt_length=$prompt_length, epochs=$num_epochs, adversarial=$adversarial_flag"

