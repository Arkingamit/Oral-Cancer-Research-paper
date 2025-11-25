#!/bin/bash
#SBATCH --job-name=oral_cls
#SBATCH --output=logs/oral_%j.log
#SBATCH --error=logs/error_%j.err



echo "Starting job on $(hostname) at $(date)"
echo "Current directory: $(pwd)"
echo "Listing files:"
ls -l

cd $SLURM_SUBMIT_DIR

# Load any modules required
module load anaconda3

# Activate the virtual environment correctly
source /home/21bce072/oral2/env/bin/activate

echo "Python executable: $(which python)"
echo "Python version: $(python --version)"

echo "Running script..."
srun python redish_legion.py

echo "Job finished at $(date)"
