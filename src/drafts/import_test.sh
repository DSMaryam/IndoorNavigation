#!/bin/bash

#SBATCH --job-name=preprocessing
# SBATCH --output=%x.o%j 
#SBATCH --time=01:00:00 
#SBATCH --ntasks=1 
#SBATCH --partition=cpu_short
#SBATCH --mem=10GB

# Load necessary modules
module purge
module load anaconda3/2020.02/gcc-9.2.0

# Activate anaconda environment
source activate numpy-env

# Run python script
python3 import_test.py 
