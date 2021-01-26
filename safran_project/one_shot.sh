#!/bin/bash

#SBATCH --job-name=one_shot
#SBATCH --output=%x.o%j.txt
#SBATCH --time=3:00:00 
#SBATCH --ntasks=1 
#SBATCH --partition=cpu_med
#SBATCH --mem=100GB
#SBATCH --mail-user=geoffroy.dunoyer@student.ecp.fr

# Load necessary modules
module purge
module load anaconda3/2020.02/gcc-9.2.0

# Activate anaconda environment
source activate numpy-env

# Run python script
python3 one_shot_learning.py $WORKDIR/one_shot_data/cave/ $WORKDIR/one_shot_data/data/