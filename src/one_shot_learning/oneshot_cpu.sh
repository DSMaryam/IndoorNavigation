#!/bin/bash

#SBATCH --job-name=test
#SBATCH --output=%x.o%j.txt
#SBATCH --time=1:00:00 
#SBATCH --ntasks=1 
#SBATCH --partition=cpu_med
#SBATCH --mem=10GB
#SBATCH --mail-user=geoffroy.dunoyer@student.ecp.fr


# Load necessary modules
module purge
module load anaconda3/2020.02/gcc-9.2.0
module load cuda/10.2.89/intel-19.0.3.199

# Activate anaconda environment
source activate cifar10

# Run python script
# python3 test_import.py
python3 oneshotlearning.py $WORKDIR/apprentissage $WORKDIR/temp_data