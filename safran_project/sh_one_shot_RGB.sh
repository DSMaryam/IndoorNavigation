#!/bin/bash

#SBATCH --job-name=one_shot_RGB
#SBATCH --output=%x.o%j.txt
#SBATCH --time=1:00:00 
#SBATCH --ntasks=4
#SBATCH --nodes=1
#SBATCH --mem=700GB
#SBATCH --mail-user=geoffroy.dunoyer@student.ecp.fr
#SBATCH --mail-type=END
#SBATCH --gres=gpu:4
#SBATCH --partition=gpu_test


# Load necessary modules
module purge
module load anaconda3/2020.02/gcc-9.2.0
module load cuda/10.2.89/intel-19.0.3.199

# Activate anaconda environment
source activate cifar10

# Run python script
# python3 test_import.py
# python3 oneshotlearning_RGB.py $WORKDIR/photos_apprentissage_visage $WORKDIR/output_faces.csv
python3 oneshotlearning_RGB.py $WORKDIR/temp_data ./output.csv $WORKDIR/temp_data