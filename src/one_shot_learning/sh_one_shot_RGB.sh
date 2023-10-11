#!/bin/bash

#SBATCH --job-name=one_shot_RGB
#SBATCH --output=%x.o%j.txt
#SBATCH --time=01:00:00 
#SBATCH --ntasks=4
#SBATCH --nodes=1
#SBATCH --mem=100GB
#SBATCH --mail-user=geoffroy.dunoyer@student.ecp.fr
#SBATCH --mail-type=END
# SBATCH --gres=gpu:4
#SBATCH --partition=cpu_short


# Load necessary modules
module purge
module load anaconda3/2020.02/gcc-9.2.0
module load cuda/10.2.89/intel-19.0.3.199

# Activate anaconda environment
source activate cifar10

# Run python script
# python3 test_import.py
# python3 oneshotlearning_RGB.py $WORKDIR/photos_apprentissage_visage_rgb $WORKDIR/output_faces_rgb.csv $WORKDIR/output_print/faces_${SLURM_JOBID}.csv
python3  -m torch.distributed.launch --nproc_per_node=1 --master_port 22222 oneshotlearning_RGB_DDP.py $WORKDIR/photos_apprentissage_visage_rgb $WORKDIR/output_faces_rgb.csv $WORKDIR/output_print/faces_${SLURM_JOBID}.csv
# python3 oneshotlearning_RGB.py $WORKDIR/temp_data ./output.csv $WORKDIR/output_print/indoor_${SLURM_JOBID}.csv