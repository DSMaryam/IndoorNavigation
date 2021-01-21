#!/bin/bash

#SBATCH --job-name=preprocessing
#SBATCH --output=%x.o%j 
#SBATCH --time=12:00:00 
#SBATCH --ntasks=1 
#SBATCH --partition=cpu_long
#SBATCH --mem=10GB

# Load necessary modules
module purge
module load anaconda3/2020.02/gcc-9.2.0

# Activate anaconda environment
source activate numpy-env

# Run python script
# python3 preprocessing_final.py $WORKDIR/Photos/Photos/Photos\ de\ la\ cave $WORKDIR 1000
# python3 score_matrix_generation.py $WORKDIR/ $WORKDIR/Photos/Photos/Photos\ de\ la\ cave $WORKDIR/results 1000 2
# python3 score_matrix_generation.py $WORKDIR/ $WORKDIR/Photos/Photos/Photos\ de\ la\ cave $WORKDIR/results 1000 3
# python3 score_matrix_generation.py $WORKDIR/ $WORKDIR/Photos/Photos/Photos\ de\ la\ cave $WORKDIR/results 1000 4
# python3 score_matrix_generation.py $WORKDIR/ $WORKDIR/Photos/Photos/Photos\ de\ la\ cave $WORKDIR/results 1000 5
# python3 preprocessing_final.py $WORKDIR/Photos/Photos/Photos\ de\ la\ cave $WORKDIR 1500
# python3 score_matrix_generation.py $WORKDIR/ $WORKDIR/Photos/Photos/Photos\ de\ la\ cave $WORKDIR/results 1500 2
# python3 score_matrix_generation.py $WORKDIR/ $WORKDIR/Photos/Photos/Photos\ de\ la\ cave $WORKDIR/results 1500 3
# python3 score_matrix_generation.py $WORKDIR/ $WORKDIR/Photos/Photos/Photos\ de\ la\ cave $WORKDIR/results 1500 4
# python3 score_matrix_generation.py $WORKDIR/ $WORKDIR/Photos/Photos/Photos\ de\ la\ cave $WORKDIR/results 1500 5
# python3 preprocessing_final.py $WORKDIR/Photos/Photos/Photos\ de\ la\ cave $WORKDIR 3000
# python3 score_matrix_generation.py $WORKDIR/ $WORKDIR/Photos/Photos/Photos\ de\ la\ cave $WORKDIR/results 3000
# python3 preprocessing_final.py $WORKDIR/Photos/Photos/Photos\ de\ la\ cave $WORKDIR 5000
# python3 score_matrix_generation.py $WORKDIR/ $WORKDIR/Photos/Photos/Photos\ de\ la\ cave $WORKDIR/results 5000
# python3 preprocessing_final.py $WORKDIR/Photos/Photos/Photos\ de\ la\ cave $WORKDIR 10000
# python3 score_matrix_generation.py $WORKDIR/ $WORKDIR/Photos/Photos/Photos\ de\ la\ cave $WORKDIR/results 10000
# python3 preprocessing_final.py $WORKDIR/Photos/Photos/Photos\ de\ la\ cave $WORKDIR 500
# python3 score_matrix_generation.py $WORKDIR/ $WORKDIR/Photos/Photos/Photos\ de\ la\ cave $WORKDIR/results 500
python3 score_matrix_generation.py $WORKDIR/ $WORKDIR/Photos/Photos/Photos\ de\ la\ cave $WORKDIR/results 500 2
python3 score_matrix_generation.py $WORKDIR/ $WORKDIR/Photos/Photos/Photos\ de\ la\ cave $WORKDIR/results 500 3
python3 score_matrix_generation.py $WORKDIR/ $WORKDIR/Photos/Photos/Photos\ de\ la\ cave $WORKDIR/results 500 4
python3 score_matrix_generation.py $WORKDIR/ $WORKDIR/Photos/Photos/Photos\ de\ la\ cave $WORKDIR/results 500 5