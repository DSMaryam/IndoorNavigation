#!/bin/bash

#SBATCH --job-name=test
#SBATCH --output=%x.o%j.txt
#SBATCH --time=5:00:00 
#SBATCH --ntasks=1 
# SBATCH --partition=cpu_med
#SBATCH --mem=10GB
#SBATCH --mail-user=geoffroy.dunoyer@student.ecp.fr
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu


# Load necessary modules
module purge
module load anaconda3/2020.02/gcc-9.2.0
module load cuda/10.2.89/intel-19.0.3.199

# Activate anaconda environment
source activate cifar10

# Run python script
# python3 test_import.py
python3 oneshotlearning.py $WORKDIR/photos_apprentissage_visage $WORKDIR/output_faces.csv $WORKDIR/photos_apprentissage_visage
# python3 oneshotlearning.py $WORKDIR/apprentissage $WORKDIR/output.csv $WORKDIR/temp_data
# curl --ntlm --user geoffroy.dunoyer@student.ecp.fr:G$3af0c\!ECP --upload-file output.csv https://centralesupelec.sharepoint.com/sites/Projetinfonum12/output.csv
# curl --ntlm --user geoffroy.dunoyer@student.ecp.fr:G$3af0c\!ECP --upload-file output.csv https://teams.microsoft.com/_#/school/files/G%C3%A9n%C3%A9ral?threadId=19%3A30cd1d3075f048c3a94a94dc90899d41%40thread.tacv2&ctx=channel&context=General&rootfolder=%252Fsites%252FProjetinfonum12%252FDocuments%2520partages%252FGeneral/output.csv