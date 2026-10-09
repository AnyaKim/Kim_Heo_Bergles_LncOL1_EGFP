#!/bin/bash
#SBATCH --account=dbergle1
#SBATCH --job-name=build_index
#SBATCH --time=2:00:00
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --mem=90gb
#SBATCH --mail-type=end
#SBATCH --mail-user=akim129@jh.edu

HOME=/home/akim129/scratch/

cd $HOME

ml anaconda
ml singularity/3.8.7
singularity exec scAnalysis.sif bash scripts/build_index_1_AK.sh
