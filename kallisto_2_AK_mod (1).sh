#!/bin/bash
#SBATCH --account=dbergle1
#SBATCH --job-name=kallisto
#SBATCH --time=8:00:00
#SBATCH --partition=parallel
#SBATCH --nodes=1
#SBATCH --mem=90gb
#SBATCH --ntasks=19
#SBATCH --mail-type=end
#SBATCH --mail-user=akim129@jh.edu

HOME=/home/akim129/scratch

cd $HOME

ml anaconda
ml singularity/3.8.7

singularity exec scAnalysis.sif bash scripts/kallisto_1_AK_mod.sh
