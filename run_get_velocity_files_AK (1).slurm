#!/bin/bash
#SBATCH --account=dbergle1
#SBATCH --job-name=getVeloFiles
#SBATCH --time=2:00:00
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --mem=20gb
#SBATCH --mail-type=end
#SBATCH --mail-user=akim129@jh.edu

##---------------------------------------------------------------------
## SLURM job script to run kallisto index building
##---------------------------------------------------------------------

##Set up environment
ml singularity/3.8.7

cd /home/akim129/scratch/

#This is an example of how to call a scripts within the singularity container.
#For a bash script, use "bash" instead of "Rscript"
singularity exec scAnalysis.sif Rscript scripts/get_velocity_files_AK.R

echo "we got here"


