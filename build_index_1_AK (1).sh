#!/bin/bash
#SBATCH --account=dbergle1
#SBATCH --job-name=build_index
#SBATCH --time=4:00:00
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --mem=90gb
#SBATCH --mail-type=end
#SBATCH --mail-user=akim129@jh.edu

HOME=/home/akim129/scratch/kallistobus/required_files


cd $HOME

echo "Building index..."
kallisto index -i mm_ens101_velocity_index_10xv3.idx -k 31 cDNA_introns.fa
