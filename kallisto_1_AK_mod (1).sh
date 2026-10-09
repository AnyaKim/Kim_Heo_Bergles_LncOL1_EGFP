#!/bin/bash
#SBATCH --account=dbergle1
#SBATCH --job-name=kallisto
#SBATCH --time=14:00:00
#SBATCH --partition=parallel
#SBATCH --nodes=1
#SBATCH --mem=90gb
#SBATCH --ntasks=19
#SBATCH --mail-type=end
#SBATCH --mail-user=akim129@jh.edu

#HOME=/home/akim129/data_dbergle1/lncol1/Anya_GEO

#cd $HOME

#mkdir kallisto_out
#cd kallisto_out
#mkdir AK1_4/ AK2/ AK3

#cd AK1_4

#echo working on AK1_4 pseudoalignment

#kallisto bus -i /home/akim129/scratch/kallistobus/required_files/mm_ens101_velocity_index_10xv3.idx -o /home/akim129/data_dbergle1/lncol1/Anya_GEO/kallisto_out/AK1_4 -x 10xv3 -t 4 \
#	/home/akim129/data_dbergle1/lncol1/Anya_GEO/AK-1-4-lib_S1_L001_R1_001.fastq.gz \
#	/home/akim129/data_dbergle1/lncol1/Anya_GEO/AK-1-4-lib_S1_L001_R2_001.fastq.gz \

#echo AK1_4 pseudoalignment complete, working on AK2

#cd /home/akim129/data_dbergle1/lncol1/Anya_GEO/kallisto_out/AK2

#kallisto bus -i /home/akim129/scratch/kallistobus/required_files/mm_ens101_velocity_index_10xv3.idx -o /home/akim129/data_dbergle1/lncol1/Anya_GEO/kallisto_out/AK2 -x 10xv3 -t 4 \
#	/home/akim129/data_dbergle1/lncol1/Anya_GEO/AK-2-lib_S2_L001_R1_001.fastq.gz \
#	/home/akim129/data_dbergle1/lncol1/Anya_GEO/AK-2-lib_S2_L001_R2_001.fastq.gz \

#echo AK2 pseudoalignment complete, working on AK3

cd /home/akim129/data_dbergle1/lncol1/Anya_GEO/kallisto_out/AK3

kallisto bus -i /home/akim129/scratch/kallistobus/required_files/mm_ens101_velocity_index_10xv3.idx -o /home/akim129/data_dbergle1/lncol1/Anya_GEO/kallisto_out/AK3 -x 10xv3 -t 4 \
	/home/akim129/data_dbergle1/lncol1/Anya_GEO/AK-3-lib_S3_L001_R1_001.fastq.gz \
	/home/akim129/data_dbergle1/lncol1/Anya_GEO/AK-3-lib_S3_L001_R2_001.fastq.gz \

echo finished
#
