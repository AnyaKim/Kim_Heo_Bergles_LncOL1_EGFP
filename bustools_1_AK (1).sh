#!/bin/bash
#SBATCH --account=dbergle1
#SBATCH --job-name=bustools
#SBATCH --time=11:00:00
#SBATCH --partition=parallel
#SBATCH --nodes=1
#SBATCH --mem=90gb
#SBATCH --ntasks=19
#SBATCH --mail-type=end
#SBATCH --mail-user=akim129@jh.edu

#Note- the original code pointed to a whitelist that wasn't included in the documents I had, so I found the whitelist for the 10x v 3.1 kit and uploaded it. 

cd /home/akim129/data_dbergle1/lncol1/Anya_GEO/kallisto_out/AK1_4
mkdir cDNA_capture/ introns_capture/ spliced/ unspliced/ 

echo Quantifying transcripts for AK1_4 with bustools

# -s flag specifies that the capture list is transcript list.
#Change directories and whitelist as needed
#File names should stay be the same
bustools correct -w /home/akim129/scratch/kallistobus/required_files/10xv3_whitelist.txt -p output.bus | bustools sort -o output.correct.sort.bus -t 4 -
bustools capture -s -o cDNA_capture/spliced.bus -c /home/akim129/scratch/kallistobus/required_files/cDNA_tx_to_capture.txt -e matrix.ec -t transcripts.txt output.correct.sort.bus
bustools capture -s -o introns_capture/unspliced.bus -c /home/akim129/scratch/kallistobus/required_files/introns_tx_to_capture.txt -e matrix.ec -t transcripts.txt output.correct.sort.bus
bustools count -o unspliced/u -g /home/akim129/scratch/kallistobus/required_files/tr2g.tsv -e matrix.ec -t transcripts.txt --genecounts introns_capture/unspliced.bus
bustools count -o spliced/s -g /home/akim129/scratch/kallistobus/required_files/tr2g.tsv -e matrix.ec -t transcripts.txt --genecounts cDNA_capture/spliced.bus

cd /home/akim129/data_dbergle1/lncol1/Anya_GEO/kallisto_out/AK2
mkdir cDNA_capture/ introns_capture/ spliced/ unspliced/

echo Quantifying transcripts for AK2 with bustools

bustools correct -w /home/akim129/scratch/kallistobus/required_files/10xv3_whitelist.txt -p output.bus | bustools sort -o output.correct.sort.bus -t 4 -
bustools capture -s -o cDNA_capture/spliced.bus -c /home/akim129/scratch/kallistobus/required_files/cDNA_tx_to_capture.txt -e matrix.ec -t transcripts.txt output.correct.sort.bus
bustools capture -s -o introns_capture/unspliced.bus -c /home/akim129/scratch/kallistobus/required_files/introns_tx_to_capture.txt -e matrix.ec -t transcripts.txt output.correct.sort.bus
bustools count -o unspliced/u -g /home/akim129/scratch/kallistobus/required_files/tr2g.tsv -e matrix.ec -t transcripts.txt --genecounts introns_capture/unspliced.bus
bustools count -o spliced/s -g /home/akim129/scratch/kallistobus/required_files/tr2g.tsv -e matrix.ec -t transcripts.txt --genecounts cDNA_capture/spliced.bus

cd /home/akim129/data_dbergle1/lncol1/Anya_GEO/kallisto_out/AK3
mkdir cDNA_capture/ introns_capture/ spliced/ unspliced/

echo Quantifying transcripts for AK3 with bustools

bustools correct -w /home/akim129/scratch/kallistobus/required_files/10xv3_whitelist.txt -p output.bus | bustools sort -o output.correct.sort.bus -t 4 -
bustools capture -s -o cDNA_capture/spliced.bus -c /home/akim129/scratch/kallistobus/required_files/cDNA_tx_to_capture.txt -e matrix.ec -t transcripts.txt output.correct.sort.bus
bustools capture -s -o introns_capture/unspliced.bus -c /home/akim129/scratch/kallistobus/required_files/introns_tx_to_capture.txt -e matrix.ec -t transcripts.txt output.correct.sort.bus
bustools count -o unspliced/u -g /home/akim129/scratch/kallistobus/required_files/tr2g.tsv -e matrix.ec -t transcripts.txt --genecounts introns_capture/unspliced.bus
bustools count -o spliced/s -g /home/akim129/scratch/kallistobus/required_files/tr2g.tsv -e matrix.ec -t transcripts.txt --genecounts cDNA_capture/spliced.bus

echo "Done"