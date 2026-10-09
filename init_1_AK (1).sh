#Script to create conda environment from .yaml configuration file

HOME=/home/akim129/scratch/

cd $HOME

ml anaconda
conda env create --file kallistobus/configs/kallistobus_AK.yaml

conda env list





