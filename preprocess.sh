#!/bin/bash 
# setup environment 

#SBATCH -n 1
#SBATCH --mem-per-cpu=12G
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=preprocess.out

conda activate neuronet2

DATASET_NAME="dodh_processed_wavelet_full" 
TECHNIQUE="wavelet_transform"

#/cluster/home/choij/miniconda3/envs/neuronet/bin/python ./preprocess/sleepEDF_cassette_process.py --multiprocess 1 --reduced --root_folder "/cluster/project/jbuhmann/choij/NeuroNet/dataset/physionet.org/files/sleep-edfx/1.0.0/sleep-cassette" --dest_folder ./dataset
/cluster/home/choij/miniconda3/envs/neuronet/bin/python ./preprocess/dodh_spectro.py --multiprocess 1 --reduced --root_folder "/cluster/project/jbuhmann/choij/dreem-learning-open/data/" --dest_folder ./dataset
#/cluster/home/choij/miniconda3/envs/neuronet/bin/python ./preprocess/dodh_spectro.py --multiprocess 1 --reduced --root_folder "/cluster/project/jbuhmann/choij/NeuroNet/dataset/physionet.org/files/sleep-edfx/1.0.0/sleep-cassette" --dest_folder ./dataset --dataset_name $DATASET_NAME --time_freq_technique $TECHNIQUE

cd /cluster/project/jbuhmann/choij/sleep-stage-classification/dataset/
zip -r dodh_processed_wavelet_full.zip dodh_processed_wavelet_full/