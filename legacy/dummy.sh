#!/bin/bash 
# setup environment 

#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=02:00:00
#SBATCH --output=plot.out

conda activate neuronet



# cd /cluster/project/jbuhmann/choij/sleep-stage-classification/dataset/
# zip -r dodh-processed-2.zip dodh-processed-2/
#/cluster/home/choij/miniconda3/envs/neuronet/bin/python ./preprocess/sleepEDF_cassette_process.py --multiprocess 1 --reduced --standard --root_folder "/cluster/project/jbuhmann/choij/NeuroNet/dataset/physionet.org/files/sleep-edfx/1.0.0/sleep-cassette" --dest_folder "/cluster/project/jbuhmann/choij/dataset"
#/cluster/home/choij/miniconda3/envs/neuronet/bin/python ./counting.py
#/cluster/home/choij/miniconda3/envs/neuronet/bin/python /cluster/project/jbuhmann/choij/sleep-stage-classification/preprocess/pca.py --root_folder "/cluster/project/jbuhmann/choij/sleep-stage-classification/dataset/dodh-processed-full"
/cluster/home/choij/miniconda3/envs/neuronet/bin/python ./output_eval_reader.py