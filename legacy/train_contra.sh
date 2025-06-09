#!/bin/bash 
# setup environment

#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --time=72:00:00
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24G
#SBATCH --tmp=24G
#SBATCH -o "/cluster/project/jbuhmann/choij/sleep-stage-classification/output_log/07-03-25/train-cassette_processed_full-simsiam.out"

#module load stack/2024-06
nvidia-smi
conda activate neuronet

DATASET="cassette_processed_full"

rsync /cluster/project/jbuhmann/choij/sleep-stage-classification/dataset/$DATASET.zip $TMPDIR
cd $TMPDIR
unzip -q $TMPDIR/$DATASET.zip

ls $TMPDIR/$DATASET/
ls $TMPDIR
cd /cluster/project/jbuhmann/choij/sleep-stage-classification

# NUM_WORKERS=16
# TIME=72:00:00
# MEM_PER_CPU=16G
# MEM_PER_GPU=24G
# NAME="/cluster/project/jbuhmann/choij/sleep-stage-classification/output_log/train_neuro_bs_256.out"

# hyperparams
TRAIN_EPOCHS=100
TRAIN_BS=256
BASE_PATH="${TMPDIR}/${DATASET}/"
MODEL_NAME="SimSiam-${DATASET}"
N_CHN=2
LR=2e-4

#JOB="/cluster/home/choij/miniconda3/envs/myenv/bin/python ./preprocess/sleepEDF_cassette_process.py"
/cluster/home/choij/miniconda3/envs/neuronet/bin/python /cluster/project/jbuhmann/choij/sleep-stage-classification/neuronet_train.py train.train_epochs=$TRAIN_EPOCHS train.train_batch_size=$TRAIN_BS dataset.base_path=$BASE_PATH train.model_name=$MODEL_NAME frames.n_channels=$N_CHN train.train_base_learning_rate=$LR
#sbatch -o "$NAME" -n 1 --cpus-per-task "$NUM_WORKERS" --mem-per-cpu="$MEM_PER_CPU" --time="$TIME" -p gpu --gpus=1 --gres=gpumem:$MEM_PER_GPU --tmp=3g --wrap="nvidia-smi; $JOB"