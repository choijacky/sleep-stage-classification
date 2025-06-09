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
#SBATCH -o "/cluster/project/jbuhmann/choij/sleep-stage-classification/output_log/04-03-25/train_neuronet-b_cassette_reduced_contrawr-pca_sampling.out"

#module load stack/2024-06
nvidia-smi
conda activate neuronet

DATASET="cassette_reduced_standard"

rsync /cluster/project/jbuhmann/choij/sleep-stage-classification/dataset/$DATASET.zip $TMPDIR
cd $TMPDIR
unzip -q $TMPDIR/$DATASET.zip

ls $TMPDIR/$DATASET/
ls $TMPDIR
cd /cluster/project/jbuhmann/choij/sleep-stage-classification

# hyperparams
TRAIN_EPOCHS=50
TRAIN_BS=256
MASK_RATIO=0.4
TEMP=0.5
FRAME_SIZE=3
TIME_STEP=0.75
BASE_PATH="${TMPDIR}/${DATASET}/"
MODEL_NAME="neuronet-b-${DATASET}-pca-sampling"
ALPHA=1.0
MASKING="pca_sampling"
ASYM=False
FOLD=0

#JOB="/cluster/home/choij/miniconda3/envs/myenv/bin/python ./preprocess/sleepEDF_cassette_process.py"
/cluster/home/choij/miniconda3/envs/neuronet/bin/python /cluster/project/jbuhmann/choij/sleep-stage-classification/neuronet_kfold.py train.train_epochs=$TRAIN_EPOCHS train.train_batch_size=$TRAIN_BS dataset.base_path=$BASE_PATH train.model_name=$MODEL_NAME frames.time_window=$FRAME_SIZE frames.time_step=$TIME_STEP neuronet.mask_ratio=$MASK_RATIO dataset.masking=$MASKING neuronet.alpha=$ALPHA neuronet.asym_loss=$ASYM
#sbatch -o "$NAME" -n 1 --cpus-per-task "$NUM_WORKERS" --mem-per-cpu="$MEM_PER_CPU" --time="$TIME" -p gpu --gpus=1 --gres=gpumem:$MEM_PER_GPU --tmp=3g --wrap="nvidia-smi; $JOB"