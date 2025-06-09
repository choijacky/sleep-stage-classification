#!/bin/bash 
# setup environment

#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --time=24:00:00
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24G
#SBATCH --tmp=24G
#SBATCH -o "/cluster/project/jbuhmann/choij/sleep-stage-classification/output_log/13-04-25/train-neuronet-dodh-processed-kfold-debug.out"

# train_cassette_reduced_neuronet-b_recon_masked_time_patches
#module load stack/2024-06
nvidia-smi
conda activate neuronet

FOLD=0

DATASET="dodh-processed-kfold"

rsync /cluster/project/jbuhmann/choij/sleep-stage-classification/dataset/$DATASET.zip $TMPDIR
cd $TMPDIR
unzip -q $TMPDIR/$DATASET.zip

mv /cluster/project/jbuhmann/choij/sleep-stage-classification/dataset/$DATASET/$FOLD/*.pkl /cluster/project/jbuhmann/choij/sleep-stage-classification/dataset/$DATASET/test/

for I in {0..4}
do
if ! [ "$FOLD" -eq "$I" ];
then
    mv /cluster/project/jbuhmann/choij/sleep-stage-classification/dataset/$DATASET/$I/*.pkl /cluster/project/jbuhmann/choij/sleep-stage-classification/dataset/$DATASET/train/
fi
done;


ls $TMPDIR/$DATASET/
ls $TMPDIR
cd /cluster/project/jbuhmann/choij/sleep-stage-classification

# hyperparams
TRAIN_EPOCHS=3
TRAIN_BS=256
MASK_RATIO=0.5
TEMP=0.5
FRAME_SIZE=3
TIME_STEP=0.75
BASE_PATH="${TMPDIR}/${DATASET}/"
MODEL_NAME="neuronet-b-${DATASET}-${FOLD}"
MASKING="token"
RECON_MODE="masked_tokens" #masked_time_patches or time_signal or masked_tokens
CONTRASTIVE=True
ASYM_LOSS=False
ALPHA=1.0
N_CHANNELS=1
LR=1e-4

#JOB="/cluster/home/choij/miniconda3/envs/myenv/bin/python ./preprocess/sleepEDF_cassette_process.py"
/cluster/home/choij/miniconda3/envs/neuronet/bin/python /cluster/project/jbuhmann/choij/sleep-stage-classification/neuronet_train.py train.train_epochs=$TRAIN_EPOCHS train.train_batch_size=$TRAIN_BS dataset.base_path=$BASE_PATH train.model_name=$MODEL_NAME frames.time_window=$FRAME_SIZE frames.time_step=$TIME_STEP dataset.masking=$MASKING neuronet.alpha=$ALPHA neuronet.asym_loss=$ASYM_LOSS neuronet.recon_mode=$RECON_MODE neuronet.contrastive=$CONTRASTIVE frames.n_channels=$N_CHANNELS train.train_base_learning_rate=$LR
#sbatch -o "$NAME" -n 1 --cpus-per-task "$NUM_WORKERS" --mem-per-cpu="$MEM_PER_CPU" --time="$TIME" -p gpu --gpus=1 --gres=gpumem:$MEM_PER_GPU --tmp=3g --wrap="nvidia-smi; $JOB"