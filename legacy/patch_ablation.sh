#!/bin/bash 
# setup environment 
conda activate neuronet

DATASET="cassette_processed_wavelet_full"

# hyperparams
TRAIN_EPOCHS=50
TRAIN_BS=256
TEMP=0.5
FRAME_SIZE=3
TIME_STEP=0.75
MASKING="patch"
RECON_MODE="masked_tokens" #masked_time_patches or time_signal or masked_tokens
CONTRASTIVE=True
ENC_DIM=256
DEC_DIM=128
ENC_DEPTH=4
FREQ_BINS=30
LR=1e-4
FOLD=0

#FREQ_MASK_RATIO=0.0
#MASK_RATIO=0.0

NUM_WORKERS=8
CPUS_PER_TASK=16
MEM_PER_CPU=8G
TIME=24:00:00
MEM_PER_GPU=24G
TMP=24G

#MASKs=(0.2 0.4 0.6 0.8)
MASK=0.5

PATCHs=(2 3 5 6)


for PATCH in "${PATCHs[@]}"
do
TMP_NAME="${DATASET}-masking-${MASK}-${PATCH}"
TMP_DIR="/scratch/${TMP_NAME}/"
BASE_PATH="${TMP_DIR}/${DATASET}/"
MODEL_NAME="neuronet-b-${DATASET}-patch-${MASK}-${PATCH}-spectro"
OUTPUT_LOG="/cluster/project/jbuhmann/choij/sleep-stage-classification/output_log/patch_ablation/train_${DATASET}_neuro-b-mask-${MASK}-${PATCH}.out"
JOB="/cluster/home/choij/miniconda3/envs/neuronet/bin/python /cluster/project/jbuhmann/choij/sleep-stage-classification/neuronet_kfold.py train.train_epochs=${TRAIN_EPOCHS} train.train_batch_size=${TRAIN_BS} dataset.base_path=${BASE_PATH} train.model_name=${MODEL_NAME} frames.time_window=${FRAME_SIZE} frames.time_step=${TIME_STEP} neuronet.mask_ratio=${MASK} neuronet.asym_loss=False dataset.masking=${MASKING} neuronet.recon_mode=${RECON_MODE} neuronet.contrastive=${CONTRASTIVE} neuronet.encoder_embed_dim=${ENC_DIM} neuronet.decoder_embed_dim=${DEC_DIM} neuronet.encoder_depths=${ENC_DEPTH} neuronet.freq_bins=${FREQ_BINS} train.train_base_learning_rate=${LR} spectro.patch_size=${PATCH} train.fold=${FOLD}"
sbatch -o "${OUTPUT_LOG}" -n 1 --cpus-per-task "$NUM_WORKERS" --mem-per-cpu="$MEM_PER_CPU" --time="$TIME" -p gpu --gpus=1 --gres=gpumem:$MEM_PER_GPU --wrap="nvidia-smi;rsync /cluster/project/jbuhmann/choij/sleep-stage-classification/dataset/${DATASET}.zip $TMP_DIR;cd $TMP_DIR;unzip -q $TMP_DIR/${DATASET}.zip;ls $TMP_DIR/${DATASET}/;ls $TMP_DIR;cd /cluster/project/jbuhmann/choij/sleep-stage-classification;$JOB; rm -rf $TMP_DIR"
done;