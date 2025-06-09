#!/bin/bash 
# setup environment 
conda activate neuronet

DATASET="cassette_processed_wavelet_full"

# hyperparams
TRAIN_EPOCHS=100
TRAIN_BS=512
LR=1e-4
N_CHANNELS=1
FT_SPLIT="train"
RECON_MODE="tokens"
REPRESENTATION="CLS"
SOFT_LABEL=False
FOLD=1
PATCH_SIZE=6

NUM_WORKERS=8
CPUS_PER_TASK=16
MEM_PER_CPU=8G
TIME=03:00:00
MEM_PER_GPU=24G
TMP=24G

MASKs=(0.2)

for MASK in "${MASKs[@]}"
do
# for F_MASK in "${MASKs[@]}"
# do
TMP_NAME="${DATASET}-pca-masking-${MASK}"
TMP_DIR="/scratch/${TMP_NAME}/"
BASE_PATH="${TMP_DIR}/${DATASET}/"
MODEL_NAME="neuronet-b-${DATASET}-tera_masking-patch-${MASK}-1"
CKPT_PATH="/cluster/project/jbuhmann/choij/sleep-stage-classification/ckpt/Sleep-EDFX/${MODEL_NAME}/"
OUTPUT_LOG="/cluster/project/jbuhmann/choij/sleep-stage-classification/output_log/tera_new_mask_ablation/linear_prob_${DATASET}_neuro-b-mask-${MASK}.out"
JOB="/cluster/home/choij/miniconda3/envs/neuronet/bin/python /cluster/project/jbuhmann/choij/sleep-stage-classification/finetune/kfold_eval_sleepedfx.py epochs=${TRAIN_EPOCHS} batch_size=${TRAIN_BS} ckpt_path=${CKPT_PATH} lr=$LR base_path=${BASE_PATH} model_name=${MODEL_NAME} n_channels=${N_CHANNELS} ft_split=${FT_SPLIT} fold=${FOLD} soft_label=${SOFT_LABEL} representation=${REPRESENTATION} patch_size=${PATCH_SIZE}"
sbatch -o "${OUTPUT_LOG}" -n 1 --cpus-per-task "$NUM_WORKERS" --mem-per-cpu="$MEM_PER_CPU" --time="$TIME" -p gpu --gpus=1 --gres=gpumem:$MEM_PER_GPU --wrap="nvidia-smi;rsync /cluster/project/jbuhmann/choij/sleep-stage-classification/dataset/${DATASET}.zip $TMP_DIR;cd $TMP_DIR;unzip -q $TMP_DIR/${DATASET}.zip;ls $TMP_DIR/${DATASET}/;ls $TMP_DIR;cd /cluster/project/jbuhmann/choij/sleep-stage-classification;$JOB; rm -rf $TMP_DIR"
done;