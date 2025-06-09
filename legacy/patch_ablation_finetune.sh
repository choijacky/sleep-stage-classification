#!/bin/bash 
# setup environment 
conda activate neuronet

DATASET="cassette_reduced_wavelet"

# hyperparams
TRAIN_EPOCHS=300
TRAIN_BS=512
LR=1e-4
N_CHANNELS=1
FT_SPLIT="train"
RECON_MODE="tokens"

NUM_WORKERS=8
CPUS_PER_TASK=16
MEM_PER_CPU=8G
TIME=72:00:00
MEM_PER_GPU=24G
TMP=24G

MASKs=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
PATCHs=(2 3 5 6)

for MASK in "${MASKs[@]}"
do
for PATCH in "${PATCHs[@]}"
do
TMP_NAME="${DATASET}-masking-${MASK}-${PATCH}"
TMP_DIR="/scratch/${TMP_NAME}/"
BASE_PATH="${TMP_DIR}/${DATASET}/"
MODEL_NAME="neuronet-b-${DATASET}-patch-${MASK}-${PATCH}-spectro"
CKPT_PATH="/cluster/project/jbuhmann/choij/sleep-stage-classification/ckpt/Sleep-EDFX/${MODEL_NAME}/"
OUTPUT_LOG="/cluster/project/jbuhmann/choij/sleep-stage-classification/output_log/patch_ablation/linear_prob_${DATASET}_neuro-b-mask-${MASK}-${PATCH}.out"
JOB="/cluster/home/choij/miniconda3/envs/neuronet/bin/python /cluster/project/jbuhmann/choij/sleep-stage-classification/finetune/linear_prob.py epochs=${TRAIN_EPOCHS} batch_size=${TRAIN_BS} ckpt_path=${CKPT_PATH} lr=$LR base_path=${BASE_PATH} model_name=${MODEL_NAME} n_channels=${N_CHANNELS} ft_split=${FT_SPLIT} patch_size=${PATCH}"
sbatch -o "${OUTPUT_LOG}" -n 1 --cpus-per-task "$NUM_WORKERS" --mem-per-cpu="$MEM_PER_CPU" --time="$TIME" -p gpu --gpus=1 --gres=gpumem:$MEM_PER_GPU --wrap="nvidia-smi;rsync /cluster/project/jbuhmann/choij/sleep-stage-classification/dataset/${DATASET}.zip $TMP_DIR;cd $TMP_DIR;unzip -q $TMP_DIR/${DATASET}.zip;ls $TMP_DIR/${DATASET}/;ls $TMP_DIR;cd /cluster/project/jbuhmann/choij/sleep-stage-classification;$JOB; rm -rf $TMP_DIR"
done
done;