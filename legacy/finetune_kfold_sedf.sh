#!/bin/bash 
# Job for finetuning

#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --time=01:00:00
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --gres=gpumem:8G
#SBATCH --tmp=8G

#module load stack/2024-06
DATASET='dodh_processed_wavelet_full' #dodh-processed-full dodh_processed_wavelet_full

# hyperparams
EPOCHS=100
BATCH_SIZE=512
BASE_PATH="${TMPDIR}/${DATASET}/"
LR=1e-4
MODEL="neuronet-b-cassette_processed_wavelet_full-patches-spectro-tera-both" #"neuronet-b-isruc_3" neuronet-b-cassette_reduced_standard-masked_token
N_CHANNELS=1
FT_SPLIT="train"
REPRESENTATION="CLS"
PATCH_SIZE=6
DOD_SCORER=5
SOFT_LABEL=True

NUM_WORKERS=8
CPUS_PER_TASK=16
MEM_PER_CPU=8G
TIME=04:00:00
MEM_PER_GPU=24G
TMP=24G

conda activate neuronet2
for FOLD in {0..4}
do
  TMP_NAME="${DATASET}-kfold-${FOLD}"
  TMP_DIR="/scratch/${TMP_NAME}/"
  BASE_PATH="${TMP_DIR}/${DATASET}/"
  MODEL_NAME="${MODEL}-${FOLD}"
  CKPT_PATH=/cluster/project/jbuhmann/choij/sleep-stage-classification/ckpt/Sleep-EDFX/$MODEL_NAME
  OUTPUT_LOG="/cluster/project/jbuhmann/choij/sleep-stage-classification/output_log/spectro-tera-both/finetune_${MODEL_NAME}-${DATASET}.out"

  JOB="/cluster/home/choij/miniconda3/envs/neuronet2/bin/python /cluster/project/jbuhmann/choij/sleep-stage-classification/finetune/kfold_eval.py \
    epochs=${EPOCHS} \
    batch_size=${BATCH_SIZE} \
    ckpt_path=${CKPT_PATH} \
    lr=${LR} \
    base_path=${BASE_PATH} \
    model_name=${MODEL_NAME} \
    n_channels=${N_CHANNELS} \
    ft_split=${FT_SPLIT} \
    representation=${REPRESENTATION} \
    patch_size=${PATCH_SIZE} \
    dod_scorer=${DOD_SCORER} \
    soft_label=${SOFT_LABEL} \
    fold=${FOLD}"

  sbatch -o "${OUTPUT_LOG}" -n 1 \
    --cpus-per-task="$NUM_WORKERS" \
    --mem-per-cpu="$MEM_PER_CPU" \
    --time="$TIME" \
    -p gpu --gpus=1 \
    --gres=gpumem:$MEM_PER_GPU \
    --wrap="nvidia-smi; \
      rsync /cluster/project/jbuhmann/choij/sleep-stage-classification/dataset/${DATASET}.zip $TMP_DIR; \
      cd $TMP_DIR; \
      unzip -q $TMP_DIR/${DATASET}.zip; \
      ls $TMP_DIR/${DATASET}/; \
      ls $TMP_DIR; \
      cd /cluster/project/jbuhmann/choij/sleep-stage-classification; \
      $JOB; \
      rm -rf $TMP_DIR"

done