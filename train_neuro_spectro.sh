#!/bin/bash 
# setup environment

#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --time=04:00:00
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --gres=gpumem:24G
#SBATCH --tmp=24G
#SBATCH -o "/cluster/project/jbuhmann/choij/sleep-stage-classification/output_log/09-05-25/train_neuronet-b_wavelet_tera_both-kfold-4.out"

# train_cassette_reduced_neuronet-b_recon_masked_time_patches
#module load stack/2024-06
nvidia-smi
conda activate neuronet

DATASET="cassette_processed_wavelet_full"

rsync /cluster/project/jbuhmann/choij/sleep-stage-classification/dataset/$DATASET.zip $TMPDIR
cd $TMPDIR
unzip -q $TMPDIR/$DATASET.zip

ls $TMPDIR/$DATASET/
ls $TMPDIR
cd /cluster/project/jbuhmann/choij/sleep-stage-classification

# hyperparams
TRAIN_EPOCHS=30
TRAIN_BS=256
MASK_RATIO=0.6
TEMP=0.5
FRAME_SIZE=3
TIME_STEP=0.75
BASE_PATH="${TMPDIR}/${DATASET}/"
MODEL_NAME="neuronet-b-${DATASET}-patches-spectro-tera-both"
MASKING="tera"
RECON_MODE="masked_tokens" #masked_time_patches or time_signal or masked_tokens
CONTRASTIVE=True
FREQ_MASK_RATIO=0.2
ALPHA=1.0
PATCH_SIZE=6

ENC_DIM=256
DEC_DIM=256
ENC_DEPTH=4
FREQ_BINS=30
LR=1e-4
FOLD=4

#JOB="/cluster/home/choij/miniconda3/envs/myenv/bin/python ./preprocess/sleepEDF_cassette_process.py"
/cluster/home/choij/miniconda3/envs/neuronet/bin/python /cluster/project/jbuhmann/choij/sleep-stage-classification/neuronet_kfold.py train.train_epochs=$TRAIN_EPOCHS train.train_batch_size=$TRAIN_BS dataset.base_path=$BASE_PATH train.model_name=$MODEL_NAME frames.time_window=$FRAME_SIZE frames.time_step=$TIME_STEP dataset.masking=$MASKING neuronet.recon_mode=$RECON_MODE neuronet.contrastive=$CONTRASTIVE neuronet.encoder_embed_dim=$ENC_DIM neuronet.decoder_embed_dim=$DEC_DIM neuronet.encoder_depths=$ENC_DEPTH neuronet.freq_bins=$FREQ_BINS train.train_base_learning_rate=$LR neuronet.freq_mask_ratio=$FREQ_MASK_RATIO neuronet.mask_ratio=$MASK_RATIO neuronet.alpha=$ALPHA spectro.patch_size=$PATCH_SIZE train.fold=$FOLD
#sbatch -o "$NAME" -n 1 --cpus-per-task "$NUM_WORKERS" --mem-per-cpu="$MEM_PER_CPU" --time="$TIME" -p gpu --gpus=1 --gres=gpumem:$MEM_PER_GPU --tmp=3g --wrap="nvidia-smi; $JOB"