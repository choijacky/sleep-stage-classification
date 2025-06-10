#!/bin/bash 
# Job for finetuning

#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --time=04:00:00
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --gres=gpumem:8G
#SBATCH --tmp=8G
#SBATCH -o "/cluster/project/jbuhmann/choij/sleep-stage-classification/output_log/16-04-25/neuronet-b-cassette_reduced_standard-masked_token-finetune-dodh-processed-retry-softlabels.out"

#module load stack/2024-06
nvidia-smi
conda activate neuronet2

DATASET='cassette_reduced_standard' # cassette_reduced_contrawr

rsync /cluster/project/jbuhmann/choij/sleep-stage-classification/dataset/$DATASET.zip $TMPDIR
cd $TMPDIR
unzip -q $TMPDIR/$DATASET.zip

ls $TMPDIR/$DATASET/
ls $TMPDIR

#mkdir $TMPDIR/$DATASET/Subgroup_2

cd /cluster/project/jbuhmann/choij/sleep-stage-classification

# hyperparams
EPOCHS=50
BATCH_SIZE=512
BASE_PATH="${TMPDIR}/${DATASET}/"
LR=1e-4
MODEL_NAME="neuronet-b-cassette_reduced_standard-balanced-1e-5" #"neuronet-b-isruc_3" neuronet-b-cassette_reduced_standard-masked_token
CKPT_PATH=/cluster/project/jbuhmann/choij/sleep-stage-classification/ckpt/Sleep-EDFX/$MODEL_NAME/
N_CHANNELS=1
FT_SPLIT="train"
REPRESENTATION="CLS"
PATCH_SIZE=5
#DOD_SCORER=1

#JOB="/cluster/home/choij/miniconda3/envs/myenv/bin/python ./preprocess/sleepEDF_cassette_process.py"
/cluster/home/choij/miniconda3/envs/neuronet2/bin/python /cluster/project/jbuhmann/choij/sleep-stage-classification/finetune/linear_prob.py epochs=$EPOCHS batch_size=$BATCH_SIZE ckpt_path=$CKPT_PATH lr=$LR base_path=$BASE_PATH model_name=$MODEL_NAME n_channels=$N_CHANNELS ft_split=$FT_SPLIT representation=$REPRESENTATION # patch_size=$PATCH_SIZE dod_scorer=$DOD_SCORER
#sbatch -o "$NAME" -n 1 --cpus-per-task "$NUM_WORKERS" --mem-per-cpu="$MEM_PER_CPU" --time="$TIME" -p gpu --gpus=1 --gres=gpumem:$MEM_PER_GPU --tmp=3g --wrap="nvidia-smi; $JOB"