# Sleep Stage Classification

This repository contains code for classifying sleep stages using various neural network models. The models are trained and evaluated on the Sleep-EDFX dataset, which needs to be preprocessed before use.

## Getting Started

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/choijacky/sleep-stage-classification.git
   cd sleep-stage-classification
   ```

2. Set up a conda environment and install dependencies:

   ```bash
   conda create --name sleep-env python=3.9
   conda activate sleep-env
   pip install -r requirements.txt
   ```

### Dataset

1. Download the Sleep-EDFX dataset:

   ```bash
   wget -r -N -c -np https://physionet.org/files/sleep-edfx/1.0.0/
   ```

2. Preprocess the dataset:

   Run the preprocessing script with appropriate arguments to generate train/val/test splits:

   ```bash
   python preprocess/sleepEDF_cassette_process.py --root_folder path/to/sleep-edfx --dest_folder ./dataset --multiprocess 8 --reduced --standard
   ```

   or to get the spectrograms, run:

   ```bash
   python preprocess/sleepEDFX_spectro.py --multiprocess 8 --reduced --standard --root_folder path/to/sleep-edfx --dest_folder ./dataset --dataset_name casette_reduced_spectro --time_freq_technique "wavelet_transform"
   ```

Analogously, there are files to run preprocessing for the DOD-H dataset, which can be obtained [here](https://github.com/Dreem-Organization/dreem-learning-open)

## Training

You can start training using the provided shell script. Modify `train_neuro.sh` with your desired hyperparameters, then run:

```bash
sbatch train_neuro.sh
```

### Shell Script Arguments

The shell scripts for training accept several arguments to customize the training process.

- `DATASET`: Name of the dataset to use.
- `TRAIN_EPOCHS`: Number of training epochs.
- `TRAIN_BS`: Batch size for training.
- `MASK_RATIO`: Ratio of the data to be masked.
- `TEMP`: Temperature parameter for certain loss functions.
- `FRAME_SIZE`: Size of one frame in seconds.
- `TIME_STEP`: Size of the stride of the backbone in seconds.

- `BASE_PATH`: Path to the preprocessed dataset.
- `MODEL_NAME`: Name of the model to be used or saved.
- `MASKING`: Type of masking technique used (e.g., "tera", "pca_sampling").
- `RECON_MODE`: Reconstruction mode (e.g., "masked_tokens", "masked_time_patches" or "time_signal").
- `CONTRASTIVE`: Whether to use contrastive learning.
- `ASYM_LOSS`: Whether to use asymmetric loss.
- `ALPHA`: Weight for the contrastive loss.
- `N_CHANNELS`: Number of channels in the input data.
- `LR`: Learning rate for the optimizer.
- `BALANCED_SAMPLING`: Whether to sample the batch according to a balanced class distribution.
- `FREQ_MASK_RATIO`: Masking ratio of frequency bins for spectrogram masking. Then `MASK_RATIO` defaults to being the ratio for the time bins.

## Model Architecture

The repository includes several neural network architectures, such as `NeuroNet`, `PCA_NeuroNet`, and `Spectro_NeuroNet`, each with unique configurations for handling sleep stage classification.

## Evaluation

After training, evaluate your models using the `linear_prob.py` script or other evaluation scripts provided in the `finetune` directory.

You can start training using the provided shell script. Modify `finetune.sh` with your desired hyperparameters, then run:

```bash
sbatch finetune.sh
```

### Shell Script Arguments

The shell scripts for finetuning accept several arguments to customize the training process.

- `EPOCHS`: Finetune epochs
- `BATCH_SIZE`: Eval batch size
- `BASE_PATH`: Path to dataset
- `LR`: Learning rate
- `MODEL_NAME`: Name of the model name to run evaluation
- `CKPT_PATH`: Path to the checkpoint file
- `N_CHANNELS`: Number of channels in the input data
- `FT_SPLIT`: Which split to finetune on. One of "train" and "val"
- `REPRESENTATION`: Which representation to pass to the linear probe. One of "CLS" or "Embed"
- `PATCH_SIZE`: (Optional) Patch size if you run masking in patches.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The Sleep-EDFX dataset is provided by PhysioNet.
- The project uses PyTorch Lightning for training and evaluation.