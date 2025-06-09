import hydra
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import torch

import pytorch_lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

print(torch.__version__)


from data.datamodule import DataModule

@hydra.main(version_base=None, config_path="conf", config_name="train_defaults")
def main(config: DictConfig) -> None:

    print(config)
    # set up seed and tensorboard logger
    pl.seed_everything(456)
    tb_logger = TensorBoardLogger("tb_logs", name="my_model")


    # Create datamodule
    datamodule = instantiate(
        config.datamodule,
        masking=config.masking,
        pca_paths=config.pca_paths,
    )

    # create model
    model = instantiate(config.model_config)
    print("neuronet is instantiated")
    model_train = instantiate(
        config.pl_module, 
        model=model,
        masking = config.masking,
        datamodule = datamodule,
    )
    print(config.user)
    # Model Checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.user.user.ckpt_dir,  # Directory where to save the checkpoints
        filename='{epoch:02d}-{train_loss:.2f}',  # Filename format
        save_top_k=1,  # Save all checkpoints
        save_weights_only=False,  # Save the full model (True for weights only)
        every_n_epochs=1  # Save every epoch
    )

    # Init Trainer and train
    trainer_configs = OmegaConf.to_container(config.trainer, resolve=True)
    trainer = pl.Trainer(
        **trainer_configs,
        logger=tb_logger,
        enable_checkpointing = True,
        num_sanity_val_steps=0,
        callbacks=[checkpoint_callback],
        check_val_every_n_epoch=1,
    )

    print("------------------------- Start Training")
    trainer.fit(model_train, datamodule=datamodule)
    print("------------------------- End Training")
    # Eval data module

    # Final Eval

    







if __name__ == "__main__":
    main()