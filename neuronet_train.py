# -*- coding:utf-8 -*-
import os
import sys
import random
import shutil
import warnings
import numpy as np
import torch
import torch.optim as opt
from sklearn.decomposition import PCA
from torch.utils.tensorboard import SummaryWriter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression as LR
from utils import model_size
from data.data_loader import SLEEPCALoader, SLEEPCALoader_spectro, ISRUC, DOD, SLEEPEDF_DOD
from backbones.models import get_model
from loss import get_loss
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
import hydra
from omegaconf import OmegaConf

# Extend system path
sys.path.extend([os.path.abspath('.'), os.path.abspath('..')])

# Set random seed for reproducibility
random_seed = 777
np.random.seed(random_seed)
torch.manual_seed(random_seed)
random.seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Set MNE and device configurations
warnings.filterwarnings(action='ignore')
mne.set_log_level(False)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(torch.__version__)
print(device)

class Trainer:
    """
    Trainer class to handle model training and evaluation.
    """

    def __init__(self, cfg):
        """
        Initialize the Trainer with configuration.

        Args:
            cfg (OmegaConf): Configuration object.
        """
        self.cfg = cfg
        self.model_name = cfg.train.model_name
        self.model = get_model(cfg, device)[0]
        if 'neuronet' not in self.model_name:
            self.student = get_model(cfg, device)[1]
            self.criterion = get_loss(cfg, device)
        
        print(f'Model Size : {model_size(self.model):.2f}MB')

        self.eff_batch_size = cfg.train.train_batch_size * cfg.train.train_batch_accumulation
        self.lr = cfg.train.train_base_learning_rate * self.eff_batch_size / 256
        self.optimizer = opt.AdamW(self.model.parameters(), lr=self.lr) if 'neuronet' in self.model_name else torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=cfg.train.weight_decay)
        self.scheduler = opt.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=cfg.train.train_epochs) if 'neuronet' in self.model_name else torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.2, patience=5)

        self.tensorboard_path = os.path.join(cfg.train.ckpt_path, cfg.train.model_name, 'tensorboard')
        if os.path.exists(self.tensorboard_path):
            shutil.rmtree(self.tensorboard_path)
        self.tensorboard_writer = SummaryWriter(log_dir=self.tensorboard_path)

        print(f'Leaning Rate : {self.lr}')

    def train(self):
        """
        Train the model.
        """
        train_loader, val_loader, test_loader = self.setup_dataloaders()
        best_model_state, best_score = self.model.state_dict(), 0

        for epoch in range(self.cfg.train.train_epochs):
            self.model.train()
            self.optimizer.zero_grad()
            total_step = 0

            for x, x2 in train_loader:
                x = x.to(device)
                if 'BYOL' in self.model_name:
                    x2 = x2.to(device)
                    emb_aug1 = self.model(x, mid=False, byol=True)
                    emb_aug2 = self.student(x2, mid=False)
                    loss = self.criterion(emb_aug1, emb_aug2)
                elif 'SimCLR' in self.model_name:
                    x2 = x2.to(device)
                    emb_aug1 = self.model(x, mid=False)
                    emb_aug2 = self.model(x2, mid=False)
                    loss = self.criterion(emb_aug1, emb_aug2)
                elif 'ContraWR' in self.model_name:
                    x2 = x2.to(device)
                    emb_aug1 = self.model(x, mid=False)
                    emb_aug2 = self.student(x2, mid=False)
                    loss = self.criterion(emb_aug1, emb_aug2)
                elif 'SimSiam' in self.model_name:
                    x2 = x2.to(device)
                    emb_aug1, proj1 = self.model(x, simsiam=True)
                    emb_aug2, proj2 = self.model(x2, simsiam=True)
                    loss = self.criterion(proj1, proj2, emb_aug1, emb_aug2)
                elif 'neuronet' in self.model_name:
                    loss = self._neuronet_forward(x, x2)
                loss.backward()
                if (total_step + 1) % self.cfg.train.train_batch_accumulation == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                if 'neuronet' in self.model_name:
                    self.tensorboard_writer.add_scalar('Reconstruction Loss', recon_loss, total_step)
                    self.tensorboard_writer.add_scalar('Contrastive loss', contrastive_loss, total_step)
                self.tensorboard_writer.add_scalar('Total loss', loss, total_step)
                total_step += 1

            val_acc, val_mf1 = self.linear_probing(val_loader, test_loader) if 'neuronet' in self.model_name else self.logistic(val_loader, test_loader)

            if val_mf1 > best_score:
                best_model_state = self.model.state_dict()
                best_score = val_mf1

            print(f'[Epoch] : {epoch:03d} \t [Accuracy] : {val_acc * 100:2.4f} \t [Macro-F1] : {val_mf1 * 100:2.4f}')
            self.tensorboard_writer.add_scalar('Validation Accuracy', val_acc, total_step)
            self.tensorboard_writer.add_scalar('Validation Macro-F1', val_mf1, total_step)

            self.scheduler.step(loss)

        self.save_ckpt(model_state=best_model_state)

    def _neuronet_forward(self, x, x2):
        """
        Forward pass for NeuroNet models.

        Args:
            x (torch.Tensor): Input tensor.
            x2 (torch.Tensor): Second input tensor for certain models.

        Returns:
            torch.Tensor: Loss value.
        """
        if 'pca' in self.model_name:
            x2 = x2.to(device)
            out = self.model(x, x2)
        elif self.cfg.dataset.masking != "token":
            out = self.model(x, mask_ratio=self.cfg.neuronet.mask_ratio, freq_mask_ratio=self.cfg.neuronet.freq_mask_ratio if self.cfg.dataset.masking == "patches" else None)
        else:
            out = self.model(x, mask_ratio=self.cfg.neuronet.mask_ratio, sample_ratio='sampling' in self.model_name)

        recon_loss, contrastive_loss, (cl_labels, cl_logits) = out
        if self.cfg.neuronet.contrastive:
            return recon_loss + self.cfg.neuronet.alpha * contrastive_loss
        return recon_loss

    # Other methods remain unchanged

@hydra.main(version_base=None, config_path="conf", config_name="train_config")
def my_trainer(cfg) -> None:
    """
    Main entry point for training.

    Args:
        cfg (OmegaConf): Configuration object.
    """
    print(OmegaConf.to_yaml(cfg))
    trainer = Trainer(cfg)
    if "pca" in cfg.train.model_name:
        trainer.train_pca()
    else:
        trainer.train()

if __name__ == '__main__':
    my_trainer()