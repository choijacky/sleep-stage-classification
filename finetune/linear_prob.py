# -*- coding:utf-8 -*-
import os
import sys
import random
import warnings
import pickle
import torch
import torch.nn as nn
import torch.optim as opt
import numpy as np
import pandas as pd
import seaborn as sn
import mne

from typing import List
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from backbones.neuronet import (
    NeuroNet, NeuroNetEncoderWrapper, NeuroNetSpectroEncoderWrapper, 
    Spectro_NeuroNet, Spectro_NeuroNet_ViT
)
from backbones.cnn_backbone import CNNEncoder2D_SLEEP
from data.data_loader import SLEEPCALoader, SLEEPCALoader_spectro

import hydra
from omegaconf import OmegaConf

# Extend system path for module imports
sys.path.extend([
    os.path.abspath('.'),
    os.path.abspath('..'),
    "/cluster/project/jbuhmann/choij/sleep-stage-classification"
])

# Suppress warnings and set random seeds for reproducibility
warnings.filterwarnings('ignore')
random_seed = 777
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Classifier(nn.Module):
    """
    Linear classifier on top of a frozen backbone.
    """
    def __init__(self, backbone, backbone_final_length, model_name, representation):
        super().__init__()
        self.backbone = self.freeze_backbone(backbone)
        self.model_name = model_name
        self.representation = representation

        self.feature_num = backbone_final_length * 2
        self.fc = nn.Linear(backbone_final_length, self.feature_num)
        self.fc2 = nn.Sequential(
            nn.BatchNorm1d(self.feature_num),
            nn.Dropout(0.5),
            nn.Linear(self.feature_num, 5)
        )

    def forward(self, x):
        if isinstance(self.backbone, Spectro_NeuroNet_ViT):
            x = self.backbone.forward_latent(x)
        else:
            x = self.backbone(x)
            if 'neuronet' in self.model_name:
                x = x[:, :1, :].squeeze() if self.representation == "CLS" else torch.mean(x[:, 1:, :], dim=1)
        x = self.fc(x)
        return self.fc2(x)

    @staticmethod
    def freeze_backbone(backbone: nn.Module) -> nn.Module:
        """
        Freeze all parameters of the backbone to prevent gradient updates.
        """
        for param in backbone.parameters():
            param.requires_grad = False
        return backbone


class Trainer:
    """
    Trainer class for linear probing of a pretrained backbone.
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.ckpt_path = os.path.join(cfg.ckpt_path, 'model', 'best_model.pth')
        self.ckpt = torch.load(self.ckpt_path, map_location='cpu')

        self.ft_split = cfg.ft_split
        self.representation = cfg.representation
        self.patch_size = cfg.patch_size

        self.sfreq, self.rfreq = self.ckpt['hyperparameter']['_content']['dataset'].values()

        self.model = self.get_pretrained_model().to(device)
        self.optimizer = opt.AdamW(self.model.parameters(), lr=cfg.lr)
        self.scheduler = opt.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=cfg.epochs)
        self.criterion = nn.CrossEntropyLoss()

        self.tensorboard_writer = SummaryWriter(log_dir=os.path.join(cfg.ckpt_path, 'tensorboard'))

    def train(self):
        """
        Train the linear classifier and log performance.
        """
        val_loader, test_loader = self.setup_dataloaders()

        best_model_state = None
        best_mf1 = 0.0
        best_pred, best_real = [], []

        for epoch in range(self.cfg.epochs):
            self.model.train()
            train_losses = [
                self.criterion(self.model(x.to(device)), y.to(device)).item()
                for x, y in val_loader
            ]

            self.model.eval()
            test_losses, real_labels, predictions = [], [], []
            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.to(device), y.to(device)
                    logits = self.model(x)
                    test_losses.append(self.criterion(logits, y).item())
                    predictions.extend(logits.argmax(dim=-1).cpu().numpy())
                    real_labels.extend(y.cpu().numpy())

            eval_acc = accuracy_score(real_labels, predictions)
            eval_mf1 = f1_score(real_labels, predictions, average='macro')

            print(f"[Epoch {epoch + 1:03d}] Train Loss: {np.mean(train_losses):.4f}, "
                  f"Eval Loss: {np.mean(test_losses):.4f}, Accuracy: {eval_acc:.4f}, Macro-F1: {eval_mf1:.4f}")

            matrix = confusion_matrix(real_labels, predictions)
            per_class_acc = matrix.diagonal() / matrix.sum(axis=1)
            print("[Per-class accuracy] => ", per_class_acc)

            if eval_mf1 > best_mf1:
                best_mf1 = eval_mf1
                best_model_state = self.model.state_dict()
                best_pred, best_real = predictions, real_labels

            self.scheduler.step()

            # TensorBoard logging
            self.tensorboard_writer.add_scalar('Train Loss', np.mean(train_losses), epoch)
            self.tensorboard_writer.add_scalar('Eval Loss', np.mean(test_losses), epoch)
            self.tensorboard_writer.add_scalar('Eval Accuracy', eval_acc, epoch)
            self.tensorboard_writer.add_scalar('Eval F1', eval_mf1, epoch)

            # Confusion Matrix
            classes = ['W', 'N1', 'N2', 'N3', 'R']
            df_cm = pd.DataFrame(matrix / matrix.sum(axis=1)[:, None], index=classes, columns=classes)
            self.tensorboard_writer.add_figure("Confusion matrix", sn.heatmap(df_cm, annot=True).get_figure(), epoch)

        self.save_ckpt(best_model_state, best_pred, best_real)

    def save_ckpt(self, model_state, pred, real):
        """
        Save the best performing model.
        """
        save_dir = os.path.join(self.cfg.ckpt_path, 'linear_prob')
        os.makedirs(save_dir, exist_ok=True)
        torch.save({
            'backbone_name': 'NeuroNet_LinearProb',
            'model_state': model_state,
            'hyperparameter': self.cfg.__dict__,
            'result': {'real': real, 'pred': pred},
        }, os.path.join(save_dir, 'best_model.pth'))

    def get_pretrained_model(self) -> nn.Module:
        """
        Load and return the frozen backbone model based on checkpoint.
        """
        model_parameter = self.ckpt['model_parameter']
        model_name = self.cfg.model_name

        if 'patch' in model_name:
            backbone = Spectro_NeuroNet_ViT(
                time_dim=60, freq_dim=30, patch_size=self.patch_size, **model_parameter
            )
            backbone.load_state_dict(self.ckpt['model_state'])
            backbone_final_length = backbone.autoencoder.embed_dim

        elif 'spectro' in model_name:
            pretrained = Spectro_NeuroNet(**model_parameter, freq_bins=30)
            pretrained.load_state_dict(self.ckpt['model_state'])
            backbone = NeuroNetSpectroEncoderWrapper(
                fs=model_parameter['fs'], second=model_parameter['second'],
                time_window=model_parameter['time_window'], time_step=model_parameter['time_step'],
                patch_embed=pretrained.autoencoder.patch_embed,
                encoder_block=pretrained.autoencoder.encoder_block,
                encoder_norm=pretrained.autoencoder.encoder_norm,
                cls_token=pretrained.autoencoder.cls_token,
                pos_embed=pretrained.autoencoder.pos_embed,
                final_length=pretrained.autoencoder.embed_dim
            )
            backbone_final_length = pretrained.autoencoder.embed_dim

        elif 'neuronet' in model_name:
            pretrained = NeuroNet(**model_parameter)
            pretrained.load_state_dict(self.ckpt['model_state'])
            backbone = NeuroNetEncoderWrapper(
                fs=model_parameter['fs'], second=model_parameter['second'],
                time_window=model_parameter['time_window'], time_step=model_parameter['time_step'],
                frame_backbone=pretrained.frame_backbone,
                patch_embed=pretrained.autoencoder.patch_embed,
                encoder_block=pretrained.autoencoder.encoder_block,
                encoder_norm=pretrained.autoencoder.encoder_norm,
                cls_token=pretrained.autoencoder.cls_token,
                pos_embed=pretrained.autoencoder.pos_embed,
                final_length=pretrained.autoencoder.embed_dim
            )
            backbone_final_length = pretrained.autoencoder.embed_dim

        else:
            backbone = CNNEncoder2D_SLEEP(self.ckpt['hyperparameter']['_content']['contra']['n_dim'])
            backbone.load_state_dict(self.ckpt['model_state'])
            backbone_final_length = backbone.n_dim

        return Classifier(backbone, backbone_final_length, model_name, self.representation)

    def setup_dataloaders(self):
        """
        Prepare DataLoader objects for validation and test splits.
        """
        val_dir = os.path.join(self.cfg.base_path, self.ft_split)
        test_dir = os.path.join(self.cfg.base_path, "test")
        val_index = os.listdir(val_dir)
        test_index = os.listdir(test_dir)

        if 'spectro' in self.cfg.model_name:
            val_dataset = SLEEPCALoader_spectro(val_index, val_dir)
            test_dataset = SLEEPCALoader_spectro(test_index, test_dir)
        else:
            val_dataset = SLEEPCALoader(val_index, val_dir, self.cfg.n_channels, False)
            test_dataset = SLEEPCALoader(test_index, test_dir, self.cfg.n_channels, False)

        val_loader = DataLoader(val_dataset, batch_size=self.cfg.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.cfg.batch_size)

        return val_loader, test_loader


@hydra.main(version_base=None, config_path="../conf", config_name="finetune_config")
def finetune(cfg) -> None:
    """
    Hydra entry point for linear probing.
    """
    print(OmegaConf.to_yaml(cfg))
    trainer = Trainer(cfg)
    trainer.train()


if __name__ == '__main__':
    finetune()
