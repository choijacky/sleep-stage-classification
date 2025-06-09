# -*- coding:utf-8 -*-
import os
import sys
sys.path.extend([os.path.abspath('.'), os.path.abspath('..'), "/cluster/project/jbuhmann/choij/sleep-stage-classification"])

print(sys.path)

import mne
import pickle
import torch
import random
import warnings
import numpy as np
import torch.nn as nn
from typing import List
import torch.optim as opt
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from backbones.neuronet import NeuroNet, NeuroNetEncoderWrapper, NeuroNetSpectroEncoderWrapper, Spectro_NeuroNet, Spectro_NeuroNet_ViT
from backbones.cnn_backbone import CNNEncoder2D_SLEEP
from data.data_loader import SLEEPCALoader, SLEEPCALoaderComb
from torch.utils.tensorboard import SummaryWriter
import seaborn as sn
import pandas as pd

import hydra
from omegaconf import OmegaConf

warnings.filterwarnings(action='ignore')


random_seed = 777
np.random.seed(random_seed)
torch.manual_seed(random_seed)
random.seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Classifier(nn.Module):
    def __init__(self, backbone, backbone_final_length, model_name, representation):
        super().__init__()
        self.backbone = self.freeze_backbone(backbone)
        self.backbone_final_length = backbone_final_length
        self.feature_num = self.backbone_final_length * 2
        self.dropout_p = 0.5
        self.model_name = model_name
        self.representation = representation
        self.fc = nn.Sequential(
            nn.Linear(2 * backbone_final_length, self.feature_num),
        )
        self.fc2 = nn.Sequential(
            nn.BatchNorm1d(self.feature_num),
            nn.ELU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(self.feature_num, 5)
        )

    def forward(self, x):
        if isinstance(self.backbone, Spectro_NeuroNet_ViT):
            x = self.backbone.forward_latent(x)
        else:
            x1 = x[:, 0, :]
            x2 = x[:, 1, :]

            x1 = self.backbone(x1)
            x2 = self.backbone(x2)
        
            if 'neuronet' in self.model_name:
                if self.representation == "CLS":
                    x1 = x1[:, :1 , :].squeeze() #take the CLS token
                    x2 = x2[:, :1 , :].squeeze() #take the CLS token
                else:
                    x1 = torch.mean(x1[:, 1: , :], dim=1)
                    x2 = torch.mean(x2[:, 1: , :], dim=1)

        x = torch.cat((x1, x2), 1)

        x = self.fc(x)
        x = self.fc2(x)
        return x

    @staticmethod
    def freeze_backbone(backbone: nn.Module):
        for name, module in backbone.named_modules():
            for param in module.parameters():
                param.requires_grad = False
        return backbone

class Trainer(object):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        print(cfg)
        self.ckpt_path = os.path.join(self.cfg.ckpt_path, 'model', 'best_model.pth')
        self.ft_split = self.cfg.ft_split
        self.representation = self.cfg.representation
        self.ckpt = torch.load(self.ckpt_path, map_location='cpu')
        self.sfreq, self.rfreq = self.ckpt['hyperparameter']['_content']['dataset']['sfreq'], self.ckpt['hyperparameter']['_content']['dataset']['rfreq']
        self.model = self.get_pretrained_model().to(device)
        self.optimizer = opt.AdamW(self.model.parameters(), lr=self.cfg.lr)
        self.scheduler = opt.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.cfg.epochs)
        self.criterion = nn.CrossEntropyLoss()

        self.patch_size = cfg.patch_size

        self.tensorboard_path = os.path.join(self.cfg.ckpt_path, 'tensorboard')
        self.tensorboard_writer = SummaryWriter(log_dir=self.tensorboard_path)

    def train(self):
        print('Checkpoint File Path : {}'.format(self.ckpt_path))

        val_dataloader, test_dataloader = self.setup_dataloaders()

        best_model_state, best_mf1 = None, 0.0
        best_pred, best_real = [], []

        step = 0

        for epoch in range(self.cfg.epochs):
            self.model.train()
            epoch_train_loss = []
            for data in val_dataloader:
                self.optimizer.zero_grad()
                x, y = data
                x, y = x.to(device), y.to(device)

                pred = self.model(x)

                loss = self.criterion(pred, y)


                epoch_train_loss.append(float(loss.detach().cpu().item()))
                loss.backward()
                self.optimizer.step()

            self.model.eval()
            epoch_test_loss = []
            epoch_real, epoch_pred = [], []
            for data in test_dataloader:
                with torch.no_grad():
                    x, y = data
                    x, y = x.to(device), y.to(device)
                    pred = self.model(x)
                    loss = self.criterion(pred, y)
                    pred = pred.argmax(dim=-1)
                    real = y

                    epoch_real.extend(list(real.detach().cpu().numpy()))
                    epoch_pred.extend(list(pred.detach().cpu().numpy()))
                    epoch_test_loss.append(float(loss.detach().cpu().item()))


            epoch_train_loss, epoch_test_loss = np.mean(epoch_train_loss), np.mean(epoch_test_loss)
            eval_acc, eval_mf1 = accuracy_score(y_true=epoch_real, y_pred=epoch_pred), \
                                 f1_score(y_true=epoch_real, y_pred=epoch_pred, average='macro')

            print('[Epoch] : {0:03d} \t '
                  '[Train Loss] => {1:.4f} \t '
                  '[Evaluation Loss] => {2:.4f} \t '
                  '[Evaluation Accuracy] => {3:.4f} \t'
                  '[Evaluation Macro-F1] => {4:.4f}'.format(epoch + 1, epoch_train_loss, epoch_test_loss,
                                                            eval_acc, eval_mf1))
            
            matrix = confusion_matrix(y_true=epoch_real, y_pred=epoch_pred)
            per_class = matrix.diagonal()/matrix.sum(axis=1)
            print('[Per-class accuracy] => ', per_class)

            if best_mf1 < eval_mf1:
                best_mf1 = eval_mf1
                best_model_state = self.model.state_dict()
                best_pred, best_real = epoch_pred, epoch_real

            self.scheduler.step()

            self.tensorboard_writer.add_scalar('Finetune Train Loss', epoch_train_loss, epoch)
            self.tensorboard_writer.add_scalar('Finetune Eval Loss', epoch_test_loss, epoch)
            self.tensorboard_writer.add_scalar('Finetune Eval Acc', eval_acc, epoch)
            self.tensorboard_writer.add_scalar('Finetune Eval F1', eval_mf1, epoch)

            classes = ['W', 'N1', 'N2', 'N3', 'R']

            df_cm = pd.DataFrame(matrix / np.sum(matrix, axis=1)[:, None], index=classes,
                columns=classes)

            self.tensorboard_writer.add_figure("Confusion matrix", sn.heatmap(df_cm, annot=True).get_figure(), epoch)

            step += 1

        self.save_ckpt(best_model_state, best_pred, best_real)

    def save_ckpt(self, model_state, pred, real):
        if not os.path.exists(os.path.join(self.cfg.ckpt_path, 'linear_prob')):
            os.makedirs(os.path.join(self.cfg.ckpt_path, 'linear_prob'))

        save_path = os.path.join(self.cfg.ckpt_path, 'linear_prob', 'best_model.pth')
        torch.save({
            'backbone_name': 'NeuroNet_LinearProb',
            'model_state': model_state,
            'hyperparameter': self.cfg.__dict__,
            'result': {'real': real, 'pred': pred},
        }, save_path)

    def get_pretrained_model(self):
        # 1. Prepared Pretrained Model
        model_parameter = self.ckpt['model_parameter']

        if 'patch' in self.cfg.model_name:
            backbone = Spectro_NeuroNet_ViT(
                time_dim=60,
                freq_dim=30,
                patch_size=self.cfg.patch_size,
                encoder_embed_dim=model_parameter['encoder_embed_dim'],
                encoder_heads=model_parameter['encoder_heads'],
                encoder_depths=model_parameter['encoder_depths'],
                decoder_embed_dim=model_parameter['decoder_embed_dim'],
                decoder_heads=model_parameter['decoder_heads'],
                decoder_depths=model_parameter['decoder_depths'],
                projection_hidden=model_parameter['projection_hidden'],
                recon_mode=model_parameter['recon_mode'],
                temperature=model_parameter['temperature'],
            )
            backbone.load_state_dict(self.ckpt['model_state'])

            backbone_final_length = backbone.autoencoder.embed_dim


        elif 'spectro' in self.cfg.model_name:
            print(model_parameter)
            pretrained_model = Spectro_NeuroNet(**model_parameter, freq_bins=30)
            pretrained_model.load_state_dict(self.ckpt['model_state'])

            backbone = NeuroNetSpectroEncoderWrapper(
                fs=model_parameter['fs'], second=model_parameter['second'],
                time_window=model_parameter['time_window'], time_step=model_parameter['time_step'],
                patch_embed=pretrained_model.autoencoder.patch_embed,
                encoder_block=pretrained_model.autoencoder.encoder_block,
                encoder_norm=pretrained_model.autoencoder.encoder_norm,
                cls_token=pretrained_model.autoencoder.cls_token,
                pos_embed=pretrained_model.autoencoder.pos_embed,
                final_length=pretrained_model.autoencoder.embed_dim
            )

            backbone_final_length = pretrained_model.autoencoder.embed_dim

        elif 'neuronet' in self.cfg.model_name:
            pretrained_model = NeuroNet(**model_parameter)
            pretrained_model.load_state_dict(self.ckpt['model_state'])

            backbone = NeuroNetEncoderWrapper(
                fs=model_parameter['fs'], second=model_parameter['second'],
                time_window=model_parameter['time_window'], time_step=model_parameter['time_step'],
                frame_backbone=pretrained_model.frame_backbone,
                patch_embed=pretrained_model.autoencoder.patch_embed,
                encoder_block=pretrained_model.autoencoder.encoder_block,
                encoder_norm=pretrained_model.autoencoder.encoder_norm,
                cls_token=pretrained_model.autoencoder.cls_token,
                pos_embed=pretrained_model.autoencoder.pos_embed,
                final_length=pretrained_model.autoencoder.embed_dim,
                num_channels=pretrained_model.num_channels,
            )

            backbone_final_length = pretrained_model.autoencoder.embed_dim

        else:
            backbone = CNNEncoder2D_SLEEP(self.ckpt['hyperparameter']['_content']['contra']['n_dim'])
            backbone.load_state_dict(self.ckpt['model_state'])

            backbone_final_length = backbone.n_dim

        # 3. Generator Classifier
        model = Classifier(backbone=backbone, backbone_final_length=backbone_final_length, model_name=self.cfg.model_name, representation=self.representation)
        return model

    def setup_dataloaders(self):
        val_dir = os.path.join(self.cfg.base_path, self.ft_split)
        test_dir = os.path.join(self.cfg.base_path, "test")

        val_index = os.listdir(val_dir)
        test_index = os.listdir(test_dir)

        if 'channels' in self.cfg.model_name:
            val_index = [f for f in val_index if "cassette2-" not in f]
            test_index = [f for f in test_index if "cassette2-" not in f]

            val_dataset = SLEEPCALoaderComb(val_index, val_dir)
            val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=self.cfg.batch_size, shuffle=True)

            test_dataset = SLEEPCALoaderComb(test_index, test_dir)
            test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=self.cfg.batch_size)

        else:
            val_dataset = SLEEPCALoader(val_index, val_dir, self.cfg.n_channels, False)
            val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=self.cfg.batch_size, shuffle=True)

            test_dataset = SLEEPCALoader(test_index, test_dir, self.cfg.n_channels, False)
            test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=self.cfg.batch_size)

        return val_dataloader, test_dataloader
    
@hydra.main(version_base=None, config_path="../conf", config_name="finetune_config")
def finetune(cfg) -> None:
    print(OmegaConf.to_yaml(cfg))

    print("Starting script...")
    trainer = Trainer(cfg)
    trainer.train()


if __name__ == '__main__':
    finetune()
