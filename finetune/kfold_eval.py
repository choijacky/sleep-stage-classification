# -*- coding:utf-8 -*-
import os
import sys
from datetime import datetime
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
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from backbones.neuronet import NeuroNet, NeuroNetEncoderWrapper, NeuroNetSpectroEncoderWrapper, Spectro_NeuroNet, Spectro_NeuroNet_ViT
from backbones.cnn_backbone import CNNEncoder2D_SLEEP
from data.data_loader import ISRUC, DOD, SLEEPCALoader_spectro
from torch.utils.tensorboard import SummaryWriter
import seaborn as sn
import pandas as pd
import re
from sklearn.model_selection import KFold

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
        self.feature_num = self.backbone_final_length // 4
        self.dropout_p = 0.5
        self.model_name = model_name
        self.representation = representation
        self.fc = nn.Sequential(
            nn.Linear(backbone_final_length, self.feature_num),
        )
        self.fc2 = nn.Sequential(
            nn.BatchNorm1d(self.feature_num),
            #nn.ELU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(self.feature_num, 5)
        )

    def forward(self, x):
        if isinstance(self.backbone, Spectro_NeuroNet_ViT):
            x = self.backbone.forward_latent(x)
        else:
            x = self.backbone(x)
        
            if 'neuronet' in self.model_name:
                if self.representation == "CLS":
                    x = x[:, :1 , :].squeeze() #take the CLS token
                else:
                    x = torch.mean(x[:, 1: , :], dim=1)
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
        weight = torch.tensor([7.0, 80.0, 2.0, 7.0, 5.0], device=device)
        self.criterion = nn.CrossEntropyLoss() #weight=weight)

        self.patch_size = cfg.patch_size

        now = datetime.now()

        self.tensorboard_path = os.path.join(self.cfg.ckpt_path, 'tensorboard', self.cfg.base_path.split("/")[-1], now.strftime("%d-%m-%Y %H:%M:%S"))
        self.tensorboard_writer = SummaryWriter(log_dir=self.tensorboard_path)

    def train(self, val_dataloader, test_dataloader):
        print('Checkpoint File Path : {}'.format(self.ckpt_path))

        # val_dataloader, test_dataloader = self.setup_dataloaders()

        best_model_state, best_mf1 = None, 0.0
        best_pred, best_real = [], []

        step = 0

        for epoch in range(self.cfg.epochs):
            self.model.train()
            epoch_train_loss = []
            train_real, train_pred = [], []
            for data in val_dataloader:
                self.optimizer.zero_grad()
                x, y1 = data
                x, y1 = x.to(device), y1.to(device)

                if not 'wavelet' in self.cfg.base_path:
                    x = x[:, 0, :].squeeze()

                pred = self.model(x.squeeze())

                loss = self.criterion(pred, y1)

                pred = pred.argmax(dim=-1)

                if self.cfg.soft_label == True:
                    y1 = y1.argmax(dim=-1)

                train_real.extend(list(y1.detach().cpu().numpy()))
                train_pred.extend(list(pred.detach().cpu().numpy()))
                epoch_train_loss.append(float(loss.detach().cpu().item()))
                loss.backward()
                self.optimizer.step()

            self.model.eval()
            epoch_test_loss = []
            epoch_real, epoch_pred = [], []
            for data in test_dataloader:
                with torch.no_grad():
                    x, y1 = data
                    x, y1 = x.to(device), y1.to(device)

                    if not 'wavelet' in self.cfg.base_path:
                        x = x[:, 0, :].squeeze()
                    pred = self.model(x.squeeze())
                    loss = self.criterion(pred, y1)
                    pred = pred.argmax(dim=-1)
                    real = y1

                    epoch_real.extend(list(real.detach().cpu().numpy()))
                    epoch_pred.extend(list(pred.detach().cpu().numpy()))
                    epoch_test_loss.append(float(loss.detach().cpu().item()))


            epoch_train_loss, epoch_test_loss = np.mean(epoch_train_loss), np.mean(epoch_test_loss)

            train_acc, train_mf1 = accuracy_score(y_true=train_real, y_pred=train_pred), \
                                 f1_score(y_true=train_real, y_pred=train_pred, average='macro')
            
            eval_acc, eval_mf1 = accuracy_score(y_true=epoch_real, y_pred=epoch_pred), \
                                 f1_score(y_true=epoch_real, y_pred=epoch_pred, average='macro')

            print('[Epoch] : {0:03d} \t '
                  '[Train Loss] => {1:.4f} \t '
                  '[Train Accuracy] => {3:.4f} \t'
                  '[Train Macro-F1] => {4:.4f} \t'
                  '[Evaluation Loss] => {2:.4f} \t '
                  '[Evaluation Accuracy] => {5:.4f} \t'
                  '[Evaluation Macro-F1] => {6:.4f}'.format(epoch + 1, epoch_train_loss, epoch_test_loss, train_acc, train_mf1,
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
                final_length=pretrained_model.autoencoder.embed_dim
            )

            backbone_final_length = pretrained_model.autoencoder.embed_dim

        else:
            backbone = CNNEncoder2D_SLEEP(self.ckpt['hyperparameter']['_content']['contra']['n_dim'])
            backbone.load_state_dict(self.ckpt['model_state'])

            backbone_final_length = backbone.n_dim

        # 3. Generator Classifier
        model = Classifier(backbone=backbone, backbone_final_length=backbone_final_length, model_name=self.cfg.model_name, representation=self.representation)
        return model

def setup_dataloaders(cfg):
    #cv = KFoldCrossSubject(n_splits=5, shuffle=True)
    train_dir = os.path.join(cfg.base_path, cfg.ft_split)
    test_dir = os.path.join(cfg.base_path, "test")

    train_index = os.listdir(train_dir)
    test_index = os.listdir(test_dir)

    if 'isruc' in cfg.base_path:
        train_dataset = ISRUC(train_index, train_dir, n_channels=7, multilabel=False)
        test_dataset = ISRUC(test_index, test_dir, n_channels=7, multilabel=False)
    else:
        train_dataset = DOD(train_index, train_dir, n_channels=12, multilabel=False, scorer=cfg.dod_scorer, softlabels=False)
        test_dataset = DOD(test_index, test_dir, n_channels=12, multilabel=False, scorer=cfg.dod_scorer, softlabels=False, consensus=False)
        print("Scorer = ", cfg.dod_scorer)

    #val_dataset, test_dataset = train_test_split_cross_subject(dataset=dataset)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.batch_size)

    return train_dataloader, test_dataloader
    
@hydra.main(version_base=None, config_path="../conf", config_name="finetune_config")
def finetune(cfg) -> None:
    print(OmegaConf.to_yaml(cfg))

    k_folds = 5
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    train_dir = os.path.join(cfg.base_path, cfg.ft_split)
    if cfg.test_dataset is not None:
        test_dir = os.path.join(os.path.dirname(os.path.dirname(cfg.base_path)), cfg.test_dataset, cfg.ft_split)
        test_index = os.listdir(test_dir)

    train_index = os.listdir(train_dir)
    

    patient_ids = [
        '119f9726-eb4c-5a0e-a7bb-9e15256149a1',
        '37d0da97-9ae8-5413-b889-4e843ff35488',
        '844f68ba-265e-53e6-bf47-6c85d1804a7b',
        'b5d5785d-87ee-5078-b9b9-aac6abd4d8de',
        '095d6e40-5f19-55b6-a0ec-6e0ad3793da0',
        'f2a69bdc-ed51-5e3f-b102-6b3f7d392be0',
        '64959ac4-53b5-5868-a845-c7476e9fdf7b',
        '25a6b2b0-4d09-561b-82c6-f09bb271d3be',
        '0d79f4b1-e74f-5e87-8e42-f9dd7112ada5',
        '18ede714-aba3-5ad8-bb1a-18fc9b1c4192',
        '3e842aa8-bcd9-521e-93a2-72124233fe2c', '67fa8e29-6f4d-530e-9422-bbc3aca86ed0', 'aa160c78-6da3-5e05-8fc9-d6c13e9f97e0', 'a25b2296-343b-53f6-8792-ada2669d466e', '1fa6c401-d819-50f5-8146-a0bb9e2b2516', '769df255-2284-50b3-8917-2155c759fbbd', '7d778801-88e7-5086-ad1d-70f31a371876', 'a4568951-bf87-5bbc-bc4f-28e93c360be6', 'bb474ab0-c2ce-573b-8acd-ef86b0fa26a2', '5bf0f969-304c-581e-949c-50c108f62846', '1da3544e-dc5c-5795-adc3-f5068959211f', 'a30245e3-4a71-565f-9636-92e7d2e825fc', 'd3cadb78-cb8c-5a6e-885c-392e457c68b1', '7ab8ff5f-a77f-567d-9882-f8bee0c3c9bf', '14c012bd-65b0-56f5-bc74-2dffcea69837'
    ]
    patient_ids = np.array(patient_ids)

    for fold, (train_idx, test_idx) in enumerate(kfold.split(np.arange(len(patient_ids)))):
        train_ids = patient_ids[train_idx]
        test_ids = patient_ids[test_idx]

        print(test_ids)

        if cfg.test_dataset is not None:
            test_files = [file for file in test_index if any(id in file for id in test_ids)]
            test_dataset = DOD(test_files, test_dir, n_channels=12, multilabel=False, scorer=cfg.dod_scorer, softlabels=False, consensus=False)
        else:
            if 'wavelet' in cfg.base_path:
                test_files = [file for file in train_index if any(id in file for id in test_ids)]
                test_dataset = SLEEPCALoader_spectro(test_files, train_dir, majority=True)
            else:
                test_files = [file for file in train_index if any(id in file for id in test_ids)]
                test_dataset = DOD(test_files, train_dir, n_channels=12, multilabel=False, scorer=cfg.dod_scorer, softlabels=False, consensus=True)
        
        train_files = [file for file in train_index if any(id in file for id in train_ids)]

        if 'wavelet' in cfg.base_path:
            train_dataset = SLEEPCALoader_spectro(test_files, train_dir, majority=False)
        else:
            train_dataset = DOD(train_files, train_dir, n_channels=12, multilabel=False, scorer=cfg.dod_scorer, softlabels=cfg.soft_label, consensus=False)

        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.batch_size)

        print("Starting script...")
        trainer = Trainer(cfg)

        trainer.train(train_dataloader, test_dataloader)


if __name__ == '__main__':
    finetune()
