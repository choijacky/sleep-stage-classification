import os
import random
import time
from typing import Optional
import sys

sys.path.append('data')
print(sys.path)

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from data_loader import SLEEPCALoader



class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        masking,
        pca_paths,
        data_dir: str = "",
        batch_size: int = 512,
        num_workers: int = 8,
        classes: int = 5,
        n_channels: int = 1,
    ):
        super().__init__()
        self.masking = masking
        self.data_dir = data_dir
        self.pca_paths = pca_paths
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_classes = classes
        self.n_channels = n_channels
        

    def setup(self, stage):
        train_dir = os.path.join(self.data_dir, "train")
        val_dir = os.path.join(self.data_dir, "val")
        test_dir = os.path.join(self.data_dir, "test")

        train_index = os.listdir(train_dir)
        val_index = os.listdir(val_dir)
        test_index = os.listdir(test_dir)

        self.train_dataset = SLEEPCALoader(train_index, train_dir, self.n_channels, True)
        self.val_dataset = SLEEPCALoader(val_index, train_dir, self.n_channels, False)
        self.test_dataset = SLEEPCALoader(test_index, train_dir, self.n_channels, False)

        if self.masking.mask == "pca":
            evectors = np.load(os.path.join(self.pca_paths.matrix))
            evalues = np.load(os.path.join(self.pca_paths.ev_ratios))

            self.evectors = torch.FloatTensor(evectors)
            self.evalues = torch.FloatTensor(evalues)


    def collate_fn(self, batch):
        """
        Custom collate function to handle variable-sized pc_mask.
        Pads the pc_mask to the size of the largest pc_mask in the batch.
        """
        x, y = zip(*batch)

        x = torch.stack(x)
        y = torch.Tensor(x)

        threshold = 1 - self.masking.mask_ratio

        find_threshold = lambda eigenvalues, ratio : torch.argmin(torch.abs(torch.cumsum(eigenvalues, dim=0) - ratio))

        index = torch.randperm(self.evectors.shape[0])
        #index = np.random.permutation(evecs.shape[0])

        threshold_idx = find_threshold(self.evalues[index], threshold)

        pc_mask = index[:threshold_idx]
        masked_pc = self.evectors[pc_mask]
        inverse_masked_pc = self.evectors[index[threshold_idx:]]

        return x, y, masked_pc, inverse_masked_pc


    def train_dataloader(self) -> DataLoader:
        if self.masking.mask == 'token':
            train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=False, num_workers=self.num_workers)

        elif self.masking.mask == 'pca':
            train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=False, num_workers=self.num_workers, collate_fn=self.collate_fn)
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=self.num_workers
        )

        test_loader = DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=self.num_workers
        )
        return [val_loader, test_loader]