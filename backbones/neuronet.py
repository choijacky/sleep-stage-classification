# -*- coding:utf-8 -*-
import sys

sys.path.extend(["/cluster/project/jbuhmann/choij/sleep-stage-classification/backbones"])

import torch
import numpy as np
import torch.nn as nn
from typing import List, Optional
from extractors.resnet1d import FrameBackBone
from timm.models.vision_transformer import Block
from model_utils import get_2d_sincos_pos_embed_flexible
from loss import NTXentLoss
from functools import partial
import time
import matplotlib.pyplot as plt


class NeuroNet(nn.Module):
    def __init__(self, fs: int, second: int, time_window: float, time_step: float,
                 encoder_embed_dim, encoder_heads: int, encoder_depths: int,
                 decoder_embed_dim: int, decoder_heads: int, decoder_depths: int,
                 projection_hidden: List, freq_bins: int = 0, recon_mode: str="masked_tokens", temperature=0.01, num_channels=1):
        super().__init__()
        self.fs, self.second = fs, second
        self.time_window = time_window
        self.time_step = time_step

        self.num_patches, _ = frame_size(fs=fs, second=second, time_window=time_window, time_step=time_step)
        self.frame_backbone = FrameBackBone(fs=self.fs, window=self.time_window, recon_mode=recon_mode, num_channels=num_channels)
        self.num_channels = num_channels
        input_size = self.frame_backbone.feature_num # * num_channels
        self.autoencoder = MaskedAutoEncoderViT(input_size=input_size,
                                                encoder_embed_dim=encoder_embed_dim, num_patches=self.num_patches,
                                                encoder_heads=encoder_heads, encoder_depths=encoder_depths,
                                                decoder_embed_dim=decoder_embed_dim, decoder_heads=decoder_heads,
                                                decoder_depths=decoder_depths,
                                                window_size=time_window,
                                                fs=fs,
                                                recon_mode=recon_mode,
                                                )
        self.contrastive_loss = NTXentLoss(temperature=temperature)

        projection_hidden = [encoder_embed_dim] + projection_hidden
        projectors = []
        for i, (h1, h2) in enumerate(zip(projection_hidden[:-1], projection_hidden[1:])):
            if i != len(projection_hidden) - 2:
                projectors.append(nn.Linear(h1, h2))
                projectors.append(nn.BatchNorm1d(h2))
                projectors.append(nn.ELU())
            else:
                projectors.append(nn.Linear(h1, h2))
        self.projectors = nn.Sequential(*projectors)
        self.projectors_bn = nn.BatchNorm1d(projection_hidden[-1], affine=False)
        self.norm_pix_loss = False
        self.recon_mode = recon_mode
        self.freq_bins = freq_bins

    def forward(self, x: torch.Tensor, mask_ratio: float = 0.5, sample_ratio=None) -> (torch.Tensor, torch.Tensor):
        if sample_ratio == True:
            choices = torch.arange(0.1, 1.0, 0.1)
            mask_ratio = choices[torch.randint(0, 9, (1,))].item()

        if self.recon_mode != "masked_tokens":
            frames = self.make_frame(x)
            tokens = self.frame_backbone(frames)
            latent1, pred1, mask1 = self.autoencoder(tokens, mask_ratio)

            latent2, pred2, mask2 = self.autoencoder(tokens, mask_ratio)

        elif self.num_channels != 1:
            frames = []
            for i in range(self.num_channels):
                frames.append(self.make_frame(x[:, i, :]))

            x = torch.stack(frames, dim=1)

            x = self.frame_backbone(x)

            
            """B, num_channels, len_signal = x.shape
            x = x.reshape(B * num_channels, len_signal)
            x = self.make_frame(x) # (B * num_channels, num_tokens, fs * second)
            num_tokens = x.shape[2]
            x = x.reshape(B, num_channels, num_tokens, self.fs * self.second)
            x = torch.einsum("bntd->btnd", x)
            x = x.reshape(B, num_tokens, num_channels * self.fs * self.second)"""

            latent1, pred1, mask1 = self.autoencoder(x, mask_ratio)
            
            latent2, pred2, mask2 = self.autoencoder(x, mask_ratio)

        else:
            x = self.make_frame(x)
            x = self.frame_backbone(x)
            latent1, pred1, mask1 = self.autoencoder(x, mask_ratio)

            latent2, pred2, mask2 = self.autoencoder(x, mask_ratio)

        o1, o2 = latent1[:, :1, :].squeeze(), latent2[:, :1, :].squeeze()

        recon_loss1 = self.forward_mae_loss(x, pred1, mask1)

        recon_loss2 = self.forward_mae_loss(x, pred2, mask2)

        recon_loss = recon_loss1 + recon_loss2

        o1, o2 = self.projectors(o1), self.projectors(o2)

        contrastive_loss, (labels, logits) = self.contrastive_loss(o1, o2)

        return recon_loss, contrastive_loss, (labels, logits)

    def forward_latent(self, x: torch.Tensor):
        if self.num_channels != 1:
            frames = []
            for i in range(self.num_channels):
                frames.append(self.make_frame(x[:, i, :]))

            x = torch.stack(frames, dim=1)

            x = self.frame_backbone(x)

        else:
            x = self.make_frame(x)
            x = self.frame_backbone(x)

        latent = self.autoencoder.forward_encoder(x, mask_ratio=0)[0]
        latent_o = latent[:, :1, :].squeeze()
        return latent_o
    
    def forward_reconstruction(self, x: torch.Tensor):
        frames = self.make_frame(x)
        tokens = self.frame_backbone(frames)
        _, pred, mask = self.autoencoder(tokens, mask_ratio=0.5)
        return pred, mask

    def forward_mae_loss(self,
                         real: torch.Tensor,
                         pred: torch.Tensor,
                         mask: torch.Tensor):

        if self.norm_pix_loss:
            mean = real.mean(dim=-1, keepdim=True)
            var = real.var(dim=-1, keepdim=True)
            real = (real - mean) / (var + 1.e-6) ** .5

        loss = (pred - real) ** 2

        if self.recon_mode == "masked_time_patches":
            mask = torch.repeat_interleave(mask, int(self.time_window * self.fs), dim=1)
            loss = (loss * mask).sum() / mask.sum()

        elif self.recon_mode == "time_signal":
            b = real.shape[0]
            loss = loss.mean(dim=-1).reshape(b, -1)
            loss = loss.sum() / (loss.shape[1] * b)

        elif self.recon_mode == "masked_tokens":
            loss = loss.mean(dim=-1)

            if mask.isnan().any().item() == False:
                loss = (loss * mask).sum() / mask.sum()

        else:
            raise Exception("No such reconstruction mode")
        
        return loss

    def make_frame(self, x):
        size = self.fs * self.second
        step = int(self.time_step * self.fs)
        window = int(self.time_window * self.fs)
        frame = []
        for i in range(0, size, step):
            start_idx, end_idx = i, i+window
            sample = x[..., start_idx: end_idx]
            if sample.shape[-1] == window:
                frame.append(sample)
        frame = torch.stack(frame, dim=1)
        return frame
    



class MaskedAutoEncoderViT(nn.Module):
    def __init__(self, input_size: int, num_patches: int,
                 encoder_embed_dim: int, encoder_heads: int, encoder_depths: int,
                 decoder_embed_dim: int, decoder_heads: int, decoder_depths: int,
                 window_size: float, fs: int, recon_mode: str):
        super().__init__()
        self.patch_embed = nn.Linear(input_size, encoder_embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, encoder_embed_dim))
        self.embed_dim = encoder_embed_dim
        self.encoder_depths = encoder_depths
        self.mlp_ratio = 4.
        self.recon_mode = recon_mode

        self.input_size = (num_patches, encoder_embed_dim)
        self.patch_size = (1, encoder_embed_dim)
        self.grid_h = int(self.input_size[0] // self.patch_size[0])
        self.grid_w = int(self.input_size[1] // self.patch_size[1])
        self.num_patches = self.grid_h * self.grid_w

        # MAE Encoder
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, encoder_embed_dim), requires_grad=False)
        self.encoder_block = nn.ModuleList([
            Block(encoder_embed_dim, encoder_heads, self.mlp_ratio, qkv_bias=True,
                  norm_layer=partial(nn.LayerNorm, eps=1e-6))
            for _ in range(encoder_depths)
        ])
        self.encoder_norm = nn.LayerNorm(encoder_embed_dim, eps=1e-6)

        # MAE Decoder
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.randn(1, self.num_patches, decoder_embed_dim), requires_grad=False)
        self.decoder_block = nn.ModuleList([
            Block(decoder_embed_dim, decoder_heads, self.mlp_ratio, qkv_bias=True,
                  norm_layer=partial(nn.LayerNorm, eps=1e-6))
            for _ in range(decoder_depths)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim, eps=1e-6)
        if recon_mode == "masked_tokens":
            self.decoder_pred = nn.Linear(decoder_embed_dim, input_size, bias=True)
        else:
            self.decoder_pred = nn.Sequential(
                nn.Linear(decoder_embed_dim, int(window_size * fs * 2)),
                nn.Linear(int(window_size * fs * 2), int(window_size * fs)),
            )
        self.initialize_weights()

    def forward(self, x, mask_ratio=0.8, sample_ratio=False, freq_mask_ratio=None):
        if sample_ratio == True:
            choices = torch.arange(0.1, 1.0, 0.1)
            mask_ratio = choices[torch.randint(0, 9, (1,))].item()
        if freq_mask_ratio is None:
            latent, mask, ids_restore = self.forward_encoder(x, mask_ratio)
        else:
            latent, mask, ids_restore = self.forward_encoder(x, mask_ratio, freq_mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        return latent, pred, mask

    def forward_encoder(self, x: torch.Tensor, mask_ratio: float = 0.5, freq_mask_ratio=None):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        if freq_mask_ratio is None:
            x, mask, ids_restore = self.random_masking(x, mask_ratio)
        else:
            x, mask, ids_restore = self.stripe_masking_dual(x, freq_mask_ratio, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for block in self.encoder_block:
            x = block(x)

        x = self.encoder_norm(x)
        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore: torch.Tensor):
        # embed tokens
        x = self.decoder_embed(x[:, 1:, :])

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=1)  # no cls token
        x = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for block in self.decoder_block:
            x = block(x)

        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        if self.recon_mode != "masked_tokens":
            x = torch.flatten(x, start_dim=1)
        return x

    @staticmethod
    def random_masking(x, mask_ratio):
        n, l, d = x.shape  # batch, length, dim
        len_keep = int(l * (1 - mask_ratio))

        noise = torch.rand(n, l, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, d))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([n, l], device=x.device)
        mask[:, :len_keep] = 0

        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore
    
    @staticmethod
    def stripe_masking_dual(x, mask_ratio_row, mask_ratio_col, img_size=(30, 60), patch_size=6):
        """
        Applies efficient row-wise and column-wise stripe masking.
        """
        n, l, d = x.shape
        H, W = img_size
        ph = pw = patch_size

        num_patches_H = H // ph  # e.g. 10
        num_patches_W = W // pw  # e.g. 20

        assert l == num_patches_H * num_patches_W, "Patch count mismatch"

        device = x.device

        # Generate row and column indices
        row_indices = torch.randperm(num_patches_H, device=device)[:int(num_patches_H * mask_ratio_row)]
        col_indices = torch.randperm(num_patches_W, device=device)[:int(num_patches_W * mask_ratio_col)]

        # Create base mask: 1 = masked, 0 = keep
        mask_2d = torch.zeros((num_patches_H, num_patches_W), device=device)
        mask_2d[row_indices, :] = 1
        mask_2d[:, col_indices] = 1
        base_mask = mask_2d.flatten()  # shape (num_patches,)

        # Expand to batch
        mask = base_mask.unsqueeze(0).expand(n, -1)  # (B, num_patches)

        # Compute ids_keep using sort-based trick (vectorized)
        sort_order = torch.argsort(mask, dim=1, descending=False)  # 0s first
        num_keep = (mask == 0).sum(dim=1).unsqueeze(-1)  # (B, 1)

        batch_indices = torch.arange(n, device=device).unsqueeze(1)
        ids_keep = sort_order[:, :num_keep.max()]  # pad to max for batching

        # Handle uneven keep lengths by masking out excess with large index
        pad_mask = torch.arange(ids_keep.shape[1], device=device).unsqueeze(0) >= num_keep
        ids_keep[pad_mask] = 0  # will be ignored in gather due to padding

        # Restore order for decoder
        ids_restore = torch.argsort(sort_order, dim=1)

        # Gather visible patches
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, d))

        return x_masked, mask, ids_restore

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        try:
            pos_embed = get_2d_sincos_pos_embed_flexible(self.pos_embed.shape[-1],
                                                        (self.grid_h, self.grid_w),
                                                        cls_token=True)
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
            plt.imsave('/cluster/project/jbuhmann/choij/sleep-stage-classification/test.png', self.pos_embed.squeeze().detach().cpu().numpy())
            decoder_pos_embed = get_2d_sincos_pos_embed_flexible(self.decoder_pos_embed.shape[-1],
                                                                (self.grid_h, self.grid_w),
                                                                cls_token=False)
            self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

            # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
            torch.nn.init.normal_(self.cls_token, std=.02)
            torch.nn.init.normal_(self.mask_token, std=.02)

            # initialize nn.Linear and nn.LayerNorm
            self.apply(self._init_weights)
            print("Positional encoding done")
        except:
            pass

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


def frame_size(fs, second, time_window, time_step, num_channels=1):
    x = np.random.randn(1, fs * second)
    size = fs * second
    step = int(time_step * fs)
    window = int(time_window * fs)
    frame = []
    for i in range(0, size, step):
        start_idx, end_idx = i, i + window
        sample = x[..., start_idx: end_idx]
        if sample.shape[-1] == window:
            frame.append(sample)
    frame = np.stack(frame, axis=1)
    return frame.shape[1], frame.shape[2]

class PCA_NeuroNet(NeuroNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward(self, x1: torch.Tensor, evecs: torch.Tensor, evals: torch.Tensor, mask_ratio: float, asym: bool = False, sample_ratio = False) -> (torch.Tensor, torch.Tensor):
        device = x1.device

        x1, x2 = self.get_reconstructions(x1, evecs, evals, mask_ratio, sample_ratio)

        x1 = x1.to(device)
        x2 = x2.to(device)
        if asym == False:
            x1 = self.make_frame(x1)
            x1 = self.frame_backbone(x1)
            
            x2 = self.make_frame(x2)
            x2 = self.frame_backbone(x2)

            latent1, pred1, mask1 = self.autoencoder(x1, mask_ratio=0)
            latent2, pred2, mask2 = self.autoencoder(x2, mask_ratio=0)

            o1, o2 = latent1[:, :1, :].squeeze(), latent2[:, :1, :].squeeze()

            # Predict the reconstruction from inverse PCA mask
            recon_loss1 = self.forward_mae_loss(x2, pred1, mask1)
            recon_loss2 = self.forward_mae_loss(x1, pred2, mask2)

            recon_loss = recon_loss1 + recon_loss2

            # Contrastive Learning
            o1, o2 = self.projectors(o1), self.projectors(o2)
            contrastive_loss, (labels, logits) = self.contrastive_loss(o1, o2)
            
            return recon_loss, contrastive_loss, (labels, logits)

        else:
            x1 = self.make_frame(x1)
            x1 = self.frame_backbone(x1)
            
            x2 = self.make_frame(x2)
            x2 = self.frame_backbone(x2)

            latent1, pred1, mask1 = self.autoencoder(x1, mask_ratio=0)
            latent2, _, _ = self.autoencoder(x2, mask_ratio=0)

            o1, o2 = latent1[:, :1, :].squeeze(), latent2[:, :1, :].squeeze()

            # Predict the reconstruction from inverse PCA mask
            recon_loss1 = self.forward_mae_loss(x2, pred1, mask1)

            # Contrastive Learning
            o1, o2 = self.projectors(o1), self.projectors(o2)
            contrastive_loss, (labels, logits) = self.contrastive_loss(o1, o2)
            
            return recon_loss1, contrastive_loss, (labels, logits)

    def forward_mae_loss(self,
                         real: torch.Tensor,
                         pred: torch.Tensor,
                         mask: torch.Tensor):

        b = pred.shape[0]
        loss = (pred - real) ** 2
        loss = loss.mean(dim=-1).reshape(b, -1)
        loss = loss.sum() / (loss.shape[1] * b)
        return loss
    

    def get_reconstructions(self, x1, evecs, evals, mask_ratio, sample_ratio = False):
        if sample_ratio == True:
            choices = torch.arange(0.1, 1.0, 0.1)
            mask_ratio = choices[torch.randint(0, 9, (1,))].item()

        threshold = 1 - mask_ratio

        find_threshold = lambda eigenvalues, ratio : torch.argmin(torch.abs(torch.cumsum(eigenvalues, dim=0) - ratio))

        index = torch.randperm(evecs.shape[0])
        threshold_idx = find_threshold(evals[index], threshold)

        pc_mask = index[:threshold_idx]
        masked_pc = evecs[pc_mask]

        index2 = torch.randperm(evecs.shape[0])
        threshold_idx2 = find_threshold(evals[index2], threshold)

        inverse_masked_pc = evecs[index[:threshold_idx2]]

        reduced_X1 = torch.einsum('bm,pm->bp', x1, masked_pc)
        reduced_X2 = torch.einsum('bm,pm->bp', x1, inverse_masked_pc)

        X1 = torch.einsum('bp,pm->bm', reduced_X1, masked_pc)
        X2 = torch.einsum('bp,pm->bm', reduced_X2, inverse_masked_pc)

        return X1, X2


class Spectro_NeuroNet(NeuroNet):
    def __init__(self, fs: int, second: int, time_window: float, time_step: float,
                 encoder_embed_dim, encoder_heads: int, encoder_depths: int,
                 decoder_embed_dim: int, decoder_heads: int, decoder_depths: int,
                 projection_hidden: List, freq_bins: int = 20, recon_mode: str="masked_tokens", temperature=0.01):
        super().__init__(fs=fs, second=second, time_window=time_window, time_step=time_step,
                         encoder_embed_dim=encoder_embed_dim, encoder_heads=encoder_heads, encoder_depths=encoder_depths,
                         decoder_embed_dim=decoder_embed_dim, decoder_heads=decoder_heads, decoder_depths=decoder_depths,
                         projection_hidden=projection_hidden, recon_mode=recon_mode, temperature=temperature)

        self.freq_bins = freq_bins
        self.num_patches = int(1800/self.freq_bins)
        self.autoencoder = MaskedAutoEncoderViT(
            input_size=self.freq_bins,
            encoder_embed_dim=encoder_embed_dim,
            num_patches=self.num_patches,
            encoder_heads=encoder_heads,
            encoder_depths=encoder_depths,
            decoder_embed_dim=decoder_embed_dim,
            decoder_heads=decoder_heads,
            decoder_depths=decoder_depths,
            window_size=self.time_window,
            fs=self.fs,
            recon_mode=self.recon_mode,
        )
        
        del self.frame_backbone

    
    def forward(self, x: torch.Tensor, mask_ratio: float, freq_mask_ratio: float) -> (torch.Tensor, torch.Tensor):
        # Input (B, num_freq_bins, time_frames)
        # freq mask ratio
        # Needed masking for f and t domain

        # Optional: 

        # Linear transform + Positional

        if False:
            time_mask = self.choose_samples(x.shape[1], mask_ratio)
            freq_mask = self.choose_samples(x.shape[2], freq_mask_ratio)

            x, mask = self.apply_mask(x, time_mask, freq_mask)

            latent1, pred1, _ = self.autoencoder(x, mask_ratio=0.0)

            recon_loss = self.forward_mae_loss(x, pred1, mask)

            return recon_loss#, contrastive_loss, (labels, logits)

        else:
            #time_mask1 = self.choose_samples(x.shape[1], mask_ratio)
            freq_mask1 = self.choose_samples(x.shape[2], freq_mask_ratio)

            #time_mask2 = self.choose_samples(x.shape[1], mask_ratio)
            freq_mask2 = self.choose_samples(x.shape[2], freq_mask_ratio)

            x1, mask1 = self.apply_mask(x, None, freq_mask1)
            x2, mask2 = self.apply_mask(x, None, freq_mask2)

            latent1, pred1, time_mask1 = self.autoencoder(x1, mask_ratio)
            latent2, pred2, time_mask2 = self.autoencoder(x2, mask_ratio)

            o1, o2 = latent1[:, :1, :].squeeze(), latent2[:, :1, :].squeeze()
            o1, o2 = self.projectors(o1), self.projectors(o2)

            mask1 = torch.logical_or(time_mask1.unsqueeze(-1).repeat(1, 1, x1.shape[2]), mask1, out=torch.empty(mask1.shape, dtype=torch.int, device=x.device))
            mask2 = torch.logical_or(time_mask2.unsqueeze(-1).repeat(1, 1, x2.shape[2]), mask2, out=torch.empty(mask2.shape, dtype=torch.int, device=x.device))

            recon_loss1 = self.forward_mae_loss(x, pred1, mask1)
            recon_loss2 = self.forward_mae_loss(x, pred2, mask2)
            recon_loss = recon_loss1 + recon_loss2

            contrastive_loss, (labels, logits) = self.contrastive_loss(o1, o2)
            
            return recon_loss, contrastive_loss, (labels, logits)

    
    def choose_samples(self, n, ratio):
        k = int(n * ratio)
        if k == 0:
            return None
        perm = torch.randperm(n)
        idx = perm[:k]
        return idx

    def apply_mask(self, x, time_mask, freq_mask):
        mask = torch.zeros(x.shape, device=x.device)

        if time_mask is not None:
            mask[:, time_mask, :] = 1

        if freq_mask is not None:
            mask[:, :, freq_mask] = 1

        return x * mask, mask.to(x.device)
    
    def forward_mae_loss(self,
                         real: torch.Tensor,
                         pred: torch.Tensor,
                         mask: torch.Tensor):

        loss = (pred - real) ** 2
        loss = (loss * mask).sum()# / mask.sum()
        return loss
        
    
    def forward_latent(self, x: torch.Tensor):
        latent = self.autoencoder.forward_encoder(x, mask_ratio=0)[0]
        latent_o = latent[:, :1, :].squeeze()
        return latent_o

    def forward_spec(self, x: torch.Tensor, mask_ratio: float, freq_mask_ratio: float):
        freq_mask = self.choose_samples(x.shape[2], freq_mask_ratio)
        x_masked, freq_mask = self.apply_mask(x, None, freq_mask)

        pred, time_mask = self.autoencoder(x_masked, mask_ratio)[1:]
        
        mask = torch.logical_or(freq_mask, time_mask.unsqueeze(-1).repeat(1, 1, x.shape[2]), out=torch.empty(freq_mask.shape, dtype=torch.int, device=x.device))

        masked_original = x * mask
        return masked_original, pred
    


class Spectro_NeuroNet_ViT(nn.Module):
    def __init__(self, time_dim, freq_dim, patch_size,
                 encoder_embed_dim, encoder_heads: int, encoder_depths: int,
                 decoder_embed_dim: int, decoder_heads: int, decoder_depths: int,
                 projection_hidden: List, recon_mode: str="masked_tokens", temperature=0.01):
        super().__init__()
        
        self.time_dim = time_dim
        self.freq_dim = freq_dim
        self.patch_size = patch_size
        self.num_patches = (self.time_dim // self.patch_size) * (self.freq_dim // self.patch_size)

        self.autoencoder = MaskedAutoEncoderViT(input_size=self.patch_size ** 2,
                                                encoder_embed_dim=encoder_embed_dim, num_patches=self.num_patches,
                                                encoder_heads=encoder_heads, encoder_depths=encoder_depths,
                                                decoder_embed_dim=decoder_embed_dim, decoder_heads=decoder_heads,
                                                decoder_depths=decoder_depths,
                                                window_size=1,
                                                fs=1,
                                                recon_mode=recon_mode,
                                                )

        # overwrite grid_h and grid_w
        self.autoencoder.grid_w = self.time_dim // self.patch_size
        self.autoencoder.grid_h = self.freq_dim // self.patch_size

        self.autoencoder.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, encoder_embed_dim), requires_grad=False)
        self.autoencoder.decoder_pos_embed = nn.Parameter(torch.randn(1, self.num_patches, decoder_embed_dim), requires_grad=False)
        # reinitialize weights again
        self.autoencoder.initialize_weights()

        self.contrastive_loss = NTXentLoss(temperature=temperature)

        projection_hidden = [encoder_embed_dim] + projection_hidden
        projectors = []
        for i, (h1, h2) in enumerate(zip(projection_hidden[:-1], projection_hidden[1:])):
            if i != len(projection_hidden) - 2:
                projectors.append(nn.Linear(h1, h2))
                projectors.append(nn.BatchNorm1d(h2))
                projectors.append(nn.ELU())
            else:
                projectors.append(nn.Linear(h1, h2))
        self.projectors = nn.Sequential(*projectors)
        self.norm_pix_loss = False
        self.recon_mode = recon_mode


    def forward(self, x: torch.Tensor, mask_ratio: float = 0.5, freq_mask_ratio=None):
        # patchify the spectrogram
        #original_shape = x.shape[1:]

        x = self.patchify(x) # (B, num_patches, patch_size ** 2)
        
        # run forward autoencoder
        if freq_mask_ratio is None:
            latent1, pred1, mask1 = self.autoencoder(x, mask_ratio) # prediction of shape (B, num_patches, p ** 2), mask (1, num_patches)
            latent2, pred2, mask2 = self.autoencoder(x, mask_ratio)

        else:
            latent1, pred1, mask1 = self.autoencoder(x, mask_ratio, freq_mask_ratio=freq_mask_ratio) # prediction of shape (B, num_patches, p ** 2), mask (1, num_patches)
            latent2, pred2, mask2 = self.autoencoder(x, mask_ratio, freq_mask_ratio=freq_mask_ratio)

        o1, o2 = latent1[:, :1, :].squeeze(), latent2[:, :1, :].squeeze()

        recon_loss1 = self.forward_mae_loss(x, pred1, mask1)

        recon_loss2 = self.forward_mae_loss(x, pred2, mask2)

        recon_loss = recon_loss1 + recon_loss2

        o1, o2 = self.projectors(o1), self.projectors(o2)

        contrastive_loss, (labels, logits) = self.contrastive_loss(o1, o2)

        return recon_loss, contrastive_loss, (labels, logits)

    def forward_latent(self, x: torch.Tensor):
        x = self.patchify(x)
        latent = self.autoencoder.forward_encoder(x, mask_ratio=0)[0]
        latent_o = latent[:, :1, :].squeeze()
        return latent_o
    
    def forward_spec(self, x: torch.Tensor, mask_ratio: float, freq_mask_ratio=None):
        original_shape = x.shape[1:]

        x1 = self.patchify(x)

        if freq_mask_ratio is None:
            pred, mask = self.autoencoder(x1, mask_ratio)[1:] # get pred as patches

        else:
            pred, mask = self.autoencoder(x1, mask_ratio, freq_mask_ratio=freq_mask_ratio)[1:] # get pred as patches
            
        repeated_mask = mask.unsqueeze(-1).repeat(1, 1, x1.shape[2]) # repeat mask in last dim

        repeated_mask = 1 - repeated_mask

        masked_original = x1 * repeated_mask # apply mask

        masked_original = self.unpatchify(masked_original, original_shape)
        
        rec = self.unpatchify(pred, original_shape)
        return masked_original, rec
        


    def forward_mae_loss(self,
                         real: torch.Tensor,
                         pred: torch.Tensor,
                         mask: torch.Tensor):

        if self.norm_pix_loss:
            mean = real.mean(dim=-1, keepdim=True)
            var = real.var(dim=-1, keepdim=True)
            real = (real - mean) / (var + 1.e-6) ** .5

        loss = (pred - real) ** 2 # (B, num_patches, p**2)

        loss = loss.mean(dim=-1) # (B, num_patches)

        if mask.isnan().any().item() == False:
            loss = (loss * mask).sum() / mask.sum() #averaging
    
        return loss

    def patchify(self, pixel_values):
        """
        From transformers - huggingface

        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, height, width)`):
                Pixel values.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, num_patches, patch_size**2)`:
                Patchified pixel values.
        """

        # patchify
        batch_size = pixel_values.shape[0]
        num_patches_h = pixel_values.shape[1] // self.patch_size
        num_patches_w = pixel_values.shape[2] // self.patch_size

        patchified_pixel_values = pixel_values.reshape(
            batch_size, num_patches_h, self.patch_size, num_patches_w, self.patch_size
        )
        patchified_pixel_values = torch.einsum("nhpwq->nhwpq", patchified_pixel_values)
        patchified_pixel_values = patchified_pixel_values.reshape(
            batch_size, num_patches_h * num_patches_w, self.patch_size**2
        )
        return patchified_pixel_values
    
    def unpatchify(self, patchified_pixel_values, original_image_size):
        """
        From transformers - huggingface

        Args:
            patchified_pixel_values (`torch.FloatTensor` of shape `(batch_size, num_patches, patch_size**2)`:
                Patchified pixel values.
            original_image_size (`Tuple[int, int]`):
                Original image size.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, height, width)`:
                Pixel values.
        """
        original_height, original_width = original_image_size
        num_patches_h = original_height // self.patch_size
        num_patches_w = original_width // self.patch_size

        # unpatchify
        batch_size = patchified_pixel_values.shape[0]
        patchified_pixel_values = patchified_pixel_values.reshape(
            batch_size,
            num_patches_h,
            num_patches_w,
            self.patch_size,
            self.patch_size,
        )
        patchified_pixel_values = torch.einsum("nhwpq->nhpwq", patchified_pixel_values)
        pixel_values = patchified_pixel_values.reshape(
            batch_size,
            num_patches_h * self.patch_size,
            num_patches_w * self.patch_size,
        )
        return pixel_values



class NeuroNetEncoderWrapper(nn.Module):
    def __init__(self, fs: int, second: int, time_window: int, time_step: float,
                 frame_backbone, patch_embed, encoder_block, encoder_norm, cls_token, pos_embed,
                 final_length, num_channels=1):

        super().__init__()
        self.fs, self.second = fs, second
        self.time_window = time_window
        self.time_step = time_step

        self.patch_embed = patch_embed
        self.frame_backbone = frame_backbone
        self.encoder_block = encoder_block
        self.encoder_norm = encoder_norm
        self.cls_token = cls_token
        self.pos_embed = pos_embed

        self.final_length = final_length

        self.num_channels = num_channels

    def forward(self, x):
        if self.num_channels != 1:
            frames = []
            for i in range(self.num_channels):
                frames.append(self.make_frame(x[:, i, :]))

            x = torch.stack(frames, dim=1)

            x = self.frame_backbone(x)
        
            
        # frame backbone
        else:
            x = self.make_frame(x)
            x = self.frame_backbone(x)

        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for block in self.encoder_block:
            x = block(x)

        x = self.encoder_norm(x)
        return x

    def make_frame(self, x):
        size = self.fs * self.second
        step = int(self.time_step * self.fs)
        window = int(self.time_window * self.fs)
        frame = []
        for i in range(0, size, step):
            start_idx, end_idx = i, i+window
            sample = x[..., start_idx: end_idx]
            if sample.shape[-1] == window:
                frame.append(sample)
        frame = torch.stack(frame, dim=1)
        return frame
    
class NeuroNetSpectroEncoderWrapper(nn.Module):
    def __init__(self, fs: int, second: int, time_window: int, time_step: float,
                 patch_embed, encoder_block, encoder_norm, cls_token, pos_embed,
                 final_length):

        super().__init__()
        self.fs, self.second = fs, second
        self.time_window = time_window
        self.time_step = time_step

        self.patch_embed = patch_embed
        self.encoder_block = encoder_block
        self.encoder_norm = encoder_norm
        self.cls_token = cls_token
        self.pos_embed = pos_embed

        self.final_length = final_length

    def forward(self, x):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for block in self.encoder_block:
            x = block(x)

        x = self.encoder_norm(x)
        return x


if __name__ == '__main__':
    x0 = torch.randn((50, 3000))
    m0 = NeuroNet(fs=100, second=30, time_window=5, time_step=0.5,
                  encoder_embed_dim=256, encoder_depths=6, encoder_heads=8,
                  decoder_embed_dim=128, decoder_heads=4, decoder_depths=8,
                  projection_hidden=[1024, 512])
    m0.forward(x0, mask_ratio=0.5)
