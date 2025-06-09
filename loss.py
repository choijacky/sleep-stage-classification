import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss (NT-Xent).
    Used for contrastive learning by maximizing agreement between positive pairs.
    """
    def __init__(self, temperature):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(reduction='sum')
        self.similarity_f = nn.CosineSimilarity(dim=-1)
        self.temperature = temperature

    @staticmethod
    def mask_correlated_samples(batch_size):
        """Create mask to ignore positive pairs in contrastive setup."""
        n = 2 * batch_size
        mask = torch.ones((n, n), dtype=bool)
        mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        batch_size = z_j.shape[0]
        n = 2 * batch_size

        z = torch.cat((z_i, z_j), dim=0)
        z = F.normalize(z, dim=-1)
        mask = self.mask_correlated_samples(batch_size)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0))
        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(n, 1)
        negative_samples = sim[mask].reshape(n, -1)

        labels = torch.zeros(n, dtype=torch.long, device=positive_samples.device)
        logits = torch.cat((positive_samples, negative_samples), dim=1) / self.temperature

        loss = self.criterion(logits, labels) / n
        return loss, (labels, logits)


class MoCo(nn.Module):
    """
    Momentum Contrastive Loss (MoCo)
    """
    def __init__(self, device, T=0.5):
        super().__init__()
        self.T = T
        self.device = device

    def forward(self, emb_anchor, emb_positive, queue):
        # Normalize embeddings
        emb_anchor = F.normalize(emb_anchor, dim=1)
        emb_positive = F.normalize(emb_positive, dim=1)
        queue = F.normalize(queue, dim=1)

        l_pos = torch.einsum('nc,nc->n', emb_anchor, emb_positive).unsqueeze(-1)
        l_neg = torch.einsum('nc,kc->nk', emb_anchor, queue)
        logits = torch.cat([l_pos, l_neg], dim=1) / self.T

        labels = torch.zeros(logits.size(0), dtype=torch.long, device=self.device)
        return F.cross_entropy(logits, labels)


class BYOL(nn.Module):
    """
    Bootstrap Your Own Latent (BYOL) Loss.
    """
    def __init__(self, device, T=0.5):
        super().__init__()
        self.T = T
        self.device = device

    def forward(self, emb_anchor, emb_positive):
        emb_anchor = F.normalize(emb_anchor, dim=1)
        emb_positive = F.normalize(emb_positive, dim=1)
        return -torch.einsum('nc,nc->n', emb_anchor, emb_positive).sum()


class SimSiam(nn.Module):
    """
    Simple Siamese Network for Contrastive Learning (SimSiam) Loss.
    """
    def __init__(self, device, T=0.5):
        super().__init__()
        self.T = T
        self.device = device

    def forward(self, p1, p2, z1, z2):
        p1, p2 = F.normalize(p1, dim=1), F.normalize(p2, dim=1)
        z1, z2 = F.normalize(z1, dim=1), F.normalize(z2, dim=1)
        l_pos1 = torch.einsum('nc,nc->n', p1, z2.detach()).sum()
        l_pos2 = torch.einsum('nc,nc->n', p2, z1.detach()).sum()
        return -(l_pos1 + l_pos2)


class OurLoss(nn.Module):
    """
    Custom instance-aware contrastive loss with margin.
    """
    def __init__(self, device, margin=0.5, sigma=2.0, T=2.0):
        super().__init__()
        self.T = T
        self.device = device
        self.margin = margin
        self.sigma = sigma
        self.softmax = nn.Softmax(dim=1)

    def forward(self, emb_anchor, emb_positive):
        emb_anchor = F.normalize(emb_anchor, dim=1)
        emb_positive = F.normalize(emb_positive, dim=1)

        sim = torch.mm(emb_anchor, emb_positive.t()) / self.T
        weight = self.softmax(sim)
        neg = torch.mm(weight, emb_positive)

        l_pos = torch.exp(-torch.sum((emb_anchor - emb_positive) ** 2, dim=1) / (2 * self.sigma ** 2))
        l_neg = torch.exp(-torch.sum((emb_anchor - neg) ** 2, dim=1) / (2 * self.sigma ** 2))

        loss = torch.max(torch.zeros_like(l_pos), l_neg - l_pos + self.margin).mean()
        return loss


class SimCLR(nn.Module):
    """
    Simple Contrastive Learning of Visual Representations (SimCLR) Loss.
    """
    def __init__(self, device, T=1.0):
        super().__init__()
        self.T = T
        self.device = device

    def forward(self, emb_anchor, emb_positive):
        emb_anchor = F.normalize(emb_anchor, dim=1)
        emb_positive = F.normalize(emb_positive, dim=1)

        N = emb_anchor.size(0)
        emb_total = torch.cat([emb_anchor, emb_positive], dim=0)
        logits = torch.mm(emb_total, emb_total.t())
        logits.fill_diagonal_(-1e10)
        logits /= self.T

        labels = torch.cat([torch.arange(N, 2 * N), torch.arange(N)]).to(self.device)
        return F.cross_entropy(logits, labels)


def get_loss(cfg, device):
    """
    Factory function to initialize loss based on config.
    """
    if 'ContraWR' in cfg.train.model_name:
        return OurLoss(device, cfg.contra.delta, cfg.contra.sigma, cfg.contra.T).to(device)
    elif cfg.train.model_name == 'MoCo':
        return MoCo(device).to(device)
    elif 'SimCLR' in cfg.train.model_name:
        return SimCLR(device).to(device)
    elif 'BYOL' in cfg.train.model_name:
        return BYOL(device).to(device)
    elif 'SimSiam' in cfg.train.model_name:
        return SimSiam(device).to(device)