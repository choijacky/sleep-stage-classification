import sys

sys.path.extend(["./"])

from backbones.neuronet import NeuroNet, PCA_NeuroNet, Spectro_NeuroNet, Spectro_NeuroNet_ViT
from backbones.cnn_backbone import CNNEncoder2D_SLEEP

def get_model(cfg, device):
    if 'neuronet' in cfg.train.model_name:
        if 'pca' in cfg.train.model_name:
            model = PCA_NeuroNet(
                fs=cfg.dataset.rfreq,
                second=cfg.frames.second,
                time_window=cfg.frames.time_window,
                time_step=cfg.frames.time_step,
                encoder_embed_dim=cfg.neuronet.encoder_embed_dim,
                encoder_heads=cfg.neuronet.encoder_heads,
                encoder_depths=cfg.neuronet.encoder_depths,
                decoder_embed_dim=cfg.neuronet.decoder_embed_dim,
                decoder_heads=cfg.neuronet.decoder_heads,
                decoder_depths=cfg.neuronet.decoder_depths,
                projection_hidden=cfg.neuronet.projection_hidden,
                recon_mode=cfg.neuronet.recon_mode,
                temperature=cfg.neuronet.temperature
            ).to(device)

        elif 'patch' in cfg.train.model_name:
            model = Spectro_NeuroNet_ViT(
                time_dim=cfg.spectro.time_dim,
                freq_dim=cfg.spectro.freq_dim,
                patch_size=cfg.spectro.patch_size,
                encoder_embed_dim=cfg.neuronet.encoder_embed_dim,
                encoder_heads=cfg.neuronet.encoder_heads,
                encoder_depths=cfg.neuronet.encoder_depths,
                decoder_embed_dim=cfg.neuronet.decoder_embed_dim,
                decoder_heads=cfg.neuronet.decoder_heads,
                decoder_depths=cfg.neuronet.decoder_depths,
                projection_hidden=cfg.neuronet.projection_hidden,
                recon_mode=cfg.neuronet.recon_mode,
                temperature=cfg.neuronet.temperature
            ).to(device)

        elif 'spectro' in cfg.train.model_name:
            model = Spectro_NeuroNet(
                fs=cfg.dataset.rfreq,
                second=cfg.frames.second,
                time_window=cfg.frames.time_window,
                time_step=cfg.frames.time_step,
                encoder_embed_dim=cfg.neuronet.encoder_embed_dim,
                encoder_heads=cfg.neuronet.encoder_heads,
                encoder_depths=cfg.neuronet.encoder_depths,
                decoder_embed_dim=cfg.neuronet.decoder_embed_dim,
                decoder_heads=cfg.neuronet.decoder_heads,
                decoder_depths=cfg.neuronet.decoder_depths,
                projection_hidden=cfg.neuronet.projection_hidden,
                freq_bins=cfg.neuronet.freq_bins,
                recon_mode=cfg.neuronet.recon_mode,
                temperature=cfg.neuronet.temperature
            ).to(device)
        else:
            model = NeuroNet(
                fs=cfg.dataset.rfreq,
                second=cfg.frames.second,
                time_window=cfg.frames.time_window,
                time_step=cfg.frames.time_step,
                encoder_embed_dim=cfg.neuronet.encoder_embed_dim,
                encoder_heads=cfg.neuronet.encoder_heads,
                encoder_depths=cfg.neuronet.encoder_depths,
                decoder_embed_dim=cfg.neuronet.decoder_embed_dim,
                decoder_heads=cfg.neuronet.decoder_heads,
                decoder_depths=cfg.neuronet.decoder_depths,
                projection_hidden=cfg.neuronet.projection_hidden,
                recon_mode=cfg.neuronet.recon_mode,
                temperature=cfg.neuronet.temperature
            ).to(device)
        return [model]
    else:
        q_encoder = CNNEncoder2D_SLEEP(cfg.contra.n_dim).to(device)
        k_encoder = CNNEncoder2D_SLEEP(cfg.contra.n_dim).to(device)

        for param_q, param_k in zip(q_encoder.parameters(), k_encoder.parameters()):
            param_k.data.copy_(param_q.data) 
            param_k.requires_grad = False  # not update by gradient

        return [q_encoder, k_encoder]