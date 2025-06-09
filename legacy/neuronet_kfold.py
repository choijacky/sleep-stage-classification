# -*- coding:utf-8 -*-
import os
import sys

sys.path.extend([os.path.abspath('.'), os.path.abspath('..')])

import time
import mne
import csv
import torch
print(torch.__version__)
import random
import shutil
import warnings
import numpy as np
import torch.optim as opt
from utils import model_size
from sklearn.decomposition import PCA
from torch.utils.tensorboard import SummaryWriter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression as LR
from data.data_loader import SLEEPCALoader, SLEEPCALoader_spectro, ISRUC, DOD, SLEEPEDF_DOD
from backbones.models import get_model
from loss import get_loss
from sklearn.model_selection import KFold

import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq

import hydra
from omegaconf import OmegaConf, open_dict


warnings.filterwarnings(action='ignore')

random_seed = 777
np.random.seed(random_seed)
torch.manual_seed(random_seed)
random.seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

mne.set_log_level(False)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

class Trainer(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.model_name = self.cfg.train.model_name + "-" + str(self.cfg.train.fold)
        if 'neuronet' in self.cfg.train.model_name:
            self.model = get_model(self.cfg, device)[0]
        else:
            self.model = get_model(self.cfg, device)[0]
            self.student = get_model(self.cfg, device)[1]
            self.criterion = get_loss(self.cfg, device)
        
        print('Model Size : {0:.2f}MB'.format(model_size(self.model)))

        self.eff_batch_size = self.cfg.train.train_batch_size * self.cfg.train.train_batch_accumulation
        self.lr = self.cfg.train.train_base_learning_rate * self.eff_batch_size / 256
        if 'neuronet' in cfg.train.model_name:
            self.optimizer = opt.AdamW(self.model.parameters(), lr=self.lr)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=cfg.train.weight_decay)
        #scheduler
        if 'neuronet' in self.cfg.train.model_name:
            self.scheduler = opt.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.cfg.train.train_epochs)
        else:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.2, patience=5)

        self.tensorboard_path = os.path.join(self.cfg.train.ckpt_path, self.model_name, 'tensorboard')

        # remote tensorboard files
        if os.path.exists(self.tensorboard_path):
            shutil.rmtree(self.tensorboard_path)

        self.tensorboard_writer = SummaryWriter(log_dir=self.tensorboard_path)

        #print('Frame Size : {}'.format(self.model.num_patches))
        print('Leaning Rate : {0}'.format(self.lr))

    def train(self, train_loader, test_loader):
        #train_loader, val_loader, test_loader = self.setup_dataloaders()
        if "sampling" in self.model_name:
            sample_ratio = True
        else:
            sample_ratio = None

        total_step = 0
        best_model_state, best_score = self.model.state_dict(), 0

        for epoch in range(self.cfg.train.train_epochs):
            start_epoch = time.time()
            step = 0
            self.model.train()
            self.optimizer.zero_grad()

            for x, x2 in train_loader:
                batch_time = time.time()
                #print(batch_time - start_epoch)
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
                    if 'pca' in self.model_name:
                        x2 = x2.to(device)
                        out = self.model(x, x2)
                        forward_time = time.time()

                    elif self.cfg.dataset.masking != "token":
                        out = self.model(x, mask_ratio=self.cfg.neuronet.mask_ratio, freq_mask_ratio=self.cfg.neuronet.freq_mask_ratio)
                    else:
                        if 'sampling' in self.model_name:
                            out = self.model(x, mask_ratio=self.cfg.neuronet.mask_ratio, sample_ratio=True)

                        elif 'isruc' in self.cfg.dataset.base_path or 'dodh-' in self.cfg.dataset.base_path:
                            x = x[:, 0, :].squeeze()
                            out = self.model(x, mask_ratio=self.cfg.neuronet.mask_ratio, sample_ratio=sample_ratio)
                        else:
                            out = self.model(x, mask_ratio=self.cfg.neuronet.mask_ratio, sample_ratio=sample_ratio)

                    if 'spectro' in self.cfg.dataset.base_path or 'wavelet' in self.cfg.dataset.base_path:
                        if self.cfg.neuronet.contrastive == True:
                            recon_loss, contrastive_loss, (cl_labels, cl_logits) = out
                            loss = self.cfg.neuronet.alpha * contrastive_loss # recon_loss + 
                        else:
                            recon_loss, contrastive_loss, (cl_labels, cl_logits) = out
                            contrastive_loss = 0
                    else:
                        recon_loss, contrastive_loss, (cl_labels, cl_logits) = out
                    
                    if self.cfg.neuronet.contrastive == True:
                        loss = recon_loss + self.cfg.neuronet.alpha * contrastive_loss
                    else:
                        loss = recon_loss# + self.cfg.neuronet.alpha * contrastive_loss

                loss_time = time.time()
                loss.backward()
                #print("loss took ", time.time() - loss_time)
                if not 'neuronet' in self.model_name:
                    for param_q, param_k in zip(self.model.parameters(), self.student.parameters()):
                        param_k.data = param_k.data * self.cfg.contra.m + param_q.data * (1. - self.cfg.contra.m)

                if (step + 1) % self.cfg.train.train_batch_accumulation == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    #torch.cuda.empty_cache()

                if (total_step + 1) % self.cfg.train.print_point == 0:
                    if 'neuronet' in self.model_name:
                        if self.cfg.neuronet.contrastive == True:
                            print('[Epoch] : {0:03d}  [Step] : {1:06d}  '
                                '[Reconstruction Loss] : {2:02.4f}  [Contrastive Loss] : {3:02.4f}  '
                                '[Total Loss] : {4:02.4f}  [Contrastive Acc] : {5:02.4f}'.format(
                                    epoch, total_step + 1, recon_loss, contrastive_loss, loss,
                                    self.compute_metrics(cl_logits, cl_labels)))
                        else:
                            print('[Epoch] : {0:03d}  [Step] : {1:06d}  '
                                '[Reconstruction Loss] : {2:02.4f}  [Contrastive Loss] : {3:02.4f}  '
                                '[Total Loss] : {4:02.4f}'.format(
                                    epoch, total_step + 1, recon_loss, contrastive_loss, loss))
                    else:
                        print('[Epoch] : {0:03d} [Step] : {1:06d} [Total Loss] : {2:02.4f}'.format(epoch, total_step + 1, loss))

                if 'neuronet' in self.model_name:
                    self.tensorboard_writer.add_scalar('Reconstruction Loss', recon_loss, total_step)
                    self.tensorboard_writer.add_scalar('Contrastive loss', contrastive_loss, total_step)
                self.tensorboard_writer.add_scalar('Total loss', loss, total_step)

                step += 1
                total_step += 1

                # for name, param in self.model.named_parameters():
                #     if param.grad is not None:
                #         max_grad = param.grad.abs().max().item()
                #         print(f"Max gradient for {name}: {max_grad}")
            
            if 'neuronet' in self.model_name:
                if "isruc" in self.cfg.dataset.base_path or 'dodh' in self.cfg.dataset.base_path or "edfx+" in self.cfg.dataset.base_path:
                    val_acc, val_mf1 = self.linear_probing(train_loader, test_loader)
                else:
                    val_acc, val_mf1 = self.linear_probing(train_loader, test_loader)
            else:
                val_acc, val_mf1 = self.logistic(val_loader, test_loader)

            if val_mf1 > best_score:
                best_model_state = self.model.state_dict()
                best_score = val_mf1

            print('[Epoch] : {0:03d} \t [Accuracy] : {1:2.4f} \t [Macro-F1] : {2:2.4f} \n'.format(
                epoch, val_acc * 100, val_mf1 * 100))
            self.tensorboard_writer.add_scalar('Validation Accuracy', val_acc, total_step)
            self.tensorboard_writer.add_scalar('Validation Macro-F1', val_mf1, total_step)

            #self.tensorboard_writer.add_scalar('Learning Rate', self.scheduler.get_last_lr(), total_step)

            if self.cfg.neuronet.recon_mode not in ["masked_tokens", "spectro"] and epoch % 5 == 0:
                print("plotting...")
                self.save_rec_signals(train_loader, epoch)

            if 'spectro' in self.model_name and epoch % 5 == 0:
                print("plotting...")
                self.save_spectrograms(train_loader, epoch)

            self.optimizer.step()
            self.scheduler.step(loss)

            end_epoch = time.time()

            #print("Forward took ", loss_calc_time - forward_time, " seconds!")
            #print("Loss Calc took ", lossstep_time - loss_calc_time, " seconds!")
            # print("Loss step took ", optimizer_time - lossstep_time, " seconds!")
            # print("Optimizer took ", validation_time - optimizer_time, " seconds!")
            # print("Validation took ", end_val_time - validation_time, " seconds!")
            # print("Scheduler took ", end_epoch - scheduler_time, " seconds!")
            print("1 Epoch took ", end_epoch - start_epoch, " seconds!")
    
        self.save_ckpt(model_state=best_model_state)


    def train_pca(self, train_loader, test_loader):
        #train_loader, val_loader, test_loader = self.setup_dataloaders()

        self.pca_dir = os.path.join(self.cfg.dataset.base_path, "pca_train")

        try:
            evectors = np.load(os.path.join(self.pca_dir, "pc_matrix_pca.npy"))
            evalues = np.load(os.path.join(self.pca_dir, "eigenvalues_ratio_ipca.npy"))

            evectors = torch.FloatTensor(evectors).to(device)
            evalues = torch.FloatTensor(evalues).to(device)
            #self.mean = np.load(os.path.join(self.pca_dir, "mean.npy"))
        except:
            print(f"The path ", os.path.join(self.pca_dir, "pc_matrix_pca.npy"), " does not exist. Or any other PCA path...")

        sample_ratio = (self.cfg.dataset.masking == "pca_sampling")
        print("Sampling_ratio: ", sample_ratio)

        total_step = 0
        best_model_state, best_score = self.model.state_dict(), 0

        for epoch in range(self.cfg.train.train_epochs):
            start_epoch = time.time()
            step = 0
            self.model.train()
            self.optimizer.zero_grad()

            for x, _ in train_loader:
                x = x.to(device)
                
                out = self.model(x, evectors, evalues, self.cfg.neuronet.mask_ratio, self.cfg.neuronet.asym_loss, sample_ratio=sample_ratio)
                forward_time = time.time()
                recon_loss, contrastive_loss, (cl_labels, cl_logits) = out
                if self.cfg.neuronet.contrastive == True:
                    loss = recon_loss + self.cfg.neuronet.alpha * contrastive_loss
                else:
                    loss = recon_loss


                loss_time = time.time()
                loss.backward()
                print("loss took ", time.time()-loss_time)
                if not 'neuronet' in self.model_name:
                    for param_q, param_k in zip(self.model.parameters(), self.student.parameters()):
                        param_k.data = param_k.data * self.cfg.contra.m + param_q.data * (1. - self.cfg.contra.m) 

                if (step + 1) % self.cfg.train.train_batch_accumulation == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    #torch.cuda.empty_cache()

                if (total_step + 1) % self.cfg.train.print_point == 0:
                    if 'neuronet' in self.model_name:
                        print('[Epoch] : {0:03d}  [Step] : {1:06d}  '
                            '[Reconstruction Loss] : {2:02.4f}  [Contrastive Loss] : {3:02.4f}  '
                            '[Total Loss] : {4:02.4f}  [Contrastive Acc] : {5:02.4f}'.format(
                                epoch, total_step + 1, recon_loss, contrastive_loss, loss,
                                self.compute_metrics(cl_logits, cl_labels)))
                    else:
                        print('[Epoch] : {0:03d} [Step] : {1:06d} [Total Loss] : {2:02.4f}'.format(epoch, total_step + 1, loss))

                if 'neuronet' in self.model_name:
                    self.tensorboard_writer.add_scalar('Reconstruction Loss', recon_loss, total_step)
                    self.tensorboard_writer.add_scalar('Contrastive loss', contrastive_loss, total_step)
                self.tensorboard_writer.add_scalar('Total loss', loss, total_step)

                step += 1
                total_step += 1
            
            val_acc, val_mf1 = self.linear_probing(train_loader, test_loader)

            if val_mf1 > best_score:
                best_model_state = self.model.state_dict()
                best_score = val_mf1

            print('[Epoch] : {0:03d} \t [Accuracy] : {1:2.4f} \t [Macro-F1] : {2:2.4f} \n'.format(
                epoch, val_acc * 100, val_mf1 * 100))
            self.tensorboard_writer.add_scalar('Validation Accuracy', val_acc, total_step)
            self.tensorboard_writer.add_scalar('Validation Macro-F1', val_mf1, total_step)
            #self.tensorboard_writer.add_scalar('Learning Rate', self.scheduler.get_last_lr(), total_step)

            self.optimizer.step()
            self.scheduler.step(loss)

            end_epoch = time.time()

            print("1 Epoch took ", end_epoch - start_epoch, " seconds!")

        self.save_ckpt(model_state=best_model_state)

    def linear_probing(self, val_dataloader, test_dataloader):
        self.model.eval()
        (train_x, train_y), (test_x, test_y) = self.get_latent_vector(val_dataloader), \
                                               self.get_latent_vector(test_dataloader)
        
        #print(train_x.shape)
        pca = PCA(n_components=50)
        train_x = pca.fit_transform(train_x)
        test_x = pca.transform(test_x)

        model = KNeighborsClassifier()
        model.fit(train_x, train_y)

        out = model.predict(test_x)
        acc, mf1 = accuracy_score(test_y, out), f1_score(test_y, out, average='macro')
        self.model.train()
        return acc, mf1

    def logistic(self, val_dataloader, test_dataloader):
        self.model.eval()
        (train_x, train_y), (test_x, test_y) = self.get_latent_vector(val_dataloader), \
                                               self.get_latent_vector(test_dataloader)
        
        log_model = LR(solver='lbfgs', multi_class='multinomial', max_iter=500)
        log_model.fit(train_x, train_y)

        out = log_model.predict(test_x)
        acc, mf1 = accuracy_score(test_y, out), f1_score(test_y, out, average='macro')
        self.model.train()
        return acc, mf1

    def get_latent_vector(self, dataloader):
        total_x, total_y = [], []
        with torch.no_grad():
            for data in dataloader:
                x, y = data
                x, y = x.to(device), y.to(device)
                if "isruc" in self.cfg.dataset.base_path or "dodh-" in self.cfg.dataset.base_path:
                    x = x[:, 0, :].squeeze()
                b = x.shape[0]
                if "neuronet" in self.model_name:
                    latent = self.model.forward_latent(x)
                else:
                    latent = self.model(x)
                total_x.append(latent.detach().cpu().numpy().reshape(b, -1))
                total_y.append(y.detach().cpu().numpy())

        total_x, total_y = np.concatenate(total_x, axis=0), np.concatenate(total_y, axis=0)

        return total_x, total_y

    def save_ckpt(self, model_state):
        ckpt_path = os.path.join(self.cfg.train.ckpt_path, self.model_name, 'model')
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)
        
        if 'neuronet' in self.model_name:
            torch.save({
                'model_name': 'NeuroNet',
                'model_state': model_state,
                'model_parameter': {
                    'fs': self.cfg.dataset.rfreq, 'second': self.cfg.frames.second,
                    'time_window': self.cfg.frames.time_window, 'time_step': self.cfg.frames.time_step,
                    'encoder_embed_dim': self.cfg.neuronet.encoder_embed_dim, 'encoder_heads': self.cfg.neuronet.encoder_heads,
                    'encoder_depths': self.cfg.neuronet.encoder_depths,
                    'decoder_embed_dim': self.cfg.neuronet.decoder_embed_dim, 'decoder_heads': self.cfg.neuronet.decoder_heads,
                    'decoder_depths': self.cfg.neuronet.decoder_depths,
                    'projection_hidden': self.cfg.neuronet.projection_hidden, 'temperature': self.cfg.neuronet.temperature, 'recon_mode': self.cfg.neuronet.recon_mode,
                },
                'hyperparameter': self.cfg.__dict__,
            }, os.path.join(ckpt_path, 'best_model.pth'))
        else:
            torch.save({
                'model_name': self.model_name,
                'model_state': model_state,
                'model_parameter': {
                    'fs': self.cfg.dataset.rfreq, 'second': self.cfg.frames.second,
                    'time_window': self.cfg.frames.time_window, 'time_step': self.cfg.frames.time_step,
                },
                'hyperparameter': self.cfg.__dict__,
            }, os.path.join(ckpt_path, 'best_model.pth'))

    def save_rec_signals(self, val_dataloader, epoch):
        sampled = {}  # Dictionary to store one sample per class

        self.model.eval()

        for inputs, labels in val_dataloader:
            for input, label in zip(inputs, labels):
                label = label.item()  # Ensure it's a scalar
                if label not in sampled:
                    sampled[label] = input  # Store the first encountered instance
                if len(sampled) == 5:  # Stop when we have all classes
                    break
            if len(sampled) == 5:
                break
        
        with torch.no_grad():
            for label, signal in sampled.items():
                signal = torch.unsqueeze(signal, dim=0)
                signal = signal.to(device)
                recon, mask = self.model.forward_reconstruction(signal)
                recon = recon.squeeze().detach().cpu().numpy()
                mask = mask.squeeze().detach().cpu().numpy()
                self.plot_signals(signal.squeeze().detach().cpu().numpy(), recon, mask, label=label, step=epoch)
            

        self.model.train()

    def plot_signals(self, original_signal, reconstructed_signal, mask, label, step=0):
        """
        Plots two signals in both the time domain and frequency domain and logs them to TensorBoard.
        
        Parameters:
            time (numpy array): Time values for plotting.
            original_signal (numpy array): The original signal.
            reconstructed_signal (numpy array): Reconstructed signal.
            sample_rate (float): Sampling rate of the signals.
            signal_name (str): Name of the signal for titles.
            writer (SummaryWriter): TensorBoard writer for logging.
            step (int): Global step for logging.
        """
        time = np.linspace(0, self.cfg.frames.second, self.cfg.frames.second * self.cfg.dataset.sfreq)

        mask = np.repeat(mask, int(self.cfg.frames.time_window * self.cfg.dataset.sfreq))

        shaded_regions = []
        in_shade = False
        start = 0

        assert len(mask) == len(original_signal)

        for i in range(len(mask)):
            if mask[i] == 1 and not in_shade:  # Start of a shaded region
                start = time[i]
                in_shade = True
            elif mask[i] == 0 and in_shade:  # End of a shaded region
                end = time[i]
                shaded_regions.append((start, end))
                in_shade = False

        # Handle case where mask ends with a shaded region
        if in_shade:
            shaded_regions.append((start, time[-1]))
        
        # Time Domain Plot
        plt.figure(figsize=(25, 6))
        for start, end in shaded_regions:
            plt.axvspan(start, end, color='red', alpha=0.3)
        plt.plot(time, reconstructed_signal, label="Reconstructed Signal", linewidth=0.5, color='red', alpha=0.7)
        plt.plot(time, original_signal, label="Original Signal", color='blue', linewidth=0.5, alpha=1.0)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.xlim(0, max(time))
        plt.grid(True)
        plt.legend()
        
        # Log to TensorBoard
        self.tensorboard_writer.add_figure(str(label) + "_Time_Domain", plt.gcf(), global_step=step)
        plt.close()
        
        # Frequency Domain Plot
        N = len(original_signal)  # Number of samples
        yf_original = rfft(original_signal)
        yf_rec = rfft(reconstructed_signal)
        xf = rfftfreq(N, 1 / self.cfg.dataset.sfreq)
        
        plt.figure(figsize=(25, 6))
        plt.plot(xf, np.abs(yf_original), linewidth=0.5, color='blue', label="Original Signal", alpha=1.0)
        plt.plot(xf, np.abs(yf_rec), linewidth=0.5, label="Reconstructed Signal", color='red', alpha=0.7)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        plt.legend()
        
        # Log to TensorBoard
        self.tensorboard_writer.add_figure(str(label) + "_Frequency_Domain", plt.gcf(), global_step=step)
        plt.close()

    def save_spectrograms(self, val_dataloader, epoch):
        sampled = {}  # Dictionary to store one sample per class

        self.model.eval()

        for inputs, labels in val_dataloader:
            for input, label in zip(inputs, labels):
                label = label.item()  # Ensure it's a scalar
                if label not in sampled:
                    sampled[label] = input  # Store the first encountered instance
                if len(sampled) == 5:  # Stop when we have all classes
                    break
            if len(sampled) == 5:
                break
        
        with torch.no_grad():
            for label, spec in sampled.items():
                spec = torch.unsqueeze(spec, dim=0)
                spec = spec.to(device)
                masked_original, recon = self.model.forward_spec(spec, self.cfg.neuronet.mask_ratio, self.cfg.neuronet.freq_mask_ratio)
                recon = recon.squeeze().detach().cpu().numpy()
                masked_original = masked_original.squeeze().detach().cpu().numpy()

                self.plot_spectrograms(spec.squeeze().detach().cpu().numpy(), masked_original, recon, label=label, step=epoch)

        self.model.train()

    def plot_spectrograms(self, spectrogram, masked_spectrogram, reconstructed_spectrogram, label, step):

        # spectrogram = np.transpose(spectrogram)
        # masked_spectrogram = np.transpose(masked_spectrogram)
        # reconstructed_spectrogram = np.transpose(reconstructed_spectrogram)

        fig, axes = plt.subplots(1, 3, figsize=(20, 5))
        titles = ["Original Spectrogram", "Masked Spectrogram", "Reconstructed Spectrogram"]
        images = [spectrogram, masked_spectrogram, reconstructed_spectrogram]
        
        for ax, img, title in zip(axes, images, titles):
            im = ax.imshow(img, aspect='auto', origin='lower', cmap='jet')
            ax.set_title(title)
            ax.set_xlabel("Time (compressed)")
            ax.set_ylabel("Scale (Frequency)")
            fig.colorbar(im, ax=ax, label="Power")
        
        plt.tight_layout()

        plt.savefig("/cluster/project/jbuhmann/choij/sleep-stage-classification/tera.png")

        self.tensorboard_writer.add_figure(str(label) + "_Spectrogram", plt.gcf(), global_step=step)
        
        plt.close(fig)

    def setup_dataloaders(self):
        siamese = False
        if "neuronet" not in self.cfg.train.model_name:
            siamese = True

        train_dir = os.path.join(self.cfg.dataset.base_path, "test")
        val_dir = os.path.join(self.cfg.dataset.base_path, "val")
        test_dir = os.path.join(self.cfg.dataset.base_path, "test")

        train_index = os.listdir(train_dir)
        val_index = os.listdir(val_dir)
        test_index = os.listdir(test_dir)

        print("Train: ", len(train_index))
        print("Val: ", len(val_index))
        print("Test: ", len(test_index))

        if "spectro" in self.cfg.dataset.base_path or "wavelet" in self.cfg.dataset.base_path:
            train_dataset = SLEEPCALoader_spectro(train_index, train_dir)
            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.cfg.train.train_batch_size, shuffle=True)

            val_dataset = SLEEPCALoader_spectro(val_index, val_dir)
            val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=self.cfg.train.train_batch_size, shuffle=False)

            test_dataset = SLEEPCALoader_spectro(test_index, test_dir)
            test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=self.cfg.train.train_batch_size, shuffle=False)
        
        elif "isruc" in self.cfg.dataset.base_path:
            train_dataset = ISRUC(train_index, train_dir, 7, False)
            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.cfg.train.train_batch_size, shuffle=True)

            test_dataset = ISRUC(test_index, test_dir, 7, False)
            test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=self.cfg.train.train_batch_size, shuffle=False)

            val_dataloader = None

        elif "edfx+" in self.cfg.dataset.base_path:
            train_dataset = SLEEPEDF_DOD(train_index, train_dir, 1)
            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.cfg.train.train_batch_size, shuffle=True)

            test_dataset = SLEEPEDF_DOD(test_index, test_dir, 1)
            test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=self.cfg.train.train_batch_size, shuffle=False)

            val_dataloader = None

        elif "dodh" in self.cfg.dataset.base_path:
            train_dataset = DOD(train_index, train_dir, 12, False)

            if self.cfg.dataset.balanced_sampling == True:
                train_targets = []
                for _, target in train_dataset:
                    train_targets.append(target)
                train_targets = torch.tensor(train_targets)
                
                class_sample_counts = [56948, 18945, 61529, 11921, 23233]
                weights = 1. / torch.tensor(class_sample_counts, dtype=torch.float)
                samples_weights = weights[train_targets]

                # Balanced sampler by oversampling
                sampler = torch.utils.data.WeightedRandomSampler(
                    weights=samples_weights,
                    num_samples=len(samples_weights),
                    replacement=True)

                train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.cfg.train.train_batch_size, sampler=sampler)
            
            else:
                train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.cfg.train.train_batch_size, shuffle=True)

            test_dataset = DOD(test_index, test_dir, 12, False)
            test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=self.cfg.train.train_batch_size, shuffle=False)

            val_dataloader = None

        else:
            train_dataset = SLEEPCALoader(train_index, train_dir, self.cfg.frames.n_channels, siamese)

            if self.cfg.dataset.balanced_sampling == True:
                train_targets = []
                for _, target in train_dataset:
                    train_targets.append(target)
                train_targets = torch.tensor(train_targets)
                
                class_sample_counts = [56948, 18945, 61529, 11921, 23233]
                weights = 1. / torch.tensor(class_sample_counts, dtype=torch.float)
                samples_weights = weights[train_targets]

                # Balanced sampler by oversampling
                sampler = torch.utils.data.WeightedRandomSampler(
                    weights=samples_weights,
                    num_samples=len(samples_weights),
                    replacement=True)

                train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.cfg.train.train_batch_size, sampler=sampler)

            else:
                train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.cfg.train.train_batch_size, shuffle=True)

            val_dataset = SLEEPCALoader(val_index, val_dir, self.cfg.frames.n_channels, False)
            val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=self.cfg.train.train_batch_size, shuffle=False)

            test_dataset = SLEEPCALoader(test_index, test_dir, self.cfg.frames.n_channels, False)
            test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=self.cfg.train.train_batch_size, shuffle=False)

        return train_dataloader, val_dataloader, test_dataloader


    @staticmethod
    def compute_metrics(output, target):
        output = output.argmax(dim=-1)
        accuracy = torch.mean(torch.eq(target, output).to(torch.float32))
        return accuracy

@hydra.main(version_base=None, config_path="conf", config_name="train_config")
def my_trainer(cfg) -> None:
    print(OmegaConf.to_yaml(cfg))

    print("Starting script")
    print("start training?")


    k = 5
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)

    train_dir = os.path.join(cfg.dataset.base_path, "train")
    train_index = os.listdir(train_dir)

    patient_ids = list(set([s.split("-")[1] for s in train_index]))
    patient_ids = np.array(patient_ids)

    test_ids =  []
    train_ids = []
    with open("test_ids.csv", "r") as f:
        reader = csv.reader(f, delimiter="\t")
        for i, row in enumerate(reader):
            if i == cfg.train.fold:
                test_ids.extend(row[0].split(','))
            else:
                train_ids.extend(row[0].split(','))

    print(test_ids)
    
    train_files = [file for file in train_index if any(id in file for id in train_ids)]
    test_files = [file for file in train_index if any(id in file for id in test_ids)]

    if "spectro" in cfg.dataset.base_path or "wavelet" in cfg.dataset.base_path:
        train_dataset = SLEEPCALoader_spectro(train_files, train_dir)
        test_dataset = SLEEPCALoader_spectro(test_files, train_dir)
        
    else:
        train_dataset = SLEEPCALoader(train_files, train_dir, 1, False)
        test_dataset = SLEEPCALoader(test_files, train_dir, 1, False)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.train.train_batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.train.train_batch_size)

    print("Starting script...")
    trainer = Trainer(cfg)

    if 'pca' in cfg.train.model_name:
        trainer.train_pca(train_dataloader, test_dataloader)
    else:
        trainer.train(train_dataloader, test_dataloader)
    

if __name__ == '__main__':
    my_trainer()
