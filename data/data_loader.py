"""dataloader for SHHS and sleep-edfx"""

"""
spectral data augmentation
- chaoqi Oct. 29
"""

import torch
import numpy as np
from scipy.signal import spectrogram
import pickle
from scipy.interpolate import interp1d
from scipy.signal import butter, lfilter, periodogram
import os
import time
from collections import Counter


def denoise_channel(ts, bandpass, signal_freq, bound):
    """
    bandpass: (low, high)
    """
    nyquist_freq = 0.5 * signal_freq
    filter_order = 1
    
    low = bandpass[0] / nyquist_freq
    high = bandpass[1] / nyquist_freq
    b, a = butter(filter_order, [low, high], btype="band")
    ts_out = lfilter(b, a, ts)

    ts_out[ts_out > bound] = bound
    ts_out[ts_out < -bound] = - bound

    return np.array(ts_out)

def noise_channel(ts, mode, degree, bound):
    """
    Add noise to ts
    
    mode: high, low, both
    degree: degree of noise, compared with range of ts    
    
    Input:
        ts: (n_length)
    Output:
        out_ts: (n_length)
        
    """
    len_ts = len(ts)
    num_range = np.ptp(ts)+1e-4 # add a small number for flat signal
    
    ### high frequency noise
    if mode == 'high':
        noise = degree * num_range * (2*np.random.rand(len_ts)-1)
        out_ts = ts + noise
        
    ### low frequency noise
    elif mode == 'low':
        noise = degree * num_range * (2*np.random.rand(len_ts//100)-1)
        x_old = np.linspace(0, 1, num=len_ts//100, endpoint=True)
        x_new = np.linspace(0, 1, num=len_ts, endpoint=True)
        f = interp1d(x_old, noise, kind='linear')
        noise = f(x_new)
        out_ts = ts + noise
        
    ### both high frequency noise and low frequency noise
    elif mode == 'both':
        noise1 = degree * num_range * (2*np.random.rand(len_ts)-1)
        noise2 = degree * num_range * (2*np.random.rand(len_ts//100)-1)
        x_old = np.linspace(0, 1, num=len_ts//100, endpoint=True)
        x_new = np.linspace(0, 1, num=len_ts, endpoint=True)
        f = interp1d(x_old, noise2, kind='linear')
        noise2 = f(x_new)
        out_ts = ts + noise1 + noise2

    else:
        out_ts = ts

    out_ts[out_ts > bound] = bound
    out_ts[out_ts < -bound] = - bound
        
    return out_ts

class SHHSLoader(torch.utils.data.Dataset):
    def __init__(self, list_IDs, dir, SS=True):
        self.list_IDs = list_IDs
        self.dir = dir
        self.SS = SS

        self.label_list = [0, 1, 2, 3, 4]
        self.bandpass1 = (1, 3)
        self.bandpass2 = (30, 60)
        self.n_length = 125 * 30
        self.n_channels = 2
        self.n_classes = 5
        self.signal_freq = 125
        self.bound = 0.000125

    def __len__(self):
        return len(self.list_IDs)

    def add_noise(self, x, ratio):
        """
        Add noise to multiple ts
        Input: 
            x: (n_channel, n_length)
        Output: 
            x: (n_channel, n_length)
        """
        for i in range(self.n_channels):
            if np.random.rand() > ratio:
                mode = np.random.choice(['high', 'low', 'both', 'no'])
                x[i,:] = noise_channel(x[i,:], mode=mode, degree=0.05, bound=self.bound)
        return x
    
    def remove_noise(self, x, ratio):
        """
        Remove noise from multiple ts
        Input: 
            x: (n_channel, n_length)
        Output: 
            x: (n_channel, n_length)
        """
        for i in range(self.n_channels):
            rand = np.random.rand()
            if rand > 0.75:
                x[i, :] = denoise_channel(x[i, :], self.bandpass1, self.signal_freq, bound=self.bound) +\
                        denoise_channel(x[i, :], self.bandpass2, self.signal_freq, bound=self.bound)
            elif rand > 0.5:
                x[i, :] = denoise_channel(x[i, :], self.bandpass1, self.signal_freq, bound=self.bound)
            elif rand > 0.25:
                x[i, :] = denoise_channel(x[i, :], self.bandpass2, self.signal_freq, bound=self.bound)
            else:
                pass

        return x
    
    def crop(self, x):
        l = np.random.randint(1, 3749)
        x[:, :l], x[:, l:] = x[:, -l:], x[:, :-l]

        return x
    
    def augment(self, x):
        # np.random.shuffle(x)
        t = np.random.rand()
        if t > 0.75:
            x = self.add_noise(x, ratio=0.5)
        elif t > 0.5:
            x = self.remove_noise(x, ratio=0.5)
        elif t > 0.25:
            x = self.crop(x)
        else:
            x = x[[1,0],:]
        return x
    
    def __getitem__(self, index):
        path = self.dir + self.list_IDs[index]
        sample = pickle.load(open(path, 'rb'))
        X, y = sample['X'], sample['y']
        
        # original y.unique = [0, 1, 2, 3, 5]
        if y == 4:
            y = 3
        elif y > 4:
            y = 4
        y = torch.LongTensor([y])

        if self.SS:
            aug1 = self.augment(X.copy())
            aug2 = self.augment(X.copy())
            return torch.FloatTensor(aug1), torch.FloatTensor(aug2)
        else:
            return torch.FloatTensor(X), y

class SLEEPCALoader(torch.utils.data.Dataset):
    def __init__(self, list_IDs, dir, n_channels=2, SS=False):
        self.list_IDs = list_IDs
        self.dir = dir
        self.n_channels = n_channels
        self.SS = SS

        self.label_list = ['W', 'R', 1, 2, 3]
        self.bandpass1 = (1, 5)
        self.bandpass2 = (30, 49)
        self.n_length = 100 * 30
        self.n_classes = 5
        self.signal_freq = 100
        self.bound = 0.00025

    def __len__(self):
        return len(self.list_IDs)

    def add_noise(self, x, ratio):
        """
        Add noise to multiple ts
        Input: 
            x: (n_channel, n_length)
        Output: 
            x: (n_channel, n_length)
        """
        for i in range(self.n_channels):
            if np.random.rand() > ratio:
                mode = np.random.choice(['high', 'low', 'both', 'no'])
                x[i,:] = noise_channel(x[i,:], mode=mode, degree=0.05, bound=self.bound)
        return x
    
    def remove_noise(self, x, ratio):
        """
        Remove noise from multiple ts
        Input: 
            x: (n_channel, n_length)
        Output: 
            x: (n_channel, n_length)
        """
        for i in range(self.n_channels):
            rand = np.random.rand()
            if rand > 0.75:
                x[i, :] = denoise_channel(x[i, :], self.bandpass1, self.signal_freq, bound=self.bound) +\
                        denoise_channel(x[i, :], self.bandpass2, self.signal_freq, bound=self.bound)
            elif rand > 0.5:
                x[i, :] = denoise_channel(x[i, :], self.bandpass1, self.signal_freq, bound=self.bound)
            elif rand > 0.25:
                x[i, :] = denoise_channel(x[i, :], self.bandpass2, self.signal_freq, bound=self.bound)
            else:
                pass
        return x
    
    def crop(self, x):
        l = np.random.randint(1, self.n_length - 1)
        x[:, :l], x[:, l:] = x[:, -l:], x[:, :-l]

        return x
    
    def augment(self, x):
        t = np.random.rand()
        if t > 0.75:
            x = self.add_noise(x, ratio=0.5)
        elif t > 0.5:
            x = self.remove_noise(x, ratio=0.5)
        elif t > 0.25:
            x = self.crop(x)
        else:
            x = x[[1,0],:]
        return x
    
    def __getitem__(self, index):
        path = os.path.join(self.dir, self.list_IDs[index])
        sample = pickle.load(open(path, 'rb'))
        try:
            X, y = sample['X'][:self.n_channels, :], sample['y']
        except:
            X, y = sample['X'], sample['y']
        
        # original y.unique = [0, 1, 2, 3, 5]
        # if y == 'W':
        #     y = 0
        # elif y == 'R':
        #     y = 4
        # elif y in ['1', '2', '3']:
        #     y = int(y)
        # elif y == '4':
        #     y = 3
        # else:
        #     y = 0
        try:
            y = torch.LongTensor([int(y) - 1]).squeeze()

        except:
            if y == 'W':
                y = 0
            elif y == 'R':
                y = 4
            elif y in ['1', '2', '3']:
                y = int(y)
            elif y == '4':
                y = 3
            else:
                y = 0

            y = torch.LongTensor([int(y)]).squeeze()

        if self.SS:
            aug1 = self.augment(X.copy())
            aug2 = self.augment(X.copy())
            return torch.FloatTensor(aug1), torch.FloatTensor(aug2)
        else:
            X = torch.FloatTensor(X)
            if self.n_channels == 1:
                X = X.squeeze()
            return X, y
        

class SLEEPCALoader_PCA(SLEEPCALoader):
    def __init__(self, *args, pca_dir, mask_ratio, **kwargs):
        super().__init__(*args, **kwargs)

        self.pca_dir = pca_dir
        self.mask_ratio = mask_ratio

    def setup_mask(self):
        threshold = 1 - self.mask_ratio

        try:
            self.evectors = np.load(os.path.join(self.pca_dir, "pc_matrix_pca.npy"))
            self.evalues = np.load(os.path.join(self.pca_dir, "eigenvalues_ratio_ipca.npy"))
            #self.mean = np.load(os.path.join(self.pca_dir, "mean.npy"))
        except:
            print(f"The path ", os.path.join(self.pca_dir, "pc_matrix_pca.npy"), " does not exist. Or any other PCA path...")

    def __getitem__(self, index):
        path = os.path.join(self.dir, self.list_IDs[index])
        sample = pickle.load(open(path, 'rb'))
        try:
            X, y = sample['X'][:self.n_channels, :], sample['y']
        except:
            X, y = sample['X'], sample['y']

        if y == 'W':
            y = 0
        elif y == 'R':
            y = 4
        elif y in ['1', '2', '3']:
            y = int(y)
        elif y == '4':
            y = 3
        else:
            y = 0
        
        y = torch.LongTensor([y]).squeeze()

        X1 = torch.FloatTensor(X1)
        # X2 = torch.FloatTensor(X2)

        assert self.n_channels == 1

        if self.n_channels == 1:
            X1 = X1.squeeze()
            # X2 = X2.squeeze()

        if self.SS:
            return X1#, self.evectors, self.evalues

        else:
            return X1, y
    
class SLEEPCALoader_spectro(torch.utils.data.Dataset):
    def __init__(self, list_IDs, dir, majority=None):
        self.list_IDs = list_IDs
        self.dir = dir
        self.majority = majority

        self.label_list = ['W', 'R', 1, 2, 3]
        self.n_length = 100 * 30
        self.n_classes = 5
        self.signal_freq = 100

    def __len__(self):
        return len(self.list_IDs)

    
    def __getitem__(self, index):
        if self.majority is None:

            path = os.path.join(self.dir, self.list_IDs[index])

            y = int(self.list_IDs[index].split(".")[0][-1]) - 1
            X = np.load(path)
            
            # original y.unique = [0, 1, 2, 3, 5]
            
            y = torch.LongTensor([y]).squeeze()

            X = torch.FloatTensor(X)
            X = X.squeeze()
            #X = torch.transpose(X, 0, 1)
            return X, y

        else:
            path = os.path.join(self.dir, self.list_IDs[index])

            X = np.load(path)
            X = torch.FloatTensor(X)
            X = X.squeeze()
            X = torch.transpose(X, 0, 1)

            if self.majority == True:
                label = int(self.list_IDs[index].split('--')[-1][0])
                y = torch.LongTensor([label]).squeeze()

                return X, y

            else:
                labels = self.list_IDs[index].split('--')
                labels.pop(0)
                labels.pop()
                assert len(labels) == 5
                labels = [int(label) for label in labels]

                counter = dict(Counter(labels))

                soft_label = np.zeros(5)

                for key in counter:
                    soft_label[key] = counter[key] / 5

                y = torch.Tensor(soft_label)
                
                return X, y



class ISRUC(torch.utils.data.Dataset):
    def __init__(self, list_IDs, dir, n_channels=2, multilabel=True):
        self.list_IDs = list_IDs
        self.dir = dir
        self.n_channels = n_channels
        self.multilabel = multilabel

        self.label_list = ['W', 'R', 1, 2, 3]
        self.bandpass1 = (1, 5)
        self.bandpass2 = (30, 49)
        self.n_length = 100 * 30
        self.n_classes = 5
        self.signal_freq = 100
        self.bound = 0.00025

    def __len__(self):
        return len(self.list_IDs)
    
    def __getitem__(self, index):
        path = os.path.join(self.dir, self.list_IDs[index])
        sample = pickle.load(open(path, 'rb'))
        X, y1, y2 = sample['X'][:self.n_channels, :], sample['y1'], sample['y2']

        y1 = torch.LongTensor([y1 - 1]).squeeze()
        y2 = torch.LongTensor([y2 - 1]).squeeze()
        
        X = torch.FloatTensor(X)
        if self.n_channels == 1:
            X = X.squeeze()

        if self.multilabel == True:
            return X, y1, y2

        else:
            return X, y1
        

class DOD(torch.utils.data.Dataset):
    def __init__(self, list_IDs, dir, n_channels=2, multilabel=True, scorer=1, softlabels=False, consensus=False):
        self.list_IDs = list_IDs
        self.dir = dir
        self.n_channels = n_channels
        self.multilabel = multilabel
        self.scorer = scorer
        self.softlabels = softlabels
        self.consensus = consensus

        self.label_list = ['W', 'R', 1, 2, 3]
        self.bandpass1 = (1, 5)
        self.bandpass2 = (30, 49)
        self.n_length = 100 * 30
        self.n_classes = 5
        self.signal_freq = 100
        self.bound = 0.00025

    def __len__(self):
        return len(self.list_IDs)
    
    def __getitem__(self, index):
        path = os.path.join(self.dir, self.list_IDs[index])
        sample = pickle.load(open(path, 'rb'))

        if self.multilabel == True:
            X, y1, y2, y3, y4, y5 = sample['X'][:self.n_channels, :], sample['y1'], sample['y2'], sample['y3'], sample['y4'], sample['y5']

            y1 = torch.LongTensor([y1]).squeeze()
            y2 = torch.LongTensor([y2]).squeeze()
            y3 = torch.LongTensor([y3]).squeeze()
            y4 = torch.LongTensor([y4]).squeeze()
            y5 = torch.LongTensor([y5]).squeeze()

        elif self.softlabels == True:
            X, y1, y2, y3, y4, y5 = sample['X'][:self.n_channels, :], sample['y1'], sample['y2'], sample['y3'], sample['y4'], sample['y5']

            votes = [y1, y2, y3, y4, y5]

            counter = dict(Counter(votes))

            soft_label = np.zeros(5)

            for key in counter:
                soft_label[key] = counter[key] / 5

            y = torch.Tensor(soft_label)

        elif self.consensus == True:
            X, y = sample['X'][:self.n_channels, :], sample['y']
            # X, y1, y2, y3, y4, y5 = sample['X'][:self.n_channels, :], sample['y1'], sample['y2'], sample['y3'], sample['y4'], sample['y5']

            # labels = [y1, y2, y3, y4, y5]

            # soft_agreement = {
            #     0: 0.884986,
            #     1: 0.908218,
            #     2: 0.917917,
            #     3: 0.838457,
            #     4: 0.916462,
            # }

            # # Count how many times each label appears
            # label_counts = Counter(labels)
            # max_votes = max(label_counts.values())
            
            # # Find all labels that have the highest vote count (to detect ties)
            # top_labels = [label for label, count in label_counts.items() if count == max_votes]
            
            # if len(top_labels) == 1:
            #     # No tie
            #     y = top_labels[0]
            # else:
            #     # Tie: select the label of the scorer with the highest soft agreement
            #     max_agreement = -1
            #     selected_label = None
            #     for scorer_index, label in enumerate(labels):
            #         if label in top_labels:
            #             agreement = soft_agreement.get(scorer_index, 0)
            #             if agreement > max_agreement:
            #                 max_agreement = agreement
            #                 selected_label = label

            #     y = selected_label
                
            y = torch.LongTensor([y]).squeeze()


        else:
            try:
                X, y = sample['X'][:self.n_channels, :], sample['y' + str(self.scorer)]
            except:
                 X, y = sample['X'][:self.n_channels, :], sample['y']

            y = torch.LongTensor([y]).squeeze()
        
        
        X = torch.FloatTensor(X)
        if self.n_channels == 1:
            X = X.squeeze()

        if self.multilabel == True:
            return X, y1, y2, y3, y4, y5

        else:
            return X, y
        

class SLEEPCALoaderComb(torch.utils.data.Dataset):
    def __init__(self, list_IDs, dir, n_channels=2, SS=False):
        self.list_IDs = list_IDs
        self.dir = dir
        self.n_channels = n_channels
        self.SS = SS

        self.label_list = ['W', 'R', 1, 2, 3]
        self.bandpass1 = (1, 5)
        self.bandpass2 = (30, 49)
        self.n_length = 100 * 30
        self.n_classes = 5
        self.signal_freq = 100
        self.bound = 0.00025

    def __len__(self):
        return len(self.list_IDs)

    def add_noise(self, x, ratio):
        """
        Add noise to multiple ts
        Input: 
            x: (n_channel, n_length)
        Output: 
            x: (n_channel, n_length)
        """
        for i in range(self.n_channels):
            if np.random.rand() > ratio:
                mode = np.random.choice(['high', 'low', 'both', 'no'])
                x[i,:] = noise_channel(x[i,:], mode=mode, degree=0.05, bound=self.bound)
        return x
    
    def remove_noise(self, x, ratio):
        """
        Remove noise from multiple ts
        Input: 
            x: (n_channel, n_length)
        Output: 
            x: (n_channel, n_length)
        """
        for i in range(self.n_channels):
            rand = np.random.rand()
            if rand > 0.75:
                x[i, :] = denoise_channel(x[i, :], self.bandpass1, self.signal_freq, bound=self.bound) +\
                        denoise_channel(x[i, :], self.bandpass2, self.signal_freq, bound=self.bound)
            elif rand > 0.5:
                x[i, :] = denoise_channel(x[i, :], self.bandpass1, self.signal_freq, bound=self.bound)
            elif rand > 0.25:
                x[i, :] = denoise_channel(x[i, :], self.bandpass2, self.signal_freq, bound=self.bound)
            else:
                pass
        return x
    
    def crop(self, x):
        l = np.random.randint(1, self.n_length - 1)
        x[:, :l], x[:, l:] = x[:, -l:], x[:, :-l]

        return x
    
    def augment(self, x):
        t = np.random.rand()
        if t > 0.75:
            x = self.add_noise(x, ratio=0.5)
        elif t > 0.5:
            x = self.remove_noise(x, ratio=0.5)
        elif t > 0.25:
            x = self.crop(x)
        else:
            x = x[[1,0],:]
        return x

        return x
    
    def __getitem__(self, index):
        path = os.path.join(self.dir, self.list_IDs[index])
        path2 = path.replace("cassette-", "cassette2-")
        sample = pickle.load(open(path, 'rb'))
        sample2 = pickle.load(open(path2, 'rb'))
        try:
            X, X2, y = sample['X'][:self.n_channels, :], sample2['X'][:self.n_channels, :], sample['y']
        except:
            X, X2, y = sample['X'], sample2['X'], sample['y']
        
        # original y.unique = [0, 1, 2, 3, 5]
        # if y == 'W':
        #     y = 0
        # elif y == 'R':
        #     y = 4
        # elif y in ['1', '2', '3']:
        #     y = int(y)
        # elif y == '4':
        #     y = 3
        # else:
        #     y = 0
        try:
            y = torch.LongTensor([int(y) - 1]).squeeze()

        except:
            if y == 'W':
                y = 0
            elif y == 'R':
                y = 4
            elif y in ['1', '2', '3']:
                y = int(y)
            elif y == '4':
                y = 3
            else:
                y = 0

            y = torch.LongTensor([int(y)]).squeeze()

        if self.SS:
            aug1 = self.augment(X.copy())
            aug2 = self.augment(X.copy())
            return torch.FloatTensor(aug1), torch.FloatTensor(aug2)
        else:
            X = torch.FloatTensor(X)
            X2 = torch.FloatTensor(X2)
            if self.n_channels == 1:
                X = X.squeeze()
                X2 = X2.squeeze()

            X = torch.stack([X, X2], dim=0)
            return X, y
        

class SLEEPEDF_DOD(torch.utils.data.Dataset):
    def __init__(self, list_IDs, dir, n_channels=2):
        self.list_IDs = list_IDs
        self.dir = dir
        self.n_channels = n_channels

        self.label_list = ['W', 'R', 1, 2, 3]
        self.bandpass1 = (1, 5)
        self.bandpass2 = (30, 49)
        self.n_length = 100 * 30
        self.n_classes = 5
        self.signal_freq = 100
        self.bound = 0.00025

    def __len__(self):
        return len(self.list_IDs)
    
    def __getitem__(self, index):
        path = os.path.join(self.dir, self.list_IDs[index])
        sample = pickle.load(open(path, 'rb'))

        if "cassette-SC" in self.list_IDs[index]:
            try:
                X, y = sample['X'][:self.n_channels, :], sample['y']
            except:
                X, y = sample['X'], sample['y']

        else:
            X, y = sample['X'][:self.n_channels, :], sample['y1']
        
        X = torch.FloatTensor(X).squeeze()
        y = torch.LongTensor([y]).squeeze()

        return X, y