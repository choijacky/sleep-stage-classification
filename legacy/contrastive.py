"""
Code taken from ContraWR github

# TODO
- Put dataloaders into Lightning Trainer
- Implement validation step similar to NeuroNet


"""

import torch
from data.data_loader import SLEEPCALoader, SHHSLoader
import numpy as np
import torch.nn as nn
import os
import argparse
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression as LR
from backbones.cnn_backbone import CNNEncoder2D_SLEEP, CNNEncoder2D_SHHS
from loss import MoCo, SimCLR, BYOL, OurLoss, SimSiam
from tqdm import tqdm
from utils import model_size
import lightning as L
from torch.utils.tensorboard import SummaryWriter

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class pl_model(L.LightningModule):
    def __init__(self, q_encoder, k_encoder, criterion, m):
        super().__init__()
        self.q_encoder = q_encoder
        self.k_encoder = k_encoder
        self.criterion = criterion
        self.m = m
        

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=5)

        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1000}]
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        aug1, aug2 = batch

        if args.model in ['BYOL']:
            emb_aug1 = q_encoder(aug1, mid=False, byol=True)
            emb_aug2 = k_encoder(aug2, mid=False)
        elif args.model in ['SimCLR']:
            emb_aug1 = q_encoder(aug1, mid=False)
            emb_aug2 = q_encoder(aug2, mid=False)
        elif args.model in ['ContraWR']:
            emb_aug1 = q_encoder(aug1, mid=False)
            emb_aug2 = k_encoder(aug2, mid=False)
        elif args.model in ['MoCo']:
            emb_aug1 = q_encoder(aug1, mid=False)
            emb_aug2 = k_encoder(aug2, mid=False)
        elif args.model in ['SimSiam']:
            emb_aug1, proj1 = q_encoder(aug1, simsiam=True)
            emb_aug2, proj2 = q_encoder(aug2, simsiam=True)

        # backpropagation
        if args.model == 'MoCo':
            loss = self.criterion(emb_aug1, emb_aug2, queue)
            if queue_ptr + emb_aug2.shape[0] > n_queue:
                queue[queue_ptr:] = emb_aug2[:n_queue-queue_ptr]
                queue[:queue_ptr+emb_aug2.shape[0]-n_queue] = emb_aug2[-(queue_ptr+emb_aug2.shape[0]-n_queue):]
                queue_ptr = (queue_ptr + emb_aug2.shape[0]) % n_queue
            else:
                queue[queue_ptr:queue_ptr+emb_aug2.shape[0]] = emb_aug2
        elif args.model == 'SimSiam':
            loss = self.criterion(proj1, proj2, emb_aug1, emb_aug2)
        else:
            loss = self.criterion(emb_aug1, emb_aug2)

        return loss
    
    def validation_step(self, batch, batch_idx):
        X_val, y_val = batch

    def val_dataloader(self):
        return super().val_dataloader()

    # For EMA
    def on_before_zero_grad(self, *args, **kwargs):
        for param_q, param_k in zip(self.q_encoder.parameters(), self.k_encoder.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m) 


# evaluation design
def task(X_train, X_test, y_train, y_test, n_classes):
            
    cls = LR(solver='lbfgs', multi_class='multinomial', max_iter=500)
    cls.fit(X_train, y_train)
    pred = cls.predict(X_test)
    
    res = accuracy_score(y_test, pred)
    cm = confusion_matrix(y_test, pred)
    
    return res, cm

def Pretext(q_encoder, k_encoder, optimizer, Epoch, criterion, pretext_loader, train_loader, test_loader):

    q_encoder.train()
    k_encoder.train()

    global queue
    global queue_ptr
    global n_queue

    step = 0

    best_score = 0
    best_q_encoder_state  = q_encoder.state_dict()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=5)

    all_loss, acc_score = [], []
    for epoch in range(Epoch):
        # save model
        # torch.save(q_encoder.state_dict(), open(os.path.join('ckpt', args.model, \
        #     'Epoch_{}-T-{}-delta-{}.model'.format(epoch, args.T, args.delta)), 'wb'))
        print ()
        for index, (aug1, aug2) in enumerate(tqdm(pretext_loader)):
            aug1, aug2 = aug1.to(device), aug2.to(device)
            if args.model in ['BYOL']:
                emb_aug1 = q_encoder(aug1, mid=False, byol=True)
                emb_aug2 = k_encoder(aug2, mid=False)
            elif args.model in ['SimCLR']:
                emb_aug1 = q_encoder(aug1, mid=False)
                emb_aug2 = q_encoder(aug2, mid=False)
            elif args.model in ['ContraWR']:
                emb_aug1 = q_encoder(aug1, mid=False)
                emb_aug2 = k_encoder(aug2, mid=False)
            elif args.model in ['MoCo']:
                emb_aug1 = q_encoder(aug1, mid=False)
                emb_aug2 = k_encoder(aug2, mid=False)
            elif args.model in ['SimSiam']:
                emb_aug1, proj1 = q_encoder(aug1, simsiam=True)
                emb_aug2, proj2 = q_encoder(aug2, simsiam=True)

            # backpropagation
            if args.model == 'MoCo':
                loss = criterion(emb_aug1, emb_aug2, queue)
                if queue_ptr + emb_aug2.shape[0] > n_queue:
                    queue[queue_ptr:] = emb_aug2[:n_queue-queue_ptr]
                    queue[:queue_ptr+emb_aug2.shape[0]-n_queue] = emb_aug2[-(queue_ptr+emb_aug2.shape[0]-n_queue):]
                    queue_ptr = (queue_ptr + emb_aug2.shape[0]) % n_queue
                else:
                    queue[queue_ptr:queue_ptr+emb_aug2.shape[0]] = emb_aug2
            elif args.model == 'SimSiam':
                loss = criterion(proj1, proj2, emb_aug1, emb_aug2)
            else:
                loss = criterion(emb_aug1, emb_aug2)

            # loss back
            all_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() # only update encoder_q

            # exponential moving average (EMA)
            for param_q, param_k in zip(q_encoder.parameters(), k_encoder.parameters()):
                param_k.data = param_k.data * args.m + param_q.data * (1. - args.m) 

            N = 1000
            if (step + 1) % N == 0:
                acc_score.append([sum(all_loss[-N:]) / len(all_loss[-N:]), evaluate(q_encoder, train_loader, test_loader)])
                scheduler.step(sum(all_loss[-50:]))
            step += 1

        # print the lastest result
        print ('epoch: {}'.format(epoch))
        for i in acc_score[-10:]:
            print (i)

        if len(acc_score) >= 5:
            print ('mean: {}, std: {}'.format(np.array(acc_score)[-5:, -1].mean(), np.array(acc_score)[-5:, -1].std()))

        val_acc, val_mf1 = linear_probing(q_encoder, train_loader, test_loader)

        if val_mf1 > best_score:
                best_q_encoder_state = q_encoder.state_dict()
                best_score = val_mf1

def linear_probing(q_encoder, val_dataloader, eval_dataloader):
    q_encoder.eval()

    emb_val, gt_val = [], []
    with torch.no_grad():
        for (X_val, y_val) in val_dataloader:
            X_val = X_val.to(device)
            emb_val.extend(q_encoder(X_val).cpu().tolist())
            gt_val.extend(y_val.numpy().flatten())
    emb_val, gt_val = np.array(emb_val), np.array(gt_val)

    emb_test, gt_test = [], []
    with torch.no_grad():
        for (X_test, y_test) in test_loader:
            X_test = X_test.to(device)
            emb_test.extend(q_encoder(X_test).cpu().tolist())
            gt_test.extend(y_test.numpy().flatten())
    emb_test, gt_test= np.array(emb_test), np.array(gt_test)

    pca = PCA(n_components=50)
    emb_val = pca.fit_transform(emb_val)
    emb_test = pca.transform(emb_test)

    model = KNeighborsClassifier()
    model.fit(emb_val, gt_val)

    out = model.predict(emb_test)
    acc, mf1 = accuracy_score(gt_test, out), f1_score(gt_test, out, average='macro')
    q_encoder.train()
    return acc, mf1



def evaluate(q_encoder, train_loader, test_loader):
    # freeze
    q_encoder.eval()

    # process val
    emb_val, gt_val = [], []
    with torch.no_grad():
        for (X_val, y_val) in train_loader:
            X_val = X_val.to(device)
            emb_val.extend(q_encoder(X_val).cpu().tolist())
            gt_val.extend(y_val.numpy().flatten())
    emb_val, gt_val = np.array(emb_val), np.array(gt_val)
    # print(Counter(gt_val))

    emb_test, gt_test = [], []
    with torch.no_grad():
        for (X_test, y_test) in test_loader:
            X_test = X_test.to(device)
            emb_test.extend(q_encoder(X_test).cpu().tolist())
            gt_test.extend(y_test.numpy().flatten())
    emb_test, gt_test= np.array(emb_test), np.array(gt_test)
    # print(Counter(gt_test))                

    res, cm = task(emb_val, emb_test, gt_val, gt_test, 5)
    # print (cm, 'accuracy', res)
    # print (cm)
    q_encoder.train()
    return res

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30, help="number of epochs")
    parser.add_argument('--lr', type=float, default=0.5e-3, help="learning rate")
    parser.add_argument('--n_dim', type=int, default=128, help="hidden units (for SHHS, 256, for Sleep, 128)")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="weight decay")
    parser.add_argument('--pretext', type=int, default=10, help="pretext subject")
    parser.add_argument('--training', type=int, default=10, help="training subject")
    parser.add_argument('--n_channels', type=int, default=2, help="number of eeg channels")
    parser.add_argument('--batch_size', type=int, default=256, help="batch_size")
    parser.add_argument('--m', type=float, default=0.9995, help="moving coefficient")
    parser.add_argument('--model', type=str, default='ContraWR', help="which model")
    parser.add_argument('--base_path', type=str, default='/cluster/project/jbuhmann/choij/NeuroNet/dataset', help="path to dataset")
    parser.add_argument('--T', type=float, default=0.3,  help="T")
    parser.add_argument('--sigma', type=float, default=2.0,  help="sigma")
    parser.add_argument('--delta', type=float, default=0.2,  help="delta")
    parser.add_argument('--dataset', type=str, default='SLEEP', help="dataset")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print ('device:', device)

    # set random seed
    seed = 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # torch.backends.cudnn.benchmark = True

    global queue
    global queue_ptr
    global n_queue

    if args.dataset == 'SLEEP':
        # dataset
        pretext_dir = os.path.join(args.base_path, "train")
        train_dir = os.path.join(args.base_path, "val")
        test_dir = os.path.join(args.base_path, "test")

        pretext_index = os.listdir(pretext_dir)
        train_index = os.listdir(train_dir)
        train_index = train_index[:len(train_index)//2]
        test_index = os.listdir(test_dir)

        print ('pretext (all patient): ', len(pretext_index))
        print ('train (all patient): ', len(train_index))
        print ('test (all) patient): ', len(test_index))

        pretext_loader = torch.utils.data.DataLoader(SLEEPCALoader(pretext_index, pretext_dir, args.n_channels, True), 
                        batch_size=args.batch_size, shuffle=True)
        train_loader = torch.utils.data.DataLoader(SLEEPCALoader(train_index, train_dir, args.n_channels, False), 
                        batch_size=args.batch_size, shuffle=False)
        test_loader = torch.utils.data.DataLoader(SLEEPCALoader(test_index, test_dir, args.n_channels, False), 
                        batch_size=args.batch_size, shuffle=False)

        # define and initialize the model
        q_encoder = CNNEncoder2D_SLEEP(args.n_dim)
        q_encoder.to(device)
        print('Model Size : {0:.2f}MB'.format(model_size(q_encoder)))
        k_encoder = CNNEncoder2D_SLEEP(args.n_dim)
        k_encoder.to(device)

    elif args.dataset == 'SHHS':
        # dataset
        pretext_dir = '/srv/local/data/SHHS/processed/pretext/'
        train_dir = '/srv/local/data/SHHS/processed/train/'
        test_dir = '/srv/local/data/SHHS/processed/test/'

        pretext_index = os.listdir(pretext_dir)
        pretext_index = pretext_index[:len(pretext_index)//10]
        train_index = os.listdir(train_dir)
        train_index = train_index[:len(train_index)//10]
        test_index = os.listdir(test_dir)
        test_index = test_index[:len(test_index)//10]

        print ('pretext (all patient): ', len(pretext_index))
        print ('train (all patient): ', len(train_index))
        print ('test (all) patient): ', len(test_index))

        pretext_loader = torch.utils.data.DataLoader(SHHSLoader(pretext_index, pretext_dir, True), 
                        batch_size=args.batch_size, shuffle=True, num_workers=20)
        train_loader = torch.utils.data.DataLoader(SHHSLoader(train_index, train_dir, False), 
                        batch_size=args.batch_size, shuffle=False, num_workers=20)
        test_loader = torch.utils.data.DataLoader(SHHSLoader(test_index, test_dir, False), 
                        batch_size=args.batch_size, shuffle=False, num_workers=20)

        # define the model
        q_encoder = CNNEncoder2D_SHHS(args.n_dim)
        q_encoder.to(device)

        k_encoder = CNNEncoder2D_SHHS(args.n_dim)
        k_encoder.to(device)

    for param_q, param_k in zip(q_encoder.parameters(), k_encoder.parameters()):
        param_k.data.copy_(param_q.data) 
        param_k.requires_grad = False  # not update by gradient

    optimizer = torch.optim.Adam(q_encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # assign contrastive loss function
    if args.model == 'ContraWR':
        criterion = OurLoss(device, args.delta, args.sigma, args.T).to(device)
    elif args.model == 'MoCo':
        criterion = MoCo(device).to(device)
        queue_ptr, n_queue = 0, 4096
        queue = torch.tensor(np.random.rand(n_queue, args.n_dim), dtype=torch.float).to(device)
    elif args.model == 'SimCLR':
        criterion = SimCLR(device).to(device)
    elif args.model == 'BYOL':
        criterion = BYOL(device).to(device)
    elif args.model == 'SimSiam':
        criterion = SimSiam(device).to(device)

    # optimize
    Pretext(q_encoder, k_encoder, optimizer, args.epochs, criterion, pretext_loader, train_loader, test_loader)