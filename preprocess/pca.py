print("Start script")
import sys
import os

sys.path.extend([os.path.abspath('.'), os.path.abspath('..')])

from data.data_loader import SLEEPCALoader, DOD
import numpy as np
from sklearn.decomposition import KernelPCA, PCA
import torch
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_folder', type=str, default="../dataset/small_cassette_processed", help="folder with data")
    args = parser.parse_args()

    modes = ["poly", "rbf", "cosine"]

#    for mode in modes:

    root_dir = os.path.join(args.root_folder, "train")
    dest_dir = os.path.join(args.root_folder, "pca_train")

    indices = os.listdir(root_dir)

    #dataset = SLEEPCALoader(indices, root_dir, 1, False)
    dataset = DOD(indices, root_dir, 1, False, 1, False, False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=len(indices), shuffle=False)

    data_iter = iter(dataloader)
    time_series, _ = next(data_iter)
    time_series = time_series.squeeze().numpy()

    pca_dim = min(time_series.shape[0], time_series.shape[1])
    print("PCA")
    #pca = KernelPCA(n_components=pca_dim, kernel='rbf')  # You can adjust the number of components
    pca = PCA(n_components=pca_dim)
    print("PCA done")
    pca.fit(time_series)

    if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
    # print(pca.eigenvectors_.shape)
    # print(pca.eigenvalues_.shape)
    #evals = pca.eigenvalues_/(np.sum(pca.eigenvalues_))

    np.save(os.path.join(dest_dir, "pc_matrix_pca.npy"), pca.components_)
    np.save(os.path.join(dest_dir, "eigenvalues_ratio_ipca.npy"), pca.explained_variance_ratio_)

