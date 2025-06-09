import mne
import numpy as np
import os
from multiprocessing import Process
import pickle
import argparse
import json
from matplotlib.colors import ListedColormap, Normalize
import matplotlib.pyplot as plt
from lspopt import spectrogram_lspopt
import yasa
import pandas as pd
from wavelet import cwt


def pretext_train_test(root_folder, k, N, epoch_sec, dest_folder, reduced, technique):
    index_list = os.listdir(root_folder)
    index_list.remove('index.html')
    rec_index = np.unique([path[:6] for path in index_list])
    all_index = np.unique([path[:5] for path in index_list])
    
    #train_index = np.random.choice(all_index, int(len(all_index) * 0.9), replace=False)   
    #val_index = np.random.choice(list(set(all_index) - set(train_index)), int(len(all_index) * 0.05), replace=False)
    #test_index = list(set(all_index) - set(train_index) - set(val_index))
    #test_index = np.random.choice(list(set(all_index) - set(train_index) - set(val_index)), int(len(all_index) * 0.05), replace=False)

    train_index = get_edf_from_id(all_index, rec_index)
    #val_index = get_edf_from_id(val_index, rec_index)
    #test_index = get_edf_from_id(test_index, rec_index)

    print(train_index)
    # print(val_index)
    # print(test_index)

    # print(list(set(train_index) & set(val_index)))
    # print(list(set(test_index) & set(val_index)))
    # print(list(set(train_index) & set(test_index)))

    print ('start pretext process!')
    sample_process(root_folder, k, N, epoch_sec, 'train', train_index, dest_folder, reduced, technique)
    print ()
    
    # print ('start train process!')
    # sample_process(root_folder, k, N, epoch_sec, 'val', val_index, dest_folder, reduced, technique)
    # print ()
    
    # print ('start test process!')    
    # sample_process(root_folder, k, N, epoch_sec, 'test', test_index, dest_folder, reduced, technique)
    # print ()


def sample_process(root_folder, k, N, epoch_sec, train_test_val, index, dest_folder, reduced, technique):
    patient_dict = {}
    for i, j in enumerate(index):
        if i % N == k:
            if k == 0:
                print ('Progress: {} / {}'.format(i, len(index)))
            
            print(j)

            # load signal "X" part
            try:
                data = mne.io.read_raw_edf(root_folder + '/' + list(filter(lambda x: (x[:6] == j) and ('PSG' in x), os.listdir(root_folder)))[0], preload=True)
                ann = mne.read_annotations(root_folder + '/' + list(filter(lambda x: (x[:6] == j) and ('Hypnogram' in x), os.listdir(root_folder)))[0])
            except Exception as e:
                print(e)
                print(j)
                print(os.listdir(root_folder))

            annotation_desc_2_event_id = {
                "Sleep stage W": 1,
                "Sleep stage 1": 2,
                "Sleep stage 2": 3,
                "Sleep stage 3": 4,
                "Sleep stage 4": 4,
                "Sleep stage R": 5,
            }

            # keep last 30-min wake events before sleep and first 30-min wake events after
            # sleep and redefine annotations on raw data
            ann.crop(ann[1]["onset"] - 30 * 60, ann[-2]["onset"] + 30 * 60)
            data.set_annotations(ann, emit_warning=False)

            events_train, _ = mne.events_from_annotations(
                data, event_id=annotation_desc_2_event_id, chunk_duration=30.0
            )

            tmax = 30.0 - 1.0 / data.info["sfreq"]  # tmax in included

            event_id = {
                "Sleep stage W": 1,
                "Sleep stage 1": 2,
                "Sleep stage 2": 3,
                "Sleep stage 3/4": 4,
                "Sleep stage R": 5,
            }

            epochs_train = mne.Epochs(
                raw=data,
                events=events_train,
                event_id=event_id,
                tmin=0.0,
                tmax=tmax,
                baseline=None,
                on_missing='warn',
            )

            # Extract the single EEG channel, shape (time points, epoch length)
            X = epochs_train.get_data()[:, 0, :]

            #standardise the data
            X = (X - np.mean(X)) / (np.std(X))

            num_events = X.shape[0]
            labels = epochs_train.events[:, 2]

            for idx in range(num_events):
                path = dest_folder + '/{}/'.format(train_test_val) + 'cassette-' + j + '-' + str(idx) + '-' + str(labels[idx]) + '.npy'

                if technique == "spectrogram":
                    Sxx, _, _  = plot_spectrogram(X[idx, :], 100, win_sec=0.5, fmax=40)
                    with open(path, 'wb') as f:
                        np.save(f, Sxx)
                
                elif technique == "wavelet_transform":
                    wavelet = cwt(1 / 100, 3000, device='cpu')
                    waveshape = (30, 60)
                    s = wavelet(X[idx, :], waveshape[1]).numpy()
                    with open(path, 'wb') as f:
                        np.save(f, s)

                else:
                    raise Exception("No technique named ", technique)
                

def get_edf_from_id(id_list, rec_index):
    edf_list = []
    for id in id_list:
        for i in range(1, 3):
            if id + str(i) in rec_index:
                edf_list.append(id + str(i))

    return edf_list

def plot_spectrogram(
    data,
    sf,
    hypno=None,
    win_sec=30,
    fmin=0.5,
    fmax=25,
    trimperc=2.5,
    cmap="RdBu_r",
    vmin=None,
    vmax=None,
    **kwargs,
):
    """
    Plot a full-night multi-taper spectrogram, optionally with the hypnogram on top.

    For more details, please refer to the `Jupyter notebook
    <https://github.com/raphaelvallat/yasa/blob/master/notebooks/10_spectrogram.ipynb>`_

    .. versionadded:: 0.1.8

    Parameters
    ----------
    data : :py:class:`numpy.ndarray`
        Single-channel EEG data. Must be a 1D NumPy array.
    sf : float
        The sampling frequency of data AND the hypnogram.
    hypno : array_like
        Sleep stage (hypnogram), optional.

        The hypnogram must have the exact same number of samples as ``data``.
        To upsample your hypnogram, please refer to :py:func:`yasa.hypno_upsample_to_data`.

        .. note::
            The default hypnogram format in YASA is a 1D integer
            vector where:

            - -2 = Unscored
            - -1 = Artefact / Movement
            - 0 = Wake
            - 1 = N1 sleep
            - 2 = N2 sleep
            - 3 = N3 sleep
            - 4 = REM sleep
    win_sec : int or float
        The length of the sliding window, in seconds, used for multitaper PSD
        calculation. Default is 30 seconds. Note that ``data`` must be at least
        twice longer than ``win_sec`` (e.g. 60 seconds).
    fmin, fmax : int or float
        The lower and upper frequency of the spectrogram. Default 0.5 to 25 Hz.
    trimperc : int or float
        The amount of data to trim on both ends of the distribution when
        normalizing the colormap. This parameter directly impacts the
        contrast of the spectrogram plot (higher values = higher contrast).
        Default is 2.5, meaning that the min and max of the colormap
        are defined as the 2.5 and 97.5 percentiles of the spectrogram.
    cmap : str
        Colormap. Default to 'RdBu_r'.
    vmin : int or float
        The lower range of color scale. Overwrites ``trimperc``
    vmax : int or float
        The upper range of color scale. Overwrites ``trimperc``
    **kwargs : dict
        Other arguments that are passed to :py:meth:`yasa.Hypnogram.plot_hypnogram`.

    Returns
    -------
    fig : :py:class:`matplotlib.figure.Figure`
        Matplotlib Figure

    Examples
    --------
    1. Full-night multitaper spectrogram on Cz, no hypnogram

    .. plot::

        >>> import yasa
        >>> import numpy as np
        >>> # In the next 5 lines, we're loading the data from GitHub.
        >>> import requests
        >>> from io import BytesIO
        >>> r = requests.get('https://github.com/raphaelvallat/yasa/raw/master/notebooks/data_full_6hrs_100Hz_Cz%2BFz%2BPz.npz', stream=True)
        >>> npz = np.load(BytesIO(r.raw.read()))
        >>> data = npz.get('data')[0, :]
        >>> sf = 100
        >>> fig = yasa.plot_spectrogram(data, sf)

    2. Full-night multitaper spectrogram on Cz with the hypnogram on top

    .. plot::

        >>> import yasa
        >>> import numpy as np
        >>> # In the next lines, we're loading the data from GitHub.
        >>> import requests
        >>> from io import BytesIO
        >>> r = requests.get('https://github.com/raphaelvallat/yasa/raw/master/notebooks/data_full_6hrs_100Hz_Cz%2BFz%2BPz.npz', stream=True)
        >>> npz = np.load(BytesIO(r.raw.read()))
        >>> data = npz.get('data')[0, :]
        >>> sf = 100
        >>> # Load the 30-sec hypnogram and upsample to data
        >>> hypno = np.loadtxt('https://raw.githubusercontent.com/raphaelvallat/yasa/master/notebooks/data_full_6hrs_100Hz_hypno_30s.txt')
        >>> hypno = yasa.hypno_upsample_to_data(hypno, 1/30, data, sf)
        >>> fig = yasa.plot_spectrogram(data, sf, hypno, cmap='Spectral_r')
    """
    from yasa.hypno import Hypnogram, hypno_int_to_str  # Avoiding circular imports

    # Increase font size while preserving original
    old_fontsize = plt.rcParams["font.size"]
    plt.rcParams.update({"font.size": 18})

    # Safety checks
    assert isinstance(data, np.ndarray), "`data` must be a 1D NumPy array."
    assert isinstance(sf, (int, float)), "`sf` must be int or float."
    assert data.ndim == 1, "`data` must be a 1D (single-channel) NumPy array."
    assert isinstance(win_sec, (int, float)), "`win_sec` must be int or float."
    assert isinstance(fmin, (int, float)), "`fmin` must be int or float."
    assert isinstance(fmax, (int, float)), "`fmax` must be int or float."
    assert fmin < fmax, "`fmin` must be strictly inferior to `fmax`."
    assert fmax < sf / 2, "`fmax` must be less than Nyquist (sf / 2)."
    assert isinstance(vmin, (int, float, type(None))), "`vmin` must be int, float, or None."
    assert isinstance(vmax, (int, float, type(None))), "`vmax` must be int, float, or None."
    if vmin is not None:
        assert isinstance(vmax, (int, float)), "`vmax` must be int or float if `vmin` is provided."
    if vmax is not None:
        assert isinstance(vmin, (int, float)), "`vmin` must be int or float if `vmax` is provided."
    if hypno is not None:
        assert hypno.size == data.size, "`hypno` must have the same number of samples as `data`."

    # Calculate multi-taper spectrogram
    nperseg = int(win_sec * sf)
    assert data.size > 2 * nperseg, "`data` length must be at least 2 * `win_sec`."
    f, t, Sxx = spectrogram_lspopt(data, sf, nperseg=nperseg, noverlap=0)
    Sxx = 10 * np.log10(Sxx)  # Convert uV^2 / Hz --> dB / Hz

    # Select only relevant frequencies (up to 30 Hz)
    good_freqs = np.logical_and(f >= fmin, f <= fmax)
    Sxx = Sxx[good_freqs, :]
    f = f[good_freqs]
    t /= 3600  # Convert t to hours
    return Sxx, t, f



        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--windowsize', type=int, default=30, help="unit (seconds)")
    parser.add_argument('--multiprocess', type=int, default=8, help="How many processes to use")
    parser.add_argument('--root_folder', type=str, default="../dataset/physionet.org/files/sleep-edfx/1.0.0/sleep-cassette", help="folder with raw data")
    parser.add_argument('--dest_folder', type=str, default="../dataset", help="destination folder")
    parser.add_argument('--reduced', action='store_true')
    parser.add_argument('--standard', action='store_true')
    parser.add_argument('--time_freq_technique', type=str, default="spectrogram")
    parser.add_argument('--dataset_name', type=str, default="sleep_edfx_dataset")
    args = parser.parse_args()

    dataset_name = args.dataset_name
    
    dest_folder = os.path.join(args.dest_folder, dataset_name)
    if not os.path.exists(dest_folder):
        os.makedirs(os.path.join(dest_folder, "train"))
        os.makedirs(os.path.join(dest_folder, "val"))
        os.makedirs(os.path.join(dest_folder, "test"))

    N, epoch_sec = args.multiprocess, args.windowsize
    print("DATASET: ", dataset_name)
    yasa.plot_spectrogram = plot_spectrogram
    p_list = []
    for k in range(N):
        process = Process(target=pretext_train_test, args=(args.root_folder, k, N, epoch_sec, dest_folder, args.reduced, args.time_freq_technique))
        process.start()
        p_list.append(process)

    for i in p_list:
        i.join()
