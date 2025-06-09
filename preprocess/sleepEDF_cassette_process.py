import mne
import numpy as np
import os
from multiprocessing import Process
import pickle
import argparse
import json

def pretext_train_test(root_folder, k, N, epoch_sec, dest_folder, reduced, standard):
    """
    Split the dataset into train/val/test sets and process each set using sample_process.

    Parameters:
    - root_folder: str, path to the dataset folder
    - k: int, current process index
    - N: int, total number of processes
    - epoch_sec: int, epoch window size in seconds
    - dest_folder: str, path to store processed files
    - reduced: bool, whether to use reduced channels
    - standard: bool, whether to standardize data
    """
    index_list = os.listdir(root_folder)
    index_list.remove('index.html')
    rec_index = np.unique([path[:6] for path in index_list])
    all_index = np.unique([path[:5] for path in index_list])

    # Split data into train (90%), val (5%), test (5%)
    train_index = np.random.choice(all_index, int(len(all_index) * 0.9), replace=False)   
    test_index = np.random.choice(list(set(all_index) - set(train_index)), int(len(all_index) * 0.05), replace=False)
    val_index = list(set(all_index) - set(train_index) - set(test_index))

    train_index = get_edf_from_id(train_index, rec_index)
    val_index = get_edf_from_id(val_index, rec_index)
    test_index = get_edf_from_id(test_index, rec_index)

    print("Start processing datasets")
    sample_process(root_folder, k, N, epoch_sec, 'train', train_index, dest_folder, reduced, standard)
    sample_process(root_folder, k, N, epoch_sec, 'val', val_index, dest_folder, reduced, standard)
    sample_process(root_folder, k, N, epoch_sec, 'test', test_index, dest_folder, reduced, standard)

def sample_process(root_folder, k, N, epoch_sec, train_test_val, index, dest_folder, reduced, standard):
    """
    Process EEG recordings and extract epochs for the specified subset (train, val, or test).

    Parameters:
    - root_folder: str, path to dataset
    - k: int, current process index
    - N: int, total number of processes
    - epoch_sec: int, epoch window size in seconds
    - train_test_val: str, subset label
    - index: list of str, identifiers for the subset
    - dest_folder: str, path to store processed files
    - reduced: bool, unused flag in current code
    - standard: bool, unused flag in current code
    """
    for i, j in enumerate(index):
        if i % N == k:
            if k == 0:
                print(f"Progress: {i} / {len(index)}")
            print(f"Processing subject: {j}")

            try:
                data_file = [x for x in os.listdir(root_folder) if x[:6] == j and 'PSG' in x][0]
                ann_file = [x for x in os.listdir(root_folder) if x[:6] == j and 'Hypnogram' in x][0]
                data = mne.io.read_raw_edf(os.path.join(root_folder, data_file), preload=True)
                ann = mne.read_annotations(os.path.join(root_folder, ann_file))
            except Exception as e:
                print(f"Error loading data for {j}: {e}")
                continue

            # Map annotation labels to event IDs
            annotation_desc_2_event_id = {
                "Sleep stage W": 1,
                "Sleep stage 1": 2,
                "Sleep stage 2": 3,
                "Sleep stage 3": 4,
                "Sleep stage 4": 4,
                "Sleep stage R": 5,
            }
            data.set_annotations(ann, emit_warning=False)

            # Create events and epochs
            events_train, _ = mne.events_from_annotations(data, event_id=annotation_desc_2_event_id, chunk_duration=30.0)
            tmax = 30.0 - 1.0 / data.info["sfreq"]

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

            X = epochs_train.get_data()[:, :2, :]  # shape: (n_epochs, 2, time_points)
            X = (X - np.mean(X, axis=2, keepdims=True)) / (np.std(X, axis=2, keepdims=True))

            num_events = X.shape[0]
            labels = epochs_train.events[:, 2]

            for idx in range(num_events):
                # Save one file per EEG channel
                path1 = os.path.join(dest_folder, train_test_val, f'cassette-{j}-{idx}.pkl')
                path2 = os.path.join(dest_folder, train_test_val, f'cassette2-{j}-{idx}.pkl')

                pickle.dump({'X': X[idx, 0, :], 'y': labels[idx]}, open(path1, 'wb'))
                pickle.dump({'X': X[idx, 1, :], 'y': labels[idx]}, open(path2, 'wb'))

def get_edf_from_id(id_list, rec_index):
    """
    Given a list of subject IDs, match with recordings.

    Parameters:
    - id_list: list of str, subject IDs (e.g., SC400)
    - rec_index: list of str, actual recording IDs (e.g., SC4001)

    Returns:
    - List of matched EDF identifiers
    """
    edf_list = []
    for id in id_list:
        for i in range(1, 3):
            if id + str(i) in rec_index:
                edf_list.append(id + str(i))
    return edf_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--windowsize', type=int, default=30, help="Epoch window size in seconds")
    parser.add_argument('--multiprocess', type=int, default=8, help="Number of parallel processes")
    parser.add_argument('--root_folder', type=str, default="../dataset/physionet.org/files/sleep-edfx/1.0.0/sleep-cassette", help="Path to raw dataset")
    parser.add_argument('--dest_folder', type=str, default="../dataset", help="Path to store processed data")
    parser.add_argument('--reduced', action='store_true', help="Flag to use reduced EEG channels")
    parser.add_argument('--standard', action='store_true', help="Flag to standardize EEG data")
    args = parser.parse_args()

    dataset_name = 'cassette_reduced_2channels' if args.reduced else 'sleep_cassette_processed_s'
    dest_folder = os.path.join(args.dest_folder, dataset_name)

    # Create destination directories if not exist
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(dest_folder, split), exist_ok=True)

    print(f"Processing dataset: {dataset_name}")

    N, epoch_sec = args.multiprocess, args.windowsize
    p_list = []

    # Launch multiprocessing
    for k in range(N):
        process = Process(target=pretext_train_test, args=(args.root_folder, k, N, epoch_sec, dest_folder, args.reduced, args.standard))
        process.start()
        p_list.append(process)

    for i in p_list:
        i.join()