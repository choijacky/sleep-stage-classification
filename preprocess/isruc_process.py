import mne
import numpy as np
import os
from multiprocessing import Process
import pickle
import argparse
import collections
import json

def pretext_train_test(root_folder, k, N, epoch_sec, dest_folder, reduced, standard):
    index_list = os.listdir(root_folder)
    
    train_index = np.random.choice(index_list, int(len(index_list) * 0.8), replace=False)   
    test_index = list(set(index_list) - set(train_index))
    #test_index = np.random.choice(list(set(all_index) - set(train_index) - set(val_index)), int(len(all_index) * 0.05), replace=False)

    #train_index = get_edf_from_id(train_index, rec_index)
    #test_index = get_edf_from_id(test_index, rec_index)

    print(train_index)
    print(test_index)

    print(list(set(train_index) & set(test_index)))

    print ('start train process!')
    sample_process(root_folder, k, N, epoch_sec, 'train', train_index, dest_folder, reduced, standard)
    print ()
    
    print ('start test process!')    
    sample_process(root_folder, k, N, epoch_sec, 'test', test_index, dest_folder, reduced, standard)
    print ()


def sample_process(root_folder, k, N, epoch_sec, train_test_val, index, dest_folder, reduced, standard):
    patient_dict = {}
    for i, j in enumerate(index):
        if i % N == k:
            if k == 0:
                print ('Progress: {} / {}'.format(i, len(index)))
            
            print(j)

            # load signal "X" part
            try:

                data = mne.io.read_raw_edf(os.path.join(root_folder, j, j + '.edf'), preload=True)
                ann1 = np.loadtxt(os.path.join(root_folder, j, j + '_1.txt'), dtype=int)
                ann2 = np.loadtxt(os.path.join(root_folder, j, j + '_2.txt'), dtype=int)

            except Exception as e:
                print(e)
                print(j)
                print(os.listdir(root_folder))

            filtered = data.copy().filter(l_freq=1, h_freq=50)
            raw_train = filtered.copy().resample(sfreq=100)

            epoch_duration = 30  # Each sleep stage label corresponds to a 30s epoch
            onset_times = np.arange(0, len(ann1) * epoch_duration, epoch_duration)

            stage_map = {
                0: "Sleep stage W",
                1: "Sleep stage 1",
                2: "Sleep stage 2",
                3: "Sleep stage 3",
                5: "Sleep stage R"
            }

            descriptions1 = [stage_map[label] for label in ann1]
            descriptions2 = [stage_map[label] for label in ann2]

            durations = np.diff(onset_times, append=onset_times[-1] + 30)

            ann1 = mne.Annotations(onset=onset_times, duration=durations, description=descriptions1)
            ann2 = mne.Annotations(onset=onset_times, duration=durations, description=descriptions2)


            # Extract the two EEG channels, shape (channels, time points)
            annotation_desc_2_event_id = {
                "Sleep stage W": 1,
                "Sleep stage 1": 2,
                "Sleep stage 2": 3,
                "Sleep stage 3": 4,
                "Sleep stage R": 5,
            }

            # keep last 30-min wake events before sleep and first 30-min wake events after
            # sleep and redefine annotations on raw data
            #ann.crop(ann[1]["onset"] - 30 * 60, ann[-2]["onset"] + 30 * 60)
            raw_train.set_annotations(ann1, emit_warning=False)

            events_train, _ = mne.events_from_annotations(
                raw_train, event_id=annotation_desc_2_event_id, chunk_duration=30.0
            )

            tmax = 30.0 - 1.0 / raw_train.info["sfreq"]  # tmax in included

            event_id = {
                "Sleep stage W": 1,
                "Sleep stage 1": 2,
                "Sleep stage 2": 3,
                "Sleep stage 3/4": 4,
                "Sleep stage R": 5,
            }

            epochs_train = mne.Epochs(
                raw=raw_train,
                events=events_train,
                event_id=event_id,
                tmin=0.0,
                tmax=tmax,
                baseline=None,
                on_missing='warn',
            )

            # Extract the single EEG channel, shape (time points, epoch length)
            X = epochs_train.get_data()[:, 2:8, :]

            #standardise the data
            X = (X - np.mean(X, axis=2, keepdims=True)) / (np.std(X, axis=2, keepdims=True))

            num_events = X.shape[0]
            labels1 = epochs_train.events[:, 2]

            # Repeat for 2nd annotator
            raw_train.set_annotations(ann2, emit_warning=False)

            events_train, _ = mne.events_from_annotations(
                raw_train, event_id=annotation_desc_2_event_id, chunk_duration=30.0
            )

            tmax = 30.0 - 1.0 / raw_train.info["sfreq"]  # tmax in included

            event_id = {
                "Sleep stage W": 1,
                "Sleep stage 1": 2,
                "Sleep stage 2": 3,
                "Sleep stage 3/4": 4,
                "Sleep stage R": 5,
            }

            epochs_train = mne.Epochs(
                raw=raw_train,
                events=events_train,
                event_id=event_id,
                tmin=0.0,
                tmax=tmax,
                baseline=None,
                on_missing='warn',
            )
            
            labels2 = epochs_train.events[:, 2]


            for idx in range(num_events):
                path = dest_folder + '/{}/'.format(train_test_val) + 'cassette-' + j + '-' + str(idx) + '.pkl'

                pickle.dump({'X': X[idx, :, :], 'y1': labels1[idx], 'y2': labels2[idx]}, open(path, 'wb'))
                

def get_edf_from_id(id_list, rec_index):
    edf_list = []
    for id in id_list:
        for i in range(1, 3):
            if id + str(i) in rec_index:
                edf_list.append(id + str(i))

    return edf_list



        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--windowsize', type=int, default=30, help="unit (seconds)")
    parser.add_argument('--multiprocess', type=int, default=8, help="How many processes to use")
    parser.add_argument('--root_folder', type=str, default="../dataset/physionet.org/files/sleep-edfx/1.0.0/sleep-cassette", help="folder with raw data")
    parser.add_argument('--dest_folder', type=str, default="../dataset", help="destination folder")
    parser.add_argument('--reduced', action='store_true')
    parser.add_argument('--standard', action='store_true')
    args = parser.parse_args()

    dataset_name = 'isruc_3'
    dest_folder = os.path.join(args.dest_folder, dataset_name)
    if not os.path.exists(dest_folder):
        os.makedirs(os.path.join(dest_folder, "train"))
        os.makedirs(os.path.join(dest_folder, "val"))
        os.makedirs(os.path.join(dest_folder, "test"))

    N, epoch_sec = args.multiprocess, args.windowsize
    print("DATASET: ", dataset_name)
    p_list = []
    for k in range(N):
        process = Process(target=pretext_train_test, args=(args.root_folder, k, N, epoch_sec, dest_folder, args.reduced, args.standard))
        process.start()
        p_list.append(process)

    for i in p_list:
        i.join()
