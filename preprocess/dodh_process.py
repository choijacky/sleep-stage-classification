import mne
import numpy as np
import os
from multiprocessing import Process
import pickle
import argparse
from collections import Counter
import json
from sklearn.model_selection import KFold

def pretext_train_test(root_folder, k, N, epoch_sec, dest_folder, reduced, standard):
    index_list = os.listdir(os.path.join(root_folder, 'edf', 'dodh'))
    
    train_index = np.random.choice(index_list, int(len(index_list) * 0.8), replace=False)   
    test_index = list(set(index_list) - set(train_index))

    print(train_index)
    print(test_index)

    print ('start train process!')
    sample_process(root_folder, k, N, epoch_sec, 'train', index_list, dest_folder, reduced, standard)
    print ()


def sample_process(root_folder, k, N, epoch_sec, train_test_val, index, dest_folder, reduced, standard):
    patient_dict = {}
    for i, j in enumerate(index):
        if i % N == k:
            if k == 0:
                print ('Progress: {} / {}'.format(i, len(index)))
            
            print(j)

            filename = j.split('.')[0]
            scorer_labels = []

            # load signal "X" part
            try:

                data = mne.io.read_raw_edf(os.path.join(root_folder, 'edf', 'dodh', j), preload=True)
                for i in range(1, 6):
                    
                    with open(os.path.join(root_folder, 'annotation', 'dodh', 'scorer_' + str(i), filename + ".json"), "r") as f:
                        data_list = json.load(f)
                        scorer_labels.append(data_list)
                
            except Exception as e:
                print(e)
                print(j)
                print(os.listdir(root_folder))

            filtered = data.copy().filter(l_freq=1, h_freq=50)
            raw_train = filtered.copy().resample(sfreq=100)

            epoch_duration = 30  # Each sleep stage label corresponds to a 30s epoch
            onset_times = np.arange(0, len(scorer_labels[0]) * epoch_duration, epoch_duration)

            stage_map = {
                -1: "?",
                0: "Sleep stage W",
                1: "Sleep stage 1",
                2: "Sleep stage 2",
                3: "Sleep stage 3",
                4: "Sleep stage R"
            }

            durations = np.diff(onset_times, append=onset_times[-1] + 30)

            
            anns = []
            for i in range(5):
                description = [stage_map[label] for label in scorer_labels[i]]

                anns.append(mne.Annotations(onset=onset_times, duration=durations, description=description))


            # Extract the two EEG channels, shape (channels, time points)
            annotation_desc_2_event_id = {
                "?": -1,
                "Sleep stage W": 0,
                "Sleep stage 1": 1,
                "Sleep stage 2": 2,
                "Sleep stage 3": 3,
                "Sleep stage R": 4,
            }

            event_id = {
                "?": -1,
                "Sleep stage W": 0,
                "Sleep stage 1": 1,
                "Sleep stage 2": 2,
                "Sleep stage 3/4": 3,
                "Sleep stage R": 4,
            }

            # keep last 30-min wake events before sleep and first 30-min wake events after
            # sleep and redefine annotations on raw data
            #ann.crop(ann[1]["onset"] - 30 * 60, ann[-2]["onset"] + 30 * 60)

            labels = []

            for i in range(5):
                raw_train.set_annotations(anns[i], emit_warning=False)

                events_train, _ = mne.events_from_annotations(
                    raw_train, event_id=annotation_desc_2_event_id, chunk_duration=30.0
                )

                tmax = 30.0 - 1.0 / raw_train.info["sfreq"]  # tmax in included
                
                epochs_train = mne.Epochs(
                    raw=raw_train,
                    events=events_train,
                    event_id=event_id,
                    tmin=0.0,
                    tmax=tmax,
                    baseline=None,
                    on_missing='warn',
                )

                labels.append(epochs_train.events[:, 2])

                # Extract the single EEG channel, shape (time points, epoch length)
                X = epochs_train.get_data()

            #standardise the data
            X = (X - np.mean(X, axis=2, keepdims=True)) / (np.std(X, axis=2, keepdims=True))

            num_events = X.shape[0]

            assert len(labels[0]) == len(labels[1]) == len(labels[2]) == len(labels[3]) == len(labels[4]) == num_events
            

            for idx in range(num_events):

                if labels[0][idx] == -1 or scorer_labels[1][idx] == -1 or scorer_labels[2][idx] == -1 or scorer_labels[3][idx] == -1 or scorer_labels[4][idx] == -1:
                    print("skipped epoch")
                    continue

                elif not (scorer_labels[0][idx] == scorer_labels[1][idx] == scorer_labels[2][idx] == scorer_labels[3][idx] == scorer_labels[4][idx]):
                    print("skipped epoch")
                    continue

                # votes = [scorer_labels[0][idx], scorer_labels[1][idx], scorer_labels[2][idx], scorer_labels[3][idx], scorer_labels[4][idx]]
                # majority = majority_vote(votes)


                # if majority == -1:
                #     print("skipped epoch")
                #     continue

                path = dest_folder + '/{}/'.format(train_test_val) + 'cassette-' + filename + '-' + str(idx) + '.pkl'

                pickle.dump({'X': X[idx, :, :],'y1': labels[0][idx], 'y2': labels[1][idx], 'y3': labels[2][idx], 'y4': labels[3][idx], 'y5': labels[4][idx]}, open(path, 'wb'))
                #pickle.dump({'X': X[idx, :, :], 'y1': majority}, open(path, 'wb'))
                #pickle.dump({'X': X[idx, :, :], 'y': scorer_labels[0][idx]}, open(path, 'wb'))

def majority_vote(labels):
    soft_agreement = {
        0: 0.884986,
        1: 0.908218,
        2: 0.917917,
        3: 0.838457,
        4: 0.916462,
    }

    # Count how many times each label appears
    label_counts = Counter(labels)
    max_votes = max(label_counts.values())
    
    # Find all labels that have the highest vote count (to detect ties)
    top_labels = [label for label, count in label_counts.items() if count == max_votes]
    
    if len(top_labels) == 1:
        # No tie
        return top_labels[0]
    else:
        # Tie: select the label of the scorer with the highest soft agreement
        max_agreement = -1
        selected_label = None
        for scorer_index, label in enumerate(labels):
            if label in top_labels:
                agreement = soft_agreement.get(scorer_index, 0)
                if agreement > max_agreement:
                    max_agreement = agreement
                    selected_label = label
        return selected_label


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

    dataset_name = 'dodh-processed-wavelet-full'
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
