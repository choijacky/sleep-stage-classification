import torch
import numpy as np
import os
import pickle
import collections
import re

datasets = ["dodh-processed-full"]

for dataset  in datasets:
    print(dataset)

    data_folder = os.path.join("/cluster/project/jbuhmann/choij/sleep-stage-classification/dataset", dataset)

    split = ["train", "test"]#, "val"]

    patient_ids = []

    for s in split:
        root_folder = os.path.join(data_folder, s)
        print(s)
        list_id = os.listdir(root_folder)
        labels = []
        split_ids = []

        #list_id = [id.split('.')[0] for id in list_id if not "-SC" in id]

        id_set = set()

        for id in list_id:
            # match = re.search(r'cassette-(.*)-\d+$', id)
            # result = match.group(1)

            # id_set.add(result)

            path = os.path.join(root_folder, id)
            patient_id = id.split("-")[1]
            if not patient_id in split_ids:
                split_ids.append(patient_id)

            y = int(path.split('/')[-1].split('.')[0].split('-')[-1])

            sample = pickle.load(open(path, 'rb'))
            y = sample['y1']
            labels.append(y)

        # patient_ids.append(split_ids)

        print(dict(collections.Counter(labels)))

        #print(id_set)

    # print(set(patient_ids[0]) & set(patient_ids[1]))
    # print(set(patient_ids[0]) & set(patient_ids[2]))
    # print(set(patient_ids[2]) & set(patient_ids[1]))

    #print(patient_ids[0])
    #print(patient_ids[1])
    # print(patient_ids[2])
