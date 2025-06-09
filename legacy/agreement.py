import torch
import numpy as np
import os
import pickle
import collections

data_folder = os.path.join("/cluster/project/jbuhmann/choij/ISRUC-SLEEP/Subgroup_1")

index = np.arange(1, 10)

total = collections.Counter([])

all_labels1 = np.array([])
all_labels2 = np.array([])

for i in index:
    file_path1 = os.path.join(data_folder, str(i), str(i) + "_1.txt")
    file_path2 = os.path.join(data_folder, str(i), str(i) + "_2.txt")
    sleep_labels1 = np.loadtxt(file_path1, dtype=int)
    sleep_labels2 = np.loadtxt(file_path2, dtype=int)

    all_labels1 = np.append(all_labels1, sleep_labels1)
    all_labels2 = np.append(all_labels2, sleep_labels2)

labels = ["1", "2", "3", "4", "5"]

for label in labels:
    filtered_pairs = [(str(i), str(j)) for i, j in zip(all_labels1.tolist(), all_labels2.tolist()) if label in (str(i), str(j))]

    # Compute agreement: (a == b) for filtered pairs
    num_agreements = sum(1 for a, b in filtered_pairs if a == b)
    total_filtered = len(filtered_pairs)

    # Percentage of agreement
    agreement_percentage = (num_agreements / total_filtered * 100) if total_filtered > 0 else 0

    print(f"Label: {label}")
    print(f"Percentage of Agreement : {agreement_percentage:.2f}%")

total_agreement = [str(i)==str(j) for i, j in zip(all_labels1.tolist(), all_labels2.tolist())]

print("Overall agreement: ", sum(agreement)/(len(agreement)))