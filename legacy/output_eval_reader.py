import re
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams.update({'font.size': 25})

def find_highest_f1(file_path):
    highest_f1 = 0.0
    line_pattern = re.compile(r'\[Epoch\] : (\d+) .*?\[Evaluation Accuracy\] => ([0-9.]+).*?\[Evaluation Macro-F1\] => ([0-9.]+)')

    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            match = line_pattern.search(line)
            if match:
                f1 = float(match.group(3))
                if f1 > highest_f1:
                    highest_f1 = f1

    return highest_f1 * 100

def collect_f1_scores(directory):
    f1_scores = {}

    for filename in os.listdir(directory):
        if filename.endswith(".out"):
            # Filenames like: linear_prob_cassette_reduced_wavelet_neuro-b-mask-0.7-0.7.out
            match = re.search(r'mask-([0-9.]+)-([0-9.]+)\.out', filename)
            if match:
                time_ratio = float(match.group(1))
                freq_ratio = float(match.group(2))
                print(time_ratio, freq_ratio)
                file_path = os.path.join(directory, filename)
                f1 = find_highest_f1(file_path)
                print(f1)
                f1_scores[(freq_ratio, time_ratio)] = f1

    return f1_scores

def plot_f1_heatmap(f1_scores):
    freq_values = sorted(set(key[0] for key in f1_scores.keys()))
    time_values = sorted(set(key[1] for key in f1_scores.keys()))

    heatmap_data = np.zeros((len(freq_values), len(time_values)))

    for i, freq in enumerate(freq_values):
        for j, time in enumerate(time_values):
            val = f1_scores.get((freq, time), np.nan)
            if freq == 0.0 and time == 0.0:
                val = np.nan  # Force (0,0) to be NaN
            heatmap_data[i, j] = val
            # heatmap_data[i, j] = f1_scores.get((freq, time), np.nan)

    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, xticklabels=time_values, yticklabels=freq_values, annot=True, fmt=".2f", cmap="viridis")
    plt.xlabel("Time Mask Ratio")
    plt.ylabel("Freq Mask Ratio")
    plt.title("Highest F1 Scores Heatmap")
    plt.savefig("heatmap_tera_new.pdf")

# Example usage:
directory_path = "/cluster/project/jbuhmann/choij/sleep-stage-classification/output_log/tera_new_mask_ablation/"
scores = collect_f1_scores(directory_path)
plot_f1_heatmap(scores)