import os
import json
import csv
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score

def spearman_footrule_distance(s, t):
    assert len(s) == len(t)
    return (2.0 / len(s)**2) * np.sum(np.abs(np.asarray(s) - np.asarray(t)))

def kendall_tau_distance(s, t):
    numDiscordant = 0
    for i in range(len(s)):
        for j in range(i + 1, len(t)):
            if (s[i] < s[j] and t[i] > t[j]) or (s[i] > s[j] and t[i] < t[j]):
                numDiscordant += 1
    return 2.0 * numDiscordant / (len(s) * (len(s) - 1))

def log_compound_metrics(gt, knn_dict, accuracy, projection_enabled, log_dir):
    def convert_arrays_to_integers(array1, array2):
        string_to_int_mapping = {}
        next_integer = 0
        for string in array1:
            if string not in string_to_int_mapping:
                string_to_int_mapping[string] = next_integer
                next_integer += 1
        result_array1 = [string_to_int_mapping[string] for string in array1]
        result_array2 = [string_to_int_mapping.get(string, next_integer) for string in array2]
        return result_array1, result_array2

    spearman_total, kendall_total = 0.0, 0.0
    count = 0
    skipped = 0
    sample_distances = []

    for key in gt:
        if key not in knn_dict or not gt[key] or not knn_dict[key]:
            skipped += 1
            continue

        vec1, vec2 = convert_arrays_to_integers(gt[key], knn_dict[key])
        if len(vec1) != len(vec2):
            skipped += 1
            continue

        s = spearman_footrule_distance(vec1, vec2)
        k = kendall_tau_distance(vec1, vec2)

        sample_distances.append([key, s, k])
        spearman_total += s
        kendall_total += k
        count += 1

    avg_spearman = round(spearman_total / count, 4) if count else None
    avg_kendall = round(kendall_total / count, 4) if count else None

    os.makedirs(log_dir, exist_ok=True)

    with open(os.path.join(log_dir, "ranking_metrics.json"), "w") as f:
        json.dump({
            "projection": projection_enabled,
            "accuracy": accuracy,
            "spearman_footrule": avg_spearman,
            "kendall_tau": avg_kendall,
            "num_ranked_samples": count,
            "num_skipped_samples": skipped
        }, f, indent=4)

    with open(os.path.join(log_dir, "ranking_distances.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["test_id", "spearman_footrule", "kendall_tau"])
        writer.writerows(sample_distances)

    print(f"[INFO] Logged ranking metrics to {log_dir}/ranking_metrics.json")
    print(f"[INFO] Saved per-sample distances to {log_dir}/ranking_distances.csv")


def get_knn(reference_ids, reference_features, test_features, k=5):
    X = np.array(reference_features)
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(X)
    distances, indices = nbrs.kneighbors(np.array(test_features))
    neighbors = [[reference_ids[idx] for idx in row] for row in indices]
    return neighbors

def get_knn_classification(reference_features, reference_labels, test_features, test_labels, k=5):
    X = np.array(reference_features)
    y = np.array(reference_labels)
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(X)
    distances, indices = nbrs.kneighbors(np.array(test_features))

    predictions = []
    for i in range(len(indices)):
        neighbor_labels = y[indices[i]]
        values, counts = np.unique(neighbor_labels, return_counts=True)
        predictions.append(values[np.argmax(counts)])

    return accuracy_score(test_labels, predictions)
