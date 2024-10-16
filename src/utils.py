import torch
from tqdm import trange
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import subprocess
from collections import Counter
import os

### General Utils ###
def select_gpu(num_gpus=1,verbose=False):
    # Run the nvidia-smi command to get GPU information
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader'], capture_output=True, text=True)

    # Parse the output to get GPU index and memory usage
    gpu_info = result.stdout.strip().split('\n')
    gpu_info = [info.split(',') for info in gpu_info]
    gpu_info = [(info[0], int(info[1].split()[0])) for info in gpu_info]

    # Sort the GPU info based on memory usage
    sorted_gpu_info = sorted(gpu_info, key=lambda x: x[1])

    if verbose:
        # Print the GPU info with least memory usage
        for gpu in sorted_gpu_info:
            print(f"GPU {gpu[0]}: Memory Usage {gpu[1]} MB")
    
    # Select the first num_gpus GPUs with least memory usage
    selected_gpus = [gpu[0] for gpu in sorted_gpu_info[:num_gpus]]
    return selected_gpus


### Results Utils ###
def plot_training_curves(train_losses, val_losses, val_acc, path):
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # Plot the training loss in the first subplot
    ax1.plot(train_losses, label='Train Loss', color='blue')
    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Epochs')
    ax1.legend()
    ax1.set_title('Training Loss')

    # Plot the validation loss in the second subplot
    ax2.plot(val_losses, label='Validation Loss', color='orange')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epochs')
    ax2.legend()
    ax2.set_title('Validation Loss')

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(path, 'training_curves.pdf'))
    plt.close()
    
    ### Add accuracy plot
    if val_acc is not None:
        plt.figure(figsize=(10, 10))
        plt.plot(val_acc, label='Validation Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epochs')
        plt.legend()
        plt.savefig(os.path.join(path,'validation_accuracy.pdf'))
        plt.close()

# Plot AUC-ROC curve
def plot_roc_curve(test_preds, test_labels, path):
    plt.figure(figsize=(10, 10))
    for i in range(len(test_preds)):
        fpr, tpr, _ = roc_curve(test_labels[i], test_preds[i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'ROC curve {i} (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Test Set ROC')
    plt.legend(loc="lower right")
    plt.savefig(path)
    plt.close()

# plot precision-recall curve
def plot_precision_recall_curve(test_preds, test_labels, path):
    plt.figure(figsize=(10, 10))
    for i in range(len(test_preds)):
        precision, recall, _ = precision_recall_curve(test_labels[i], test_preds[i])
        plt.plot(recall, precision, label=f'PR curve {i}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Test Set Precision-Recall')
    plt.legend(loc="lower right")
    plt.savefig(path)
    plt.close()

def plot_uniqueness(uniqueness, path):
    plt.figure(figsize=(10, 10))
    plt.plot(uniqueness)
    plt.ylabel('# of unique predictions')
    plt.xlabel('Batches')
    plt.title('Uniqueness')
    plt.savefig(path)
    plt.close()

def calculate_acc(seq,snp_id,idp,idp_id, probs_seq, probs_idp, dd_idp, dd, idp_id_map, snp_id_map):
    global master_pair_freq_dict
    pairs = []
    correct = 0.0
    for i,j in enumerate(probs_seq):
        if idp_id_map[idp_id[j]] in dd[snp_id_map[snp_id[i]]][0]:
            correct += 1
            pairs.append( (dd_idp[idp_id_map[idp_id[i]]][0][0], idp_id_map[idp_id[j]]) ) # flipped from Diego order so its always (SNP, IDP)

    for i,j in enumerate(probs_idp):
        if snp_id_map[snp_id[j]] in dd_idp[idp_id_map[idp_id[i]]][0]:
            correct += 1
            pairs.append( (snp_id_map[snp_id[j]], dd[snp_id_map[snp_id[i]]][0][0]) )

    acc = correct / len(probs_seq) /2 
    return acc

def calculate_decile_acc(seq, snp_id, idp, idp_id, probs_seq, probs_idp, dd_idp, dd, idp_id_map, snp_id_map, decile):
    # global master_pair_freq_dict
    # pairs = []
    acc = 0.0
    total_count = len(probs_seq) + len(probs_idp)
    
    for i, prob_seq in enumerate(probs_seq):
        # Calculate the top decile of the predicted values
        top_decile_indices = np.where(pd.qcut(prob_seq, 10, labels=False, duplicates='drop') == decile)[0] #top decile is 9
        top_decile_idps = {idp_id_map[idp_id[idx]] for idx in top_decile_indices}

        # Obtain the true associated values for the SNP
        true_values = set(dd[snp_id_map[snp_id[i]]][0])

        # Calculate the overlap
        overlap = len(true_values.intersection(top_decile_idps))
        acc += overlap / len(true_values) if true_values else 0

        # pairs.append( (dd_idp[idp_id_map[idp_id[i]]][0][0], idp_id_map[idp_id[j]]) )

    for i, prob_idp in enumerate(probs_idp):
        # Calculate the top decile of the predicted values
        top_decile_indices = np.where(pd.qcut(prob_idp, 10, labels=False, duplicates='drop') == decile)[0]
        top_decile_snps = {snp_id_map[snp_id[idx]] for idx in top_decile_indices}

        # Obtain the true associated values for the IDP
        true_values = set(dd_idp[idp_id_map[idp_id[i]]][0])

        # Calculate the overlap
        overlap = len(true_values.intersection(top_decile_snps))
        acc += overlap / len(true_values) if true_values else 0

        # pairs.append( (snp_id_map[snp_id[j]], dd[snp_id_map[snp_id[i]]][0][0]) )

    # counter = dict(Counter(pairs))

    # for key, value in counter.items():
    #     if key not in master_pair_freq_dict:
    #         master_pair_freq_dict[key] = []
    #         master_pair_freq_dict[key].append(value)
    #     else:
    #         master_pair_freq_dict[key].append(value)

    return acc / total_count / 2

### Training utils ###
# Obtained from https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

### Tokenization Utils ###
# Code adapted from On Embeddings for Numerical Features in Tabular Deep Learning
# https://github.com/Yura52/tabular-dl-num-embeddings/blob/main/bin/train4.py - commit e1401da
# subsample and and tree options not implemented
def idp_tokenization(device,count,seed,idx,X_num):
    subsample = None
    tree = None
    encoding = 'piecewise-linear'
    n_num_features = 139

    print('\nRunning bin-based encoding...')
    assert X_num is not None
    bin_edges = []
    _bins = {x: [] for x in X_num}
    _bin_values = {x: [] for x in X_num}
    rng = np.random.default_rng(seed)
    for feature_idx in trange(n_num_features):
        train_column = X_num['train'][:, feature_idx].copy()
        if subsample is not None:
            subsample_n = (
                subsample
                if isinstance(subsample, int)
                else int(subsample * D.size('train'))
            )
            subsample_idx = rng.choice(len(train_column), subsample_n, replace=False)
            train_column = train_column[subsample_idx]
        else:
            subsample_idx = None

        if tree is not None:
            _y = D.y['train']
            if subsample_idx is not None:
                _y = _y[subsample_idx]
            tree = (
                (DecisionTreeRegressor if D.is_regression else DecisionTreeClassifier)(
                    max_leaf_nodes=C.bins.count, **C.bins.tree
                )
                .fit(train_column.reshape(-1, 1), D.y['train'])
                .tree_
            )
            del _y
            tree_thresholds = []
            for node_id in range(tree.node_count):
                # the following condition is True only for split nodes
                # See https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html#sphx-glr-auto-examples-tree-plot-unveil-tree-structure-py
                if tree.children_left[node_id] != tree.children_right[node_id]:
                    tree_thresholds.append(tree.threshold[node_id])
            tree_thresholds.append(train_column.min())
            tree_thresholds.append(train_column.max())
            bin_edges.append(np.array(sorted(set(tree_thresholds))))
        else:
            feature_n_bins = min(count, len(np.unique(train_column)))
            quantiles = np.linspace(
                0.0, 1.0, feature_n_bins + 1
            )  # includes 0.0 and 1.0
            bin_edges.append(np.unique(np.quantile(train_column, quantiles)))

        for part in X_num:
            _bins[part].append(
                np.digitize(
                    X_num[part][:, feature_idx],
                    np.r_[-np.inf, bin_edges[feature_idx][1:-1], np.inf],
                ).astype(np.int32)
                - 1
            )
            if encoding == 'binary':
                _bin_values[part].append(np.ones_like(X_num[part][:, feature_idx]))
            elif encoding == 'piecewise-linear':
                feature_bin_sizes = (
                    bin_edges[feature_idx][1:] - bin_edges[feature_idx][:-1]
                )
                part_feature_bins = _bins[part][feature_idx]
                _bin_values[part].append(
                    (
                        X_num[part][:, feature_idx]
                        - bin_edges[feature_idx][part_feature_bins]
                    )
                    / feature_bin_sizes[part_feature_bins]
                )
            else:
                assert encoding == 'one-blob'

    n_bins = max(map(len, bin_edges)) - 1

    bins = {
        k: torch.as_tensor(np.stack(v, axis=1), dtype=torch.int64, device=device)
        for k, v in _bins.items()
    }
    del _bins

    bin_values = (
        {
            k: torch.as_tensor(np.stack(v, axis=1), dtype=torch.float32, device=device)
            for k, v in _bin_values.items()
        }
        if _bin_values['train']
        else None
    )
    del _bin_values
    bin_edges = [torch.tensor(x, dtype=torch.float32, device=device) for x in bin_edges]
    print()

    assert bins is not None
    assert bins is not None
    assert n_bins is not None

    if encoding == 'one-blob':
        assert bin_edges is not None
        assert X_num is not None
        assert C.bins.one_blob_gamma is not None
        x = torch.zeros(
            len(idx), D.n_num_features, n_bins, dtype=torch.float32, device=device
        )
        for i in range(D.n_num_features):
            n_bins_i = len(bin_edges[i]) - 1
            bin_left_edges = bin_edges[i][:-1]
            bin_right_edges = bin_edges[i][1:]
            kernel_scale = 1 / (n_bins_i ** C.bins.one_blob_gamma)
            cdf_values = [
                0.5
                * (
                    1
                    + torch.erf(
                        (edges[None] - X_num[part][idx, i][:, None])
                        / (kernel_scale * 2 ** 0.5)
                    )
                )
                for edges in [bin_left_edges, bin_right_edges]
            ]
            x[:, i, :n_bins_i] = cdf_values[1] - cdf_values[0]

    else:
        assert bin_values is not None
        bins_ = bins[part][idx]
        bin_mask_ = torch.eye(n_bins, device=device)[bins_]
        x = bin_mask_ * bin_values[part][idx, ..., None]
        previous_bins_mask = torch.arange(n_bins, device=device)[None, None].repeat(
            len(idx), n_num_features, 1
        ) < bins_.reshape(len(idx), n_num_features, 1)
        x[previous_bins_mask] = 1.0

    return x