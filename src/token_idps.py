# Code adapted from On Embeddings for Numerical Features in Tabular Deep Learning
# https://github.com/Yura52/tabular-dl-num-embeddings/blob/main/bin/train4.py - commit e1401da
# subsample and and tree options not implemented

import torch
from tqdm import trange
import numpy as np
import pandas as pd
subsample = None
tree = None
device = 'cpu' #running full dataset of IDPs consumes about 50 GB of GPU, using CPU seems like a better option.
encoding = 'piecewise-linear'
count = 256 # count should be tuned in val set 



### Test using UKB data ###
seed = 0 
idps = pd.read_csv('../data/T1_struct_brainMRI_IDPs.csv')
# idp_map = pd.read_csv('~/../shared/PGx/sharing/T1mri.csv') # mapping file 
idps_filt = idps.dropna(thresh=len(idps) - 1, axis=1)
idps_filt = idps_filt.dropna(how= 'any', axis=0)
X_num = {'train':idps_filt[1:].values.astype('float')}
n_num_features = idps_filt.shape[1]
idx = torch.arange(idps_filt.shape[0]-1) # create embeddings for the full dataset... however, in full stack has to be dividied by train/val/test to avoid data leakage


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
# lib.dump_pickle(bin_edges, output / 'bin_edges.pickle')
bin_edges = [torch.tensor(x, dtype=torch.float32, device=device) for x in bin_edges]
print()

def encode(part, idx):
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


def apply_model(part, idx):
    encode(part, idx)

apply_model('train', idx)