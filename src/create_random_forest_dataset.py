from __future__ import print_function, division, absolute_import
import os
import pandas as pd
import h5py
import numpy as np

ppi_path = '/home/giovenko/DeepLearning/ppi_with_lstm'

# Dataset containing the normalized protein fingerprints for all the proteins
# in Florian's dataset.
norm_prot_fps = pd.read_hdf(
    os.path.join(ppi_path, 'output/normalized_protein_fp.hdf5')
)

# Dataset containing the protein ID and the sequence for the pairs that pass
# the filtering, i.e., which have a length between 5 and 500 and without 'U's.
filtered_pairs = pd.read_hdf(
    os.path.join(ppi_path, 'output/filtered_ppi_dataset_500.hdf5')
)

x = np.hstack([
    norm_prot_fps.loc[filtered_pairs.uid1].values.astype('float32'),
    norm_prot_fps.loc[filtered_pairs.uid2].values.astype('float32')])
y = filtered_pairs.interaction.values

n_test = 10000
p_val = 0.05
n_train = int((x.shape[0] - n_test) * (1 - p_val))
n_val = int((x.shape[0] - n_test) * p_val)

x_tr, y_tr = x[:n_train], y[:n_train]
x_val, y_val = x[n_train:(n_train + n_val)], y[n_train:(n_train + n_val)]
x_te, y_te = x[-n_test:], y[-n_test:]

with h5py.File(
        os.path.join(ppi_path, 'output/create_random_forest_dataset_500.hdf5'),
        'w') as f:
    f['train/x'], f['train/y'] = x_tr[...], y_tr[...]
    f['val/x'], f['val/y'] = x_val[...], y_val[...]
    f['test/x'], f['test/y'] = x_te[...], y_te[...]
