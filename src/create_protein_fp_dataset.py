from __future__ import absolute_import, division, print_function
import pandas as pd
import h5py
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('maxlen', help='maximum protein length', type=int)
parser.add_argument('ppi_path', help='path to the main folder', type=str)
args = parser.parse_args()

maxlen = args.maxlen
ppi_path = args.ppi_path

# Dataset containing the normalized protein fingerprints for all the proteins
# in Florian's dataset.
norm_prot_fps = pd.read_hdf(
    os.path.join(
        ppi_path, '_'.join(
            ['output/normalized_protein_fp', str(maxlen), '.hdf5'])))

# Dataset containing the protein ID and the sequence for the pairs that pass
# the filtering, i.e., which have a length between 5 and 500 and without 'U's.
filtered_pairs = pd.read_hdf(
    os.path.join(
        ppi_path, '_'.join(
            ['output/filtered_ppi_dataset', str(maxlen), '.hdf5'])))

x1 = norm_prot_fps.loc[filtered_pairs.uid1].values
x2 = norm_prot_fps.loc[filtered_pairs.uid2].values
y = filtered_pairs.interaction.values

x1_train = x1[:-10000]
x2_train = x2[:-10000]
y_train = y[:-10000]

x1_test = x1[-10000:]
x2_test = x2[-10000:]
y_test = y[-10000:]

output_file = os.path.join(
    ppi_path,
    '_'.join(['output/create_protein_fp_dataset', str(maxlen), '.hdf5'])
)
print('Saving the dataset in {}'.format(output_file))
with h5py.File(output_file, 'w') as f:
    x1_tr = f.create_dataset('train/x1', x1_train.shape, dtype=x1.dtype,
                             compression='gzip')
    x2_tr = f.create_dataset('train/x2', x2_train.shape, dtype=x2.dtype,
                             compression='gzip')
    y_tr = f.create_dataset('train/y', y_train.shape, dtype=y.dtype,
                            compression='gzip')
    x1_tr[...] = x1_train
    x2_tr[...] = x2_train
    y_tr[...] = y_train

    x1_te = f.create_dataset('test/x1', x1_test.shape, dtype=x1.dtype,
                             compression='gzip')
    x2_te = f.create_dataset('test/x2', x2_test.shape, dtype=x2.dtype,
                             compression='gzip')
    y_te = f.create_dataset('test/y', y_test.shape, dtype=y.dtype,
                            compression='gzip')
    x1_te[...] = x1_test
    x2_te[...] = x2_test
    y_te[...] = y_test
