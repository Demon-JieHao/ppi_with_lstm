from __future__ import print_function, division, absolute_import
import os
import pandas as pd
import h5py
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('maxlen', help='maximum protein length', type=int)
parser.add_argument('ppi_path', help='path to the main folder', type=str)
args = parser.parse_args()

maxlen = args.maxlen
ppi_path = args.ppi_path

# Dataset containing the protein ID and the sequence for the pairs that pass
# the filtering, i.e., which have a length between 5 and 500 and without 'U's.
filtered_file = os.path.join(
    ppi_path, '_'.join(
        ['output/filtered_ppi_dataset', str(maxlen), '.hdf5']))
filtered_pairs = pd.read_hdf(filtered_file)

# Dataset containing the normalized protein fingerprints for all the proteins
# in Florian's dataset.
norm_prot_file = os.path.join(
    ppi_path, '_'.join(
        ['output/create_protein_fp_dataset', str(maxlen), '.hdf5']))

output_file = os.path.join(
    ppi_path, '_'.join(
        ['output/create_random_forest_dataset', str(maxlen), '.hdf5']))

print("Creating the {} output file".format(output_file))
with h5py.File(norm_prot_file, 'r') as fin:
    x1_tr, x2_tr, y_tr = fin['train/x1'], fin['train/x2'], fin['train/y']
    x1_test, x2_test, y_test = fin['test/x1'], fin['test/x2'], fin['test/y']
    x_tr = np.hstack([x1_tr, x2_tr])
    x_test = np.hstack([x1_test, x2_test])

    with h5py.File(output_file, 'w') as fout:
        x_tr_out = fout.create_dataset('train/x', x_tr.shape,
                                       dtype=x_tr.dtype,
                                       compression='gzip')
        y_tr_out = fout.create_dataset('train/y', y_tr.shape,
                                       dtype=y_tr.dtype,
                                       compression='gzip')
        x_test_out = fout.create_dataset('test/x', x_test.shape,
                                         dtype=x_test.dtype,
                                         compression='gzip')
        y_test_out = fout.create_dataset('test/y', y_test.shape,
                                         dtype=y_test.dtype,
                                         compression='gzip')

        x_tr_out[...] = x_tr
        x_test[...] = x_test
        y_tr_out[...] = y_tr
        y_test_out[...] = y_test
