from __future__ import absolute_import, division, print_function
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import h5py
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('maxlen', help='maximum protein length', type=int)
parser.add_argument('ppi_path', help='path to the main folder', type=str)
args = parser.parse_args()

maxlen = args.maxlen
ppi_path = args.ppi_path
n_test_samples = 10000

filtered_ppi_file = os.path.join(
    ppi_path, '_'.join(['output/filtered_ppi_dataset',
                       str(maxlen), '.hdf5'])
)
dataset = pd.read_hdf(filtered_ppi_file)

aas = 'ACDEFGHIKLMNPQRSTVWY'
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(aas)

seq_indices1 = dataset.sequence1.apply(lambda x: [
    tokenizer.word_index[c] for c in x])
seq_indices2 = dataset.sequence2.apply(lambda x: [
    tokenizer.word_index[c] for c in x])

x1 = pad_sequences(seq_indices1.tolist(), maxlen=maxlen, value=0)
x2 = pad_sequences(seq_indices2.tolist(), maxlen=maxlen, value=0)
y = dataset.interaction.values

# Transform the indices in int32 for later use with one_hot
x1 = x1.astype('int32')
x2 = x2.astype('int32')

x1_train, x2_train, y_train = (x1[:-(2 * n_test_samples)],
                               x2[:-(2 * n_test_samples)],
                               y[:-(2 * n_test_samples)])

x1_valid, x2_valid, y_valid = (x1[-(2 * n_test_samples):-n_test_samples],
                               x2[-(2 * n_test_samples):-n_test_samples],
                               y[-(2 * n_test_samples):-n_test_samples])

x1_test, x2_test, y_test = (x1[-n_test_samples:],
                            x2[-n_test_samples:],
                            y[-n_test_samples:])

print('Saving the dataset')
output_file = os.path.join(
    ppi_path, '_'.join(['output/create_tokenized_dataset',
                       str(maxlen), '.hdf5']))

with h5py.File(os.path.join(ppi_path, output_file), 'w') as f:

    x1_tr = f.create_dataset('train/x1', x1_train.shape, dtype=x1.dtype,
                             compression='gzip')
    x2_tr = f.create_dataset('train/x2', x2_train.shape, dtype=x2.dtype,
                             compression='gzip')
    y_tr = f.create_dataset('train/y', y_train.shape, dtype=y.dtype,
                            compression='gzip')
    x1_tr[...] = x1_train
    x2_tr[...] = x2_train
    y_tr[...] = y_train

    x1_val = f.create_dataset('val/x1', x1_valid.shape, dtype=x1.dtype,
                              compression='gzip')
    x2_val = f.create_dataset('val/x2', x2_valid.shape, dtype=x2.dtype,
                              compression='gzip')
    y_val = f.create_dataset('val/y', y_valid.shape, dtype=y.dtype,
                             compression='gzip')
    x1_val[...] = x1_valid
    x2_val[...] = x2_valid
    y_val[...] = y_valid

    x1_te = f.create_dataset('test/x1', x1_test.shape, dtype=x1.dtype,
                             compression='gzip')
    x2_te = f.create_dataset('test/x2', x2_test.shape, dtype=x2.dtype,
                             compression='gzip')
    y_te = f.create_dataset('test/y', y_test.shape, dtype=y.dtype,
                            compression='gzip')
    x1_te[...] = x1_test
    x2_te[...] = x2_test
    y_te[...] = y_test
