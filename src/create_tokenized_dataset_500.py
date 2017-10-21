from __future__ import absolute_import, division, print_function
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
# from keras.preprocessing.text import Tokenizer
import h5py
import os


maxlen = 500
ppi_path = '/lustre/scratch/dariogi1/ppi_with_lstm'

dataset = pd.read_hdf(
    os.path.join(ppi_path, 'output/filtered_ppi_dataset_500.hdf5')
)

# Map the amino-acids to integers, such that the first AA is mapped to 0.
# This differs from the behavior of Keras' tokenizer, that reserves zero for
# special purposes. To make things work we pad the sequences with -1, following
# the examle in TensorFlow tf.one_hot function.
aas = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R',
       'S', 'T', 'V', 'W', 'Y']
aa_idx = list(range(len(aas)))
aa_mapping = dict(zip(aas, aa_idx))

seq_indices1 = dataset.sequence1.apply(lambda x: [aa_mapping[c] for c in x])
seq_indices2 = dataset.sequence2.apply(lambda x: [aa_mapping[c] for c in x])

x1 = pad_sequences(seq_indices1.tolist(), maxlen=maxlen, value=-1)
x2 = pad_sequences(seq_indices2.tolist(), maxlen=maxlen, value=-1)
y = dataset.interaction.values

# Transform the indices in int32 for later use with one_hot
x1 = x1.astype('int32')
x2 = x2.astype('int32')

x1_train = x1[:-10000]
x2_train = x2[:-10000]
y_train = y[:-10000]

x1_test = x1[-10000:]
x2_test = x2[-10000:]
y_test = y[-10000:]


print('Saving the dataset')
with h5py.File(
        os.path.join(ppi_path, 'output/create_tokenized_dataset_500.hdf5'), 'w'
) as f:
    
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
