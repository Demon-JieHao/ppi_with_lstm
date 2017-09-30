import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import h5py

minlen = 10
maxlen = 500

ppi_data = pd.read_table('../data/ppi_testset.txt.gz', header=None,
                         names=['uniprot_id1', 'uniprot_id2', 'interaction'])
sequences = pd.read_table('../data/sequences.fa.gz', header=None, index_col=0,
                          names=['sequence'])

sequence1 = sequences.loc[ppi_data.uniprot_id1].values.flatten()
sequence2 = sequences.loc[ppi_data.uniprot_id2].values.flatten()
interaction = ppi_data.interaction.values

# Filter by length
len1 = []
len2 = []
for i in np.arange(len(sequence1)):
    len1.append(len(sequence1[i]))
    len2.append(len(sequence2[i]))

len1 = np.array(len1)
len2 = np.array(len2)

print('Filtering by length')
idx_max = (len1 <= maxlen) & (len2 <= maxlen)
idx_min = (len1 >= minlen) & (len2 >= minlen)
idx = idx_max & idx_min

print("{} pairs pass the filtering".format(idx.sum()))

sequence1 = sequence1[idx]
sequence2 = sequence2[idx]
interaction = interaction[idx]

# Shuffle the dataset
idx_shuffle = np.arange(len(interaction))

print('Fitting the tokenizer')
tokenizer = Tokenizer()
tokenizer.fit_on_texts(['l s a v g e p t r i k f d q n y m h c w u'])


def preprocess(seq):
    sequence = seq[idx_shuffle]
    sequence = [[c for c in s] for s in sequence]
    sequence = [' '.join(s) for s in sequence]
    sequence = tokenizer.texts_to_sequences(sequence)
    sequence = np.array(sequence)
    sequence = pad_sequences(sequence, maxlen=maxlen)
    return(sequence)


print('Preprocessing the sequences')
x1 = preprocess(sequence1)
x2 = preprocess(sequence2)
y = interaction[idx_shuffle]

print('Creating the hdf5 file')


def create_train_dev_test(test_size=10000,
                          h5_file='../output/create_tokenized_dataset.hdf5'):
    n_pairs = y.shape[0]

    slice_train = slice(0, n_pairs - 2 * test_size)
    slice_dev = slice(n_pairs - 2*test_size, n_pairs - test_size)
    slice_test = slice(n_pairs - test_size, n_pairs)

    x1_train, x2_train, y_train = (x1[slice_train], x2[slice_train],
                                   y[slice_train])
    x1_dev, x2_dev, y_dev = (x1[slice_dev], x2[slice_dev],
                             y[slice_dev])
    x1_test, x2_test, y_test = (x1[slice_test], x2[slice_test],
                                y[slice_test])

    with h5py.File(h5_file, 'w') as f:

        x1_train = f.create_dataset('train/x1',
                                    x1_train.shape,
                                    dtype='float32')
        x2_train = f.create_dataset('train/x2',
                                    x2_train.shape,
                                    dtype='float32')
        y_train = f.create_dataset('train/y',
                                   y_train.shape,
                                   dtype='float32')

        x1_dev = f.create_dataset('dev/x1',
                                  x1_dev.shape,
                                  dtype='float32')
        x2_dev = f.create_dataset('dev/x2',
                                  x2_dev.shape,
                                  dtype='float32')
        y_dev = f.create_dataset('dev/y',
                                 y_dev.shape,
                                 dtype='float32')

        x1_test = f.create_dataset('test/x1',
                                   x1_test.shape,
                                   dtype='float32')
        x2_test = f.create_dataset('test/x2',
                                   x2_test.shape,
                                   dtype='float32')
        y_test = f.create_dataset('test/y',
                                  y_test.shape,
                                  dtype='float32')

        f['train/x1'][...] = x1_train
        f['train/x2'][...] = x2_train
        f['train/y'][...] = y_train
        f['dev/y'][...] = y_dev
        f['dev/x1'][...] = x1_dev
        f['dev/x2'][...] = x2_dev
        f['test/x1'][...] = x1_test
        f['test/x2'][...] = x2_test
        f['test/y'][...] = y_test
