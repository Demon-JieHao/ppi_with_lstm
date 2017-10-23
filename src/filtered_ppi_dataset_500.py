from __future__ import absolute_import, division, print_function
import pandas as pd
import os


minlen = 50
maxlen = 500

# ppi_path = '/lustre/scratch/dariogi1/ppi_with_lstm'
ppi_path = '/home/giovenko/DeepLearning/ppi_with_lstm'

dataset = pd.read_hdf(
    os.path.join(ppi_path, 'output/full_ppi_dataset_master.hdf5')
)

# Retain only the canonical aminoacids
idx1 = dataset.sequence1.apply(lambda x: 'U' not in x)
idx2 = dataset.sequence2.apply(lambda x: 'U' not in x)
dataset = dataset.loc[idx1 & idx2]

# Compute the sequence lengths
len1 = dataset.sequence1.apply(len)
len2 = dataset.sequence2.apply(len)

idx_lenght = (
    (len1 >= minlen) & (len1 <= maxlen) &
    (len2 >= minlen) & (len2 <= maxlen)
)
dataset = dataset.loc[idx_lenght]

print("{} pairs pass the filtering".format(dataset.shape[0]))

dataset.to_hdf(
    os.path.join(ppi_path, 'output/filtered_ppi_dataset_500_master.hdf5'),
    key='filtered_set'
)
