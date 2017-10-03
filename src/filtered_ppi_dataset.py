import pandas as pd


minlen = 10
maxlen = 500
test_size = 10000

dataset = pd.read_hdf('../output/full_ppi_dataset.hdf5')

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

dataset.to_hdf('../output/filtered_ppi_dataset.hdf5', key='filtered_set')
