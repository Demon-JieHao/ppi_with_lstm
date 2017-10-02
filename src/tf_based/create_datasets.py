# Purpose: filter the protein sequences that are within a certain range
# and filter the interaction dataset such that only these pairs appear.
# Create representations of the protein sequences as np.arrays
import gzip
import pickle
import numpy as np
import pandas as pd

# This should be set to 1000
MAX_PROTEIN_LENGTH = 500

# There are very few proteins this short
MIN_PROTEIN_LENGTH = 50

allowed_aas = 'ARNDCQEGHILKMFPSTWYVU'
n_aas = len(allowed_aas)
n_aa_seq = np.arange(n_aas)

# Mapping between amino-acids and numeric indices (from zero)
aa_to_indices = dict(zip(allowed_aas, n_aa_seq))


# Help function to convert protein sequences to integer lists.
# Uncomment the lines in the function body if you want to add
# padding already at this stage.
def seq_to_array(seq):
    # out = np.zeros(MAX_PROTEIN_LENGTH, dtype='int64')
    tmp = np.array([aa_to_indices[k] for k in seq])
    # out[np.arange(len(tmp))] = tmp
    return tmp


# Read the protiein protein interaction dataset
interaction_set = pd.read_table('../data/ppi_testset.txt.gz',
                                header=None,
                                names=['uid1', 'uid2', 'interaction'])

# Read the protein sequences
protein_sequences = pd.read_table('../data/sequences.fa.gz',
                                  index_col=0, header=None,
                                  names=['sequence'])

# Step 1
# Identify the proteins shorter than MAX_PROTEIN_LENGTH and longer than
# MIN_PROTEIN_LENGTH
protein_lengths = protein_sequences.sequence.apply(len).values
in_range = np.logical_and(protein_lengths > MIN_PROTEIN_LENGTH,
                          protein_lengths < MAX_PROTEIN_LENGTH)

protein_sequences = protein_sequences.loc[in_range]

# Create a dictionary where the Uniprot ID is the key and the sequence is
# the value
uniprot_ids = protein_sequences.index.values
sequences = protein_sequences.sequence.values
uniprot_to_sequence = dict(zip(uniprot_ids, sequences))

# Step 2
# Transform the protein sequences into numeric arrays of indices
uniprot_to_indices = {k: seq_to_array(v)
                      for k, v in uniprot_to_sequence.items()}

# Step 3
# Store the interacting pairs in a better format.
filtered_uids = list(uniprot_to_indices.keys())
idx1 = interaction_set.uid1.isin(filtered_uids).values
idx2 = interaction_set.uid2.isin(filtered_uids).values
to_keep = np.logical_and(idx1, idx2)
interaction_set = interaction_set.loc[to_keep]

# Shuffle the interaction dataset before saving
interaction_set = interaction_set.sample(frac=1.)

uid1 = interaction_set.uid1.values
uid2 = interaction_set.uid2.values
y = interaction_set.interaction.values

X1 = [uniprot_to_indices[uid] for uid in uid1]
X2 = [uniprot_to_indices[uid] for uid in uid2]

# Store the length of each protein in the datasets
len_X1 = np.array([len(x) for x in X1])
len_X2 = np.array([len(x) for x in X2])

print('Saving the dataset')
pickle.dump([X1, X2, y, len_X1, len_X2],
            gzip.open('../output/create_dataset.pkl.gzip', 'w'))
