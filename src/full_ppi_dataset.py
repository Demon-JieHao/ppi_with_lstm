# Create an HDF5 version of the complete PPI dataset, with Uniprot IDs,
# sequences and interactions.
# The dataset is already shuffled.
from __future__ import absolute_import, division, print_function
import pandas as pd
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('ppi_path', help='path to the main folder', type=str)
args = parser.parse_args()
ppi_path = args.ppi_path

ppi_data = pd.read_table(
    os.path.join(ppi_path, 'data/ppi_testset.txt.gz'),
    header=None, names=['uniprot_id1', 'uniprot_id2', 'interaction']
)
sequences = pd.Series.from_csv(os.path.join(ppi_path, 'data/sequences.fa.gz'),
                               header=None, index_col=0, sep='\t')

seq1 = sequences[ppi_data['uniprot_id1']]
seq1.index = ppi_data.index
seq2 = sequences[ppi_data['uniprot_id2']]
seq2.index = ppi_data.index

dataset = pd.DataFrame({
    'uid1': ppi_data.uniprot_id1,
    'sequence1': seq1,
    'uid2': ppi_data.uniprot_id2,
    'sequence2': seq2,
    'interaction': ppi_data.interaction
})

# Shuffle the dataset
dataset = dataset.sample(frac=1.0, random_state=42)

# Compute the length of the sequences
# seq_len = pd.DataFrame({'len1': dataset.sequence1.apply(len),
#                         'len2': dataset.sequence2.apply(len)})
# dataset['mean_len'] = seq_len.mean(axis=1)
# dataset.sort_values(by='mean_len', ascending=True, inplace=True)

dataset.to_hdf(
    os.path.join(ppi_path, 'output/full_ppi_dataset_master.hdf5'),
    key='ppi_data'
)
