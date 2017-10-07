# Create an HDF5 version of the complete PPI dataset, with Uniprot IDs,
# sequences and interactions.
# The dataset is already shuffled.
import pandas as pd

ppi_data = pd.read_table('data/ppi_testset.txt.gz', header=None,
                         names=['uniprot_id1', 'uniprot_id2', 'interaction'])
sequences = pd.Series.from_csv('data/sequences.fa.gz', header=None,
                               index_col=0, sep='\t')

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

dataset.to_hdf('output/full_ppi_dataset.hdf5', key='ppi_data')
