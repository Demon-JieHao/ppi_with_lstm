from __future__ import absolute_import, division, print_function
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('maxlen', help='maximum protein length', type=int)
parser.add_argument('ppi_path', help='path to the main folder', type=str)
args = parser.parse_args()

ppi_path = args.ppi_path
maxlen = args.maxlen
minlen = 50

sequences = pd.Series.from_csv(os.path.join(ppi_path, 'data/sequences.fa.gz'),
                               header=None, index_col=0, sep='\t')

#  Filter out the sequences longer than maxlen or shorter than minlen
idx_keep = sequences.apply(lambda z: minlen < len(z) < maxlen)
sequences = sequences.loc[idx_keep]

# Remove all the sequences containing 'U'
idx_keep = sequences.apply(lambda z: 'U' not in z)
sequences = sequences.loc[idx_keep]

aas = 'ACDEFGHIKLMNPQRSTVWY'
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(aas)

seq_indices = sequences.apply(lambda z: [
    tokenizer.word_index[c] for c in z
])

x = pad_sequences(seq_indices.tolist(), maxlen=maxlen, value=0)
tokenized_seqs = pd.DataFrame(x, index=seq_indices.index)
output_file = os.path.join(ppi_path, '_'.join([
    'output/filtered_tokenized_ppi_dataset', str(maxlen), '.hdf5'
]))
tokenized_seqs.to_hdf(output_file, 'tokenized_seqs')
