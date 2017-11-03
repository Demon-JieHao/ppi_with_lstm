from __future__ import absolute_import, division, print_function
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('maxlen', help='maximum protein length', type=int)
parser.add_argument('ppi_path', help='path to the main folder', type=str)
args = parser.parse_args()
ppi_path = args.ppi_path
maxlen = args.maxlen

protein_fp = pd.read_table(
    os.path.join(ppi_path, 'data/proteinFPs.tsv.gz'),
    sep='\t', index_col=0
)
vals = protein_fp.values
idx = protein_fp.index
ssc = StandardScaler()
vals = ssc.fit_transform(vals)

protein_fp = pd.DataFrame(vals, index=idx)
output_file = os.path.join(
    ppi_path,
    '_'.join(['output/normalized_protein_fp', str(maxlen), '.hdf5'])
)

protein_fp.to_hdf(output_file, key='norm_prot_fp')
