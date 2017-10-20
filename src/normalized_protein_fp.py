from __future__ import absolute_import, division, print_function
import pandas as pd
from sklearn.preprocessing import StandardScaler


protein_fp = pd.read_table(
    '/da/dmp/cb/dariogi1/projects/2017/squads/ppi_with_lstm/output/proteinFPs.tsv.gz',
    sep='\t',
    index_col=0)
vals = protein_fp.values
idx = protein_fp.index
ssc = StandardScaler()
vals = ssc.fit_transform(vals)

protein_fp = pd.DataFrame(vals, index=idx)
protein_fp.to_hdf(
    '/da/dmp/cb/dariogi1/projects/2017/squads/ppi_with_lstm/output/normalized_protein_fp.hdf5',
    key='norm_prot_fp')
