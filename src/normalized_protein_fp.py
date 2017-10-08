import pandas as pd
from sklearn.preprocessing import StandardScaler


protein_fp = pd.read_table('output/proteinFPs.tsv.gz', sep='\t',
                           index_col=0)
vals = protein_fp.values
idx = protein_fp.index
ssc = StandardScaler()
vals = ssc.fit_transform(vals)

protein_fp = pd.DataFrame(vals, index=idx)
protein_fp.to_hdf('output/normalized_protein_fp.hdf5', key='norm_prot_fp')