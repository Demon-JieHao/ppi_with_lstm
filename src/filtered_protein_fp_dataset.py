import pandas as pd
import h5py

dataset = pd.read_hdf('../output/filtered_ppi_dataset.hdf5')
protein_fps = pd.read_table('../output/proteinFP.tsv', sep='\t',
                            index_col=0)

# Make sure that all the proteins in `dataset` appear in protein_fp
idx_all = (dataset.uid1.isin(protein_fps.index).all() &
           dataset.uid2.isin(protein_fps.index).all())
assert idx_all

protein_fp1 = protein_fps.loc[dataset['uid1']]
protein_fp2 = protein_fps.loc[dataset['uid2']]
protein_fp1.index = dataset.index
protein_fp2.index = dataset.index

y = dataset.interaction.values
x1 = protein_fp1.values
x2 = protein_fp2.values

with h5py.File('../output/filtered_protein_fp_dataset.hdf5', 'w') as f:
    x1h5 = f.create_dataset('x1', x1.shape, dtype='float32',
                            compression='gzip')
    x2h5 = f.create_dataset('x2', x2.shape, dtype='float32',
                            compression='gzip')
    yh5 = f.create_dataset('y', y.shape, dtype='int32',
                           compression='gzip')

    x1h5[...] = x1
    x2h5[...] = x2
    yh5[...] = y
