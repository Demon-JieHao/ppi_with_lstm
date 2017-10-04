import pandas as pd
from sklearn.preprocessing import StandardScaler
import h5py

protein_fps = pd.read_table('../output/proteinFP.tsv', sep='\t',
                            index_col=0)
x = protein_fps.values
ssc = StandardScaler()
ssc.fit(x)

with h5py.File('../output/filtered_protein_fp_dataset.hdf5', 'r') as g:
    x1 = g['x1']
    x2 = g['x2']
    y = g['y']

    print('Normalizing the protein FP datasets')
    x1 = ssc.transform(x1)
    x2 = ssc.transform(x2)

    x1_train = x1[:-10000]
    x2_train = x2[:-10000]
    y_train = y[:-10000]

    x1_test = x1[-10000:]
    x2_test = x2[-10000:]
    y_test = y[-10000:]

    print('Saving the dataset')
    with h5py.File('../output/create_protein_fp_dataset.hdf5', 'w') as f:

        x1_tr = f.create_dataset('train/x1', x1_train.shape, dtype=x1.dtype,
                                 compression='gzip')
        x2_tr = f.create_dataset('train/x2', x2_train.shape, dtype=x2.dtype,
                                 compression='gzip')
        y_tr = f.create_dataset('train/y', y_train.shape, dtype=y.dtype,
                                compression='gzip')
        x1_tr[...] = x1_train
        x2_tr[...] = x2_train
        y_tr[...] = y_train

        x1_te = f.create_dataset('test/x1', x1_test.shape, dtype=x1.dtype,
                                 compression='gzip')
        x2_te = f.create_dataset('test/x2', x2_test.shape, dtype=x2.dtype,
                                 compression='gzip')
        y_te = f.create_dataset('test/y', y_test.shape, dtype=y.dtype,
                                compression='gzip')
        x1_te[...] = x1_test
        x2_te[...] = x2_test
        y_te[...] = y_test
