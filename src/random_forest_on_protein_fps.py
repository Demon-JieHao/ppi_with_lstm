from __future__ import print_function, division, absolute_import
import os
import h5py
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.externals import joblib


ppi_path = '/home/giovenko/DeepLearning/ppi_with_lstm'
with h5py.File(
        os.path.join(ppi_path, 'output/create_random_forest_dataset_500.hdf5'),
        'r') as f:
    x_tr, y_tr = f['train/x'], f['train/y']
    x_val, y_val = f['val/x'], f['val/y']
    x_te, y_te = f['test/x'], f['test/y']

    rfc = RandomForestClassifier(n_jobs=6, random_state=42)
    rfc.fit(x_tr, y_tr)
    s = joblib.dump(rfc, os.path.join(ppi_path, 'output/fitted_rf.pkl'))
    preds = rfc.predict(x_te)
    print(classification_report(y_te, preds))
