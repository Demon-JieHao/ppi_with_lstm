from __future__ import print_function, division, absolute_import
import os
import h5py
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.externals import joblib
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('maxlen', help='maximum protein length', type=int)
parser.add_argument('ppi_path', help='path to the main folder', type=str)

args = parser.parse_args()
maxlen = args.maxlen
ppi_path = args.ppi_path


fp_file = os.path.join(
    ppi_path, '_'.join(
        ['output/create_random_forest_dataset', str(maxlen), '.hdf5']))

with h5py.File(os.path.join(fp_file), 'r') as f:
    x_tr, y_tr = f['train/x'], f['train/y']
    x_te, y_te = f['test/x'], f['test/y']

    rfc = RandomForestClassifier(n_jobs=6, random_state=42)
    rfc.fit(x_tr, y_tr)
    s = joblib.dump(rfc, os.path.join(ppi_path, 'output/fitted_rf.pkl'))
    preds = rfc.predict(x_te)
    print(classification_report(y_te, preds))
