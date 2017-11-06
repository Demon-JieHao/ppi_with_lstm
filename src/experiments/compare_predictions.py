from __future__ import print_function, division, absolute_import

import h5py
from keras.models import load_model
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd


def high_confidence_errors(logits, misclass, cutoff=0.05):
    hi_conf_err = np.logical_and(misclass, np.less(logits.reshape(-1), cutoff))
    return hi_conf_err


# Model based on protein FPs
model_fp = load_model('models/best_prot_fp_model')

with h5py.File('output/create_protein_fp_dataset_500_.hdf5', 'r') as fp:
    x1_te, x2_te, y_te = fp['test/x1'], fp['test/x2'], fp['test/y']
    logits_fp = model_fp.predict([x1_te, x2_te], batch_size=128)
    preds_fp = (logits_fp > 0.5).astype('int64')
    report_fp = classification_report(y_te, preds_fp)
    misclass_fp = np.not_equal(preds_fp.reshape(-1), y_te)

# Models based on embeddings and cnn
model_emb = load_model('models/best_embedding_model')
model_cnn = load_model('models/best_cnn_model')

with h5py.File('output/create_tokenized_dataset_500_.hdf5', 'r') as ft:
    x1_te, x2_te, y_te = ft['test/x1'], ft['test/x2'], ft['test/y']
    logits_emb = model_emb.predict([x1_te, x2_te], batch_size=128)
    preds_emb = (logits_emb > 0.5).astype('int64')
    logits_cnn = model_cnn.predict([x1_te, x2_te], batch_size=128)
    preds_cnn = (logits_cnn > 0.5).astype('int64')
    report_emb = classification_report(y_te, preds_emb)
    report_cnn = classification_report(y_te, preds_cnn)

    # Find the misclassified pairs
    misclass_emb = np.not_equal(preds_emb.reshape(-1), y_te)
    misclass_cnn = np.not_equal(preds_cnn.reshape(-1), y_te)

ppi_data = pd.read_hdf('output/filtered_ppi_dataset_500_.hdf5')

# Retain only the test set
ppi_data = ppi_data.iloc[-10000:]

# Are different models making similar mistakes?
emb_cnn = np.intersect1d(np.where(misclass_emb), np.where(misclass_cnn))
emb_fp = np.intersect1d(np.where(misclass_emb), np.where(misclass_fp))
cnn_fp = np.intersect1d(np.where(misclass_cnn), np.where(misclass_fp))

p_emb_cnn = len(emb_cnn) / np.min([misclass_emb.sum(), misclass_cnn.sum()])
p_emb_fp = len(emb_fp) / np.min([misclass_emb.sum(), misclass_fp.sum()])
p_cnn_fp = len(cnn_fp) / np.min([misclass_cnn.sum(), misclass_fp.sum()])
