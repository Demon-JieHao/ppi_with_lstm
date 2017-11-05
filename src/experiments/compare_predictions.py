from __future__ import print_function, division, absolute_import

import h5py
from keras.models import load_model
from sklearn.metrics import classification_report

# Model based on protein FPs
model_fp = load_model('models/best_prot_fp_model')

with h5py.File('output/create_protein_fp_dataset_500_.hdf5', 'r') as fp:
    x1_te, x2_te, y_te = fp['test/x1'], fp['test/x2'], fp['test/y']
    preds_fp = model_fp.predict([x1_te, x2_te], batch_size=128)
    preds_fp = (preds_fp > 0.5).astype('int64')
    report_fp = classification_report(y_te, preds_fp)

# Models based on embeddings and cnn
model_emb = load_model('models/best_embedding_model')
model_cnn = load_model('models/best_cnn_model')

with h5py.File('output/create_tokenized_dataset_500_.hdf5', 'r') as ft:
    x1_te, x2_te, y_te = ft['test/x1'], ft['test/x2'], ft['test/y']
    preds_emb = model_emb.predict([x1_te, x2_te], batch_size=128)
    preds_cnn = model_cnn.predict([x1_te, x2_te], batch_size=128)
    report_emb = classification_report(y_te, (preds_emb > 0.5).astype('int64'))
    report_cnn = classification_report(y_te, (preds_cnn > 0.5).astype('int64'))
