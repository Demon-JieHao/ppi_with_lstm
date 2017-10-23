from __future__ import absolute_import, division, print_function
import h5py
import os
from keras.callbacks import TensorBoard
from create_cnn_model import cnn_gru_model

batch_size = 256
ppi_path = '/home/giovenko/DeepLearning/ppi_with_lstm'

model = cnn_gru_model(conv_layers=1,
                      filters=128,
                      size_kernel=24,
                      pool_size=12,
                      gru_states=16,
                      learning_rate=0.001,
                      hidden_layers=1,
                      units=32,
                      dropout=0.2,
                      sequence_length=500,
                      n_classes=20)

logdir = 'tb_logs/cnn_gru/'
tensorboard = [
    TensorBoard(log_dir=logdir,
                histogram_freq=1,
                batch_size=batch_size,
                write_graph=True)
]

with h5py.File(
        os.path.join(ppi_path, 'output/create_tokenized_dataset_500.hdf5'),
        'r') as f:
    x1_tr, x2_tr, y_tr = (f['train/x1'], f['train/x2'], f['train/y'])
    x1_te, x2_te, y_te = (f['test/x1'], f['test/x2'], f['test/y'])

    model.fit(x=[x1_tr, x2_tr], y=y_tr,
              batch_size=batch_size,
              epochs=30,
              callbacks=tensorboard,
              validation_split=0.05,
              shuffle='batch')
