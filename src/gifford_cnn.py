from __future__ import absolute_import, division, print_function
import h5py
from keras.models import Model
import keras.layers
import keras.backend as K
import os

# ppi_path = '/lustre/scratch/dariogi1/ppi_with_lstm'
ppi_path = '/home/giovenko/DeepLearning/ppi_with_lstm'

# General parameters (don't change these)
sequence_length = 500
input_shape = (sequence_length,)
n_classes = 20
output_shape = (sequence_length, n_classes)

# Model-specific parameters
n_feature_maps1 = 128
kernel_width1 = 24
n_hidden_units1 = 32
dropout_rate1 = 0.5

tb_path = '_'.join([os.path.join(ppi_path, 'tb_logs/cnn'),
                    str(kernel_width1),
                    str(n_hidden_units1),
                    str(dropout_rate1)])

# Shared embedding and CNN layers
one_hot_encoder = keras.layers.Lambda(K.one_hot,
                                      arguments={'num_classes': n_classes},
                                      output_shape=output_shape)

conv1 = keras.layers.Conv1D(n_feature_maps1, kernel_width1, activation='relu')

# Input layers
input1 = keras.layers.Input(shape=(sequence_length,), dtype='int32',
                            name='input1')
input2 = keras.layers.Input(shape=(sequence_length,), dtype='int32',
                            name='input2')

# Shared embeddings for the two inputs
embedding1 = one_hot_encoder(input1)
embedding2 = one_hot_encoder(input2)

encoding1 = conv1(embedding1)
encoding1 = keras.layers.GlobalMaxPooling1D()(encoding1)
encoding2 = conv1(embedding2)
encoding2 = keras.layers.GlobalMaxPooling1D()(encoding2)

concatenated = keras.layers.concatenate([encoding1, encoding2], axis=-1)

# Add fully connected layers to the concatenated tensors
hidden1 = keras.layers.Dense(n_hidden_units1, activation='relu')(concatenated)
hidden1 = keras.layers.Dropout(dropout_rate1)(hidden1)
# hidden2 = keras.layers.Dense(n_hidden_units2, activation='relu')(hidden1)
# hidden2 = keras.layers.Dropout(dropout_rate2)(hidden2)
predictions = keras.layers.Dense(1, activation='sigmoid')(hidden1)

model = Model([input1, input2], predictions)

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc'])

callback = [keras.callbacks.TensorBoard(log_dir=tb_path, histogram_freq=1.0)]

with h5py.File(
        os.path.join(ppi_path, 'output/create_tokenized_dataset_500.hdf5'),
    'r') as f:
    x1_tr, x2_tr, y_tr = (f['train/x1'], f['train/x2'], f['train/y'])
    x1_te, x2_te, y_te = (f['test/x1'], f['test/x2'], f['test/y'])

    model.fit(x=[x1_tr, x2_tr], y=y_tr,
              batch_size=128,
              epochs=30,
              callbacks=callback,
              validation_split=0.05,
              shuffle='batch')
