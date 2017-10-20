import h5py
from keras.models import Model
import keras.layers
import keras.backend as K
import os

ppi_path = '/lustre/scratch/dariogi1/ppi_with_lstm'

sequence_length = 500
input_shape = (sequence_length,)
n_classes = 20
output_shape = (sequence_length, n_classes)

# Shared embedding and CNN layers
ohe = keras.layers.Lambda(K.one_hot, arguments={'num_classes': n_classes},
                          output_shape=output_shape)

conv = keras.layers.Conv1D(64, 7, activation='relu')
conv2 = keras.layers.Conv1D(128, 7, activation='relu')

# Input layers
input1 = keras.layers.Input(shape=(sequence_length,), dtype='int32',
                            name='input1')
input2 = keras.layers.Input(shape=(sequence_length,), dtype='int32',
                            name='input2')

# Shared embeddings for the two inputs
embedding1 = ohe(input1)
embedding2 = ohe(input2)

# First shared convolutional layer
encoding1 = conv(embedding1)
encoding1 = keras.layers.MaxPooling1D(5)(encoding1)
encoding2 = conv(embedding2)
encoding2 = keras.layers.MaxPooling1D(5)(encoding2)

# Second shared convolutional layer
encoding1 = conv2(encoding1)
encoding1 = keras.layers.GlobalMaxPooling1D()(encoding1)
encoding2 = conv2(encoding2)
encoding2 = keras.layers.GlobalMaxPooling1D()(encoding2)

concatenated = keras.layers.concatenate([encoding1, encoding2], axis=-1)
hidden1 = keras.layers.Dense(512, activation='relu')(concatenated)
hidden1 = keras.layers.Dropout(0.4)(hidden1)
hidden2 = keras.layers.Dense(512, activation='relu')(hidden1)
hidden2 = keras.layers.Dropout(0.4)(hidden2)
predictions = keras.layers.Dense(1, activation='sigmoid')(hidden2)

model = Model([input1, input2], predictions)

# adam = keras.optimizers.Adam(lr=0.001)
rmsprop = keras.optimizers.rmsprop(lr=0.0001)

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc'])

# callback = [
#     keras.callbacks.TensorBoard(
#         log_dir='tb_logs/cnn1d',
#         histogram_freq=0.1)
#     ]

with h5py.File(
        os.path.join(ppi_path, 'output/create_tokenized_dataset_500.hdf5'),
    'r') as f:
    x1_tr, x2_tr, y_tr = (f['train/x1'], f['train/x2'], f['train/y'])
    x1_te, x2_te, y_te = (f['test/x1'], f['test/x2'], f['test/y'])

    model.fit(x=[x1_tr, x2_tr], y=y_tr,
              batch_size=128,
              epochs=1,
              # callbacks=callback,
              validation_split=0.05,
              shuffle='batch')
