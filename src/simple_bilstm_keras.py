import h5py
from keras.models import Model
# from keras.callbacks import TensorBoard
import keras.layers
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('maxlen', help='maximum protein length', type=int)
parser.add_argument('ppi_path', help='path to the main folder', type=str)
args = parser.parse_args()

sequence_length = args.maxlen
ppi_path = args.ppi_path

embedding_dim = 16
lstm_units = 24
batch_size = 128

# Shared embedding and LSTM layers
embedding = keras.layers.Embedding(input_dim=21,
                                   mask_zero=True,
                                   output_dim=embedding_dim)
lstm = keras.layers.Bidirectional(keras.layers.GRU(units=lstm_units))

input1 = keras.layers.Input(shape=(sequence_length,), name='input1')
input2 = keras.layers.Input(shape=(sequence_length,), name='input2')

# Create a shared embedding for the two inputs
embedding1 = embedding(input1)
embedding2 = embedding(input2)

# encoding1 = keras.layers.Flatten()(embedding1)
# encoding2 = keras.layers.Flatten()(embedding2)

encoding1 = lstm(embedding1)
encoding2 = lstm(embedding2)

concatenated = keras.layers.concatenate([encoding1, encoding2], axis=-1)
hidden1 = keras.layers.Dense(64, activation='relu')(concatenated)
hidden1 = keras.layers.Dropout(0.2)(hidden1)
# hidden2 = keras.layers.Dense(64, activation='relu')(hidden1)
# hidden2 = keras.layers.Dropout(0.5)(hidden2)
predictions = keras.layers.Dense(1, activation='sigmoid')(hidden1)

model = Model([input1, input2], predictions)

adam = keras.optimizers.Adam(lr=0.001)

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])

# callback = [
#     TensorBoard(
#         log_dir='tb_logs/lstm',
#         batch_size=batch_size,
#         histogram_freq=1.,
#         write_grads=True,
#         write_graph=False
#     )
# ]

data_file = os.path.join(
    ppi_path, ''.join(['output/create_tokenized_dataset_',
                       str(sequence_length), '_master.hdf5'])
)
with h5py.File(data_file, 'r') as f:
    x1_tr, x2_tr, y_tr = (f['train/x1'], f['train/x2'], f['train/y'])
    x1_val, x2_val, y_val = (f['val/x1'], f['val/x2'], f['val/y'])
    x1_te, x2_te, y_te = (f['test/x1'], f['test/x2'], f['test/y'])

    x1_t, x2_t, y_t = x1_tr[:50000], x2_tr[:50000], y_tr[:50000]
    x1_v, x2_v, y_v = x1_val[...], x2_val[...], y_val[...]

    model.fit([x1_t, x2_t], y_t,
              batch_size=batch_size,
              epochs=30,
              shuffle=False,
              # callbacks=callback,
              validation_data=([x1_v, x2_v], y_v))
