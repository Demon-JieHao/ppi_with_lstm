import h5py
from keras.models import Model
import keras.layers
import os


sequence_length = 500
embedding_dim = 16
lstm_units = 32

ppi_path = '/lustre/scratch/dariogi1/ppi_with_lstm'
# ppi_path = '/home/giovenko/DeepLearning/ppi_with_lstm'

# Shared embedding and LSTM layers
embedding = keras.layers.Embedding(input_dim=21,
                                   mask_zero=True,
                                   output_dim=embedding_dim)
lstm = keras.layers.LSTM(units=lstm_units)

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

model.compile(optimizer=adam,
              loss='binary_crossentropy',
              metrics=['acc'])

# callback = [
#     keras.callbacks.TensorBoard(
#         log_dir='tb_logs/lstm',
#         histogram_freq=0.1)
# ]

data_file = os.path.join(ppi_path, 'output/create_tokenized_dataset_500_master.hdf5')
with h5py.File(data_file, 'r') as f:
    x1_tr, x2_tr, y_tr = (f['train/x1'], f['train/x2'], f['train/y'])
    x1_val, x2_val, y_val = (f['val/x1'], f['val/x2'], f['val/y'])
    x1_te, x2_te, y_te = (f['test/x1'], f['test/x2'], f['test/y'])

    model.fit(x=[x1_tr, x2_tr], y=y_tr,
              batch_size=32,
              epochs=3,
              shuffle=False,
              # callbacks=callback,
              validation_data=([x1_val, x2_val], y_val)
              )
