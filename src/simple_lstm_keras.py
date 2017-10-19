import h5py
from keras.models import Model
import keras.layers


sequence_length = 500
embedding_dim = 16
gru_units = 64

# Shared embedding and GRU layers
embedding = keras.layers.Embedding(input_dim=21, output_dim=embedding_dim)
gru = keras.layers.GRU(units=gru_units)

input1 = keras.layers.Input(shape=(sequence_length,), name='input1')
input2 = keras.layers.Input(shape=(sequence_length,), name='input2')

# Create a shared embedding for the two inputs
embedding1 = embedding(input1)
embedding2 = embedding(input2)

# encoding1 = keras.layers.Flatten()(embedding1)
# encoding2 = keras.layers.Flatten()(embedding2)

encoding1 = gru(embedding1)
encoding2 = gru(embedding2)

concatenated = keras.layers.concatenate([encoding1, encoding2], axis=-1)
hidden1 = keras.layers.Dense(128, activation='relu')(concatenated)
hidden1 = keras.layers.Dropout(0.5)(hidden1)
hidden2 = keras.layers.Dense(64, activation='relu')(hidden1)
hidden2 = keras.layers.Dropout(0.5)(hidden2)
predictions = keras.layers.Dense(1, activation='sigmoid')(hidden2)

model = Model([input1, input2], predictions)

adam = keras.optimizers.Adam(lr=0.001)

model.compile(optimizer=adam,
              loss='binary_crossentropy',
              metrics=['acc'])

# callback = [
#     keras.callbacks.TensorBoard(
#         log_dir='tb_logs/gru',
#         histogram_freq=0.1)
# ]

with h5py.File('output/create_tokenized_dataset_500.hdf5', 'r') as f:
    x1_tr, x2_tr, y_tr = (f['train/x1'], f['train/x2'], f['train/y'])
    x1_te, x2_te, y_te = (f['test/x1'], f['test/x2'], f['test/y'])

    model.fit(x=[x1_tr, x2_tr], y=y_tr,
              # batch_size=128,
              epochs=1,
              steps_per_epoch=500,
              # callbacks=callback,
              validation_split=0.05),
