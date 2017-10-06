import h5py
from keras.models import Model
import keras.layers

input_dim = 20
input_length = 500
output_dim = 16
encoding_dim = 256

# Shared embedding and LSTM layers
embedding = keras.layers.Embedding(input_dim=input_dim,
                                   output_dim=output_dim,
                                   input_length=input_length)
lstm = keras.layers.LSTM(units=encoding_dim)

input1 = keras.layers.Input(shape=(input_length,),
                            dtype='float32', name='input1')
input2 = keras.layers.Input(shape=(input_length,),
                            dtype='float32', name='input2')

# Create a shared embedding for the two inputs
embedding1 = embedding(input1)
embedding2 = embedding(input2)

encoding1 = lstm(embedding1)
encoding2 = lstm(embedding2)

concatenated = keras.layers.concatenate([encoding1, encoding2], axis=-1)
hidden1 = keras.layers.Dense(256, activation='relu')(concatenated)
hidden1 = keras.layers.Dropout(0.5)(hidden1)
hidden2 = keras.layers.Dense(128, activation='relu')(hidden1)
hidden2 = keras.layers.Dropout(0.5)(hidden2)
predictions = keras.layers.Dense(1, activation='sigmoid')(hidden2)

model = Model([input1, input2], predictions)


model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])

with h5py.File('../output/create_tokenized_dataset.hdf5', 'r') as f:
    x1_train, x2_train, y_train = (f['train/x1'], f['train/x2'], f['train/y'])
    x1_test, x2_test, y_test = (f['test/x1'], f['test/x2'], f['test/y'])

    model.fit(x=[x1_train, x2_train], y=y_train,
              batch_size=128,
              epochs=5,
              validation_split=0.05,
              shuffle='batch')
