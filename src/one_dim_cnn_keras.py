import h5py
from keras.models import Model
import keras.layers

input_dim = 21
input_length = 500
output_dim = 16

callbacks = [
    keras.callbacks.TensorBoard(
        log_dir='my_log_dir',
        histogram_freq=0.1,
        embeddings_freq=0.1
    )
]

# Shared embedding and LSTM layers
embedding = keras.layers.Embedding(input_dim=input_dim,
                                   output_dim=output_dim,
                                   input_length=input_length)
first_conv = keras.layers.Conv1D(32, 7, activation='relu')
second_conv = keras.layers.Conv1D(32, 7, activation='relu')

input1 = keras.layers.Input(shape=(input_length,),
                            dtype='float32', name='input1')
input2 = keras.layers.Input(shape=(input_length,),
                            dtype='float32', name='input2')

# Create a shared embedding for the two inputs
embedding1 = embedding(input1)
embedding2 = embedding(input2)

z1 = first_conv(embedding1)
z2 = first_conv(embedding2)
z1 = second_conv(z1)
z2 = second_conv(z2)
z1 = keras.layers.MaxPooling1D(5)(z1)
z2 = keras.layers.MaxPooling1D(5)(z2)
f1 = keras.layers.Flatten()(z1)
f2 = keras.layers.Flatten()(z2)

concatenated = keras.layers.concatenate([f1, f2], axis=-1)
hidden1 = keras.layers.Dense(32, activation='relu')(concatenated)
hidden1 = keras.layers.Dropout(0.5)(hidden1)
predictions = keras.layers.Dense(1, activation='sigmoid')(hidden1)

model = Model([input1, input2], predictions)


model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])

with h5py.File('../output/create_tokenized_dataset.hdf5', 'r') as f:
    x1_train, x2_train, y_train = (f['train/x1'], f['train/x2'], f['train/y'])
    x1_test, x2_test, y_test = (f['test/x1'], f['test/x2'], f['test/y'])

    model.fit(x=[x1_train, x2_train], y=y_train,
              batch_size=128,
              epochs=15,
              validation_split=0.05,
              shuffle='batch',
              callbacks=callbacks)
