import h5py
from keras.models import Model
from keras import layers
import os
import argparse
from keras import optimizers


parser = argparse.ArgumentParser()
parser.add_argument('maxlen', help='maximum protein length', type=int)
parser.add_argument('ppi_path', help='path to the main folder', type=str)
args = parser.parse_args()

sequence_length = args.maxlen
ppi_path = args.ppi_path

embedding_dim = 16
lstm_units = 64
batch_size = 128

# Shared embedding, convolutional, pooling and LSTM layers
embedding = layers.Embedding(input_dim=21, output_dim=embedding_dim)
conv1 = layers.Conv1D(filters=16, kernel_size=4, activation='relu')
conv2 = layers.Conv1D(filters=16, kernel_size=8, activation='relu')
conv3 = layers.Conv1D(filters=16, kernel_size=16, activation='relu')
maxpooling = layers.MaxPooling1D(pool_size=4)
lstm = layers.Bidirectional(layers.LSTM(units=lstm_units))

input1 = layers.Input(shape=(sequence_length,), name='input1')
embedding1 = embedding(input1)
embedding1 = conv1(embedding1)
embedding1 = conv2(embedding1)
embedding1 = conv3(embedding1)
embedding1 = maxpooling(embedding1)
# embedding1 = layers.Flatten()(embedding1)
embedding1 = lstm(embedding1)

input2 = layers.Input(shape=(sequence_length,), name='input2')
embedding2 = embedding(input2)
embedding2 = conv1(embedding2)
embedding2 = conv2(embedding2)
embedding2 = conv3(embedding2)
embedding2 = maxpooling(embedding2)
# embedding2 = layers.Flatten()(embedding2)
embedding2 = lstm(embedding2)

concatenated = layers.concatenate([embedding1, embedding2], axis=-1)
hidden1 = layers.Dense(128, activation='relu')(concatenated)
hidden1 = layers.Dropout(0.3)(hidden1)
predictions = layers.Dense(1, activation='sigmoid')(hidden1)

model = Model([input1, input2], predictions)
adam = optimizers.Adam(lr=0.0005)  # clipnorm=3.0)
model.compile(optimizer=adam,
              loss='binary_crossentropy',
              metrics=['acc'])

# # callback = [
# #     TensorBoard(
# #         log_dir='tb_logs/lstm',
# #         batch_size=batch_size,
# #         histogram_freq=1.,
# #         write_grads=True,
# #         write_graph=False
# #     )
# # ]

data_file = os.path.join(
    ppi_path, ''.join(['output/create_tokenized_dataset_',
                       str(sequence_length), '_master.hdf5'])
)
with h5py.File(data_file, 'r') as f:
    x1_tr, x2_tr, y_tr = (f['train/x1'], f['train/x2'], f['train/y'])
    x1_val, x2_val, y_val = (f['val/x1'], f['val/x2'], f['val/y'])
    x1_te, x2_te, y_te = (f['test/x1'], f['test/x2'], f['test/y'])

    model.fit([x1_tr, x2_tr], y_tr,
              batch_size=batch_size,
              epochs=50,
              shuffle="batch",
              # callbacks=callback,
              validation_data=([x1_val, x2_val], y_val))
