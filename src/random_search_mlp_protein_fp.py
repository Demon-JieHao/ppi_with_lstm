import numpy as np
import h5py
import keras.layers
from keras.models import Model


num_shared_hidden_layers = np.arange(1, 5)
size_shared_hidden_layer = np.array([256, 512])
learning_rates = np.logspace(-5, -3, 3)
n_hidden_layers = np.arange(1, 5)
dropout_rate = np.array([0.3, 0.4, 0.5])
activation_fun = 'relu'
n_units = np.array([128, 256, 512])

f = h5py.File('output/create_protein_fp_dataset_500.hdf5', 'r')
x1_tr = f['train/x1']
x2_tr = f['train/x2']
y_tr = f['train/y']

input_dim = x1_tr.shape[1]


def mlp_model(input_dim, learning_rate, hidden_layers, dropout, units,
              shared_units, num_shared_layers):

    keras.backend.clear_session()
    input1 = keras.layers.Input(shape=(input_dim,), name='input1')
    input2 = keras.layers.Input(shape=(input_dim,), name='input2')

    shared_hidden_layer = keras.layers.Dense(units=shared_units,
                                             activation='relu')
    shared_hidden_layer2 = keras.layers.Dense(units=shared_units,
                                              activation='relu')

    shared_hidden1 = shared_hidden_layer(input1)
    shared_hidden1 = keras.layers.Dropout(dropout)(shared_hidden1)
    shared_hidden2 = shared_hidden_layer(input2)
    shared_hidden2 = keras.layers.Dropout(dropout)(shared_hidden2)

    for _ in range(num_shared_layers):
        shared_hidden1 = shared_hidden_layer2(shared_hidden1)
        shared_hidden1 = keras.layers.Dropout(dropout)(shared_hidden1)
        shared_hidden2 = shared_hidden_layer2(shared_hidden2)
        shared_hidden2 = keras.layers.Dropout(dropout)(shared_hidden2)

    hidden = keras.layers.concatenate([shared_hidden1, shared_hidden2],
                                      axis=-1)
    for _ in range(hidden_layers):
        hidden = keras.layers.Dense(units, activation='relu')(hidden)
        hidden = keras.layers.Dropout(dropout)(hidden)

    output = keras.layers.Dense(units=1, activation='sigmoid')(hidden)

    model = Model([input1, input2], output)

    adam = keras.optimizers.adam(lr=learning_rate)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['acc'])
    return model


for _ in range(24):
    print("Iteration {}".format(_))

    num_shared_layers = np.random.choice(num_shared_hidden_layers)
    shared_units = np.random.choice(size_shared_hidden_layer)
    learning_rate = np.random.choice(learning_rates)
    hidden_layers = np.random.choice(n_hidden_layers)
    dropout = np.random.choice(dropout_rate)
    units = np.random.choice(n_units)

    print("N. of shared layers: {}".format(num_shared_layers))
    print("N. of shared units : {}".format(shared_units))
    print("N. of hidden layers: {}".format(hidden_layers))
    print("N. of hidden units : {}". format(units))
    print("Learning rate      : {}".format(learning_rate))
    print("Dropout rate       : {}".format(dropout))

    logdir = [str(num_shared_layers), str(shared_units),
              str(learning_rate), str(hidden_layers), str(dropout),
              str(units)]
    logdir = 'tb_logs/mlp_fp/' + '_'.join(logdir)

    tensorboard = [
        keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=0.1)
        ]

    model = mlp_model(input_dim=input_dim,
                      num_shared_layers=num_shared_layers,
                      shared_units=shared_units,
                      learning_rate=learning_rate,
                      hidden_layers=hidden_layers,
                      dropout=dropout,
                      units=units)
    model.fit(x=[x1_tr, x2_tr], y=y_tr,
              batch_size=128,
              epochs=50,
              validation_split=0.05,
              callbacks=tensorboard)
