import numpy as np
import h5py
import keras.layers
from keras.models import Model


n_shared_hidden_layers = np.arange(1, 5)
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

n_features = x1_tr.shape[1]


def mlp_model(n_shared_layers,
              shared_units,
              learning_rate,
              hidden_layers,
              units,
              dropout,
              input_dim=n_features):

    keras.backend.clear_session()
    shared_layers = {}

    # import ipdb; ipdb.set_trace()

    input1 = keras.layers.Input(shape=(input_dim,), name='input1')
    input2 = keras.layers.Input(shape=(input_dim,), name='input2')

    shared_layers[0] = keras.layers.Dense(units=shared_units,
                                          activation='relu')

    dense1 = shared_layers[0](input1)
    dense1 = keras.layers.Dropout(dropout)(dense1)
    dense2 = shared_layers[0](input2)
    dense2 = keras.layers.Dropout(dropout)(dense2)

    for k in range(1, n_shared_layers):
        shared_layers[k] = keras.layers.Dense(units=shared_units,
                                              activation='relu')
        dense1 = shared_layers[k](dense1)
        dense1 = keras.layers.Dropout(dropout)(dense1)
        dense2 = shared_layers[k](dense2)
        dense2 = keras.layers.Dropout(dropout)(dense2)

    hidden = keras.layers.concatenate([dense1, dense2], axis=-1)

    for _ in range(hidden_layers):
        hidden = keras.layers.Dense(units, activation='relu')(hidden)
        hidden = keras.layers.Dropout(dropout)(hidden)

    output = keras.layers.Dense(units=1, activation='sigmoid')(hidden)

    model = Model([input1, input2], output)

    adam = keras.optimizers.adam(lr=learning_rate)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['acc'])
    return model


for _ in range(16):
    print("Iteration {}".format(_))

    n_shared_layers = np.random.choice(n_shared_hidden_layers)
    shared_units = np.random.choice(size_shared_hidden_layer)
    learning_rate = np.random.choice(learning_rates)
    hidden_layers = np.random.choice(n_hidden_layers)
    units = np.random.choice(n_units)
    dropout = np.random.choice(dropout_rate)

    print("N. of shared layers: {}".format(n_shared_layers))
    print("N. of shared units : {}".format(shared_units))
    print("Learning rate      : {}".format(learning_rate))
    print("N. of hidden layers: {}".format(hidden_layers))
    print("N. of hidden units : {}". format(units))
    print("Dropout rate       : {}".format(dropout))

    logdir = [str(n_shared_layers),
              str(shared_units),
              str(learning_rate),
              str(hidden_layers),
              str(units),
              str(dropout)]
    logdir = 'tb_logs/mlp_fp/' + '_'.join(logdir)

    tensorboard = [
        keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=0.1)
        ]

    model = mlp_model(n_shared_layers,
                      shared_units,
                      learning_rate,
                      hidden_layers,
                      units,
                      dropout)

    model.fit(x=[x1_tr, x2_tr], y=y_tr,
              batch_size=128,
              epochs=40,
              validation_split=0.05,
              callbacks=tensorboard)
