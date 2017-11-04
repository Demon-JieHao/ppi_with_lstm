import h5py
import keras.layers
import numpy as np

from .create_models import mlp_model

n_shared_hidden_layers = np.arange(1, 5)
size_shared_hidden_layer = np.array([256, 512])
learning_rates = np.logspace(-5, -3, 3)
n_hidden_layers = np.arange(1, 5)
dropout_rate = np.array([0.3, 0.4, 0.5])
activation_fun = 'relu'
n_units = np.array([128, 256, 512])
output_dim = np.array([8, 16, 32, 64])

f = h5py.File('output/create_tokenized_dataset_500.hdf5', 'r')
x1_tr = f['train/x1']
x2_tr = f['train/x2']
y_tr = f['train/y']

input_dim = x1_tr.shape[1]

for _ in range(16):
    print("Iteration {}".format(_))

    n_shared_layers = np.random.choice(n_shared_hidden_layers)
    shared_units = np.random.choice(size_shared_hidden_layer)
    learning_rate = np.random.choice(learning_rates)
    hidden_layers = np.random.choice(n_hidden_layers)
    units = np.random.choice(n_units)
    dropout = np.random.choice(dropout_rate)
    embedding_dim = np.random.choice(output_dim)

    print("N. of shared layers: {}".format(n_shared_layers))
    print("N. of shared units : {}".format(shared_units))
    print("N. of hidden layers: {}".format(hidden_layers))
    print("N. of hidden units : {}".format(units))
    print("Learning rate      : {}".format(learning_rate))
    print("Dropout rate       : {}".format(dropout))
    print("Embedding dimension: {}".format(embedding_dim))

    logdir = [str(n_shared_layers),
              str(shared_units),
              str(learning_rate),
              str(hidden_layers),
              str(units),
              str(dropout),
              str(embedding_dim)]
    logdir = 'tb_logs/embeddings/' + '_'.join(logdir)

    tensorboard = [
        keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=0.1)
        ]

    model = mlp_model(n_shared_layers,
                      shared_units,
                      learning_rate,
                      hidden_layers,
                      units,
                      dropout,
                      embedding_dim)

    model.fit(x=[x1_tr, x2_tr], y=y_tr,
              batch_size=128,
              epochs=40,
              validation_split=0.05,
              callbacks=tensorboard)
