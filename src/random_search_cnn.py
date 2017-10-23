from __future__ import print_function, division, absolute_import
import numpy as np
import h5py
from keras.callbacks import TensorBoard
import os
from create_cnn_model import cnn_model


ppi_path = '/home/giovenko/DeepLearning/ppi_with_lstm'

n_classes = 20
sequence_length = 500
input_shape = (sequence_length,)

n_conv_layers = np.arange(1, 3)  # N. of convolutions per branch
n_filters = np.array([32, 64])  # N. of kernels. Doubles at each convolution
kernel_size = np.array([6, 12, 24])  # Convolutional kernel size
pooling_size_multiplier = np.array([0.3, 0.5])
final_global_pooling = np.array([True, False])
learning_rates = np.logspace(-5, -3, 3)
n_hidden_layers = np.arange(1, 4)
n_units = np.array([128, 256])
dropout_rate = np.array([0.3, 0.4, 0.5])
activation_fun = 'relu'
batch_size = 128

f = h5py.File(
    os.path.join(ppi_path, 'output/create_tokenized_dataset_500.hdf5'), 'r'
)

x1_tr = f['train/x1']
x2_tr = f['train/x2']
y_tr = f['train/y']

input_dim = x1_tr.shape[1]

for _ in range(24):
    print("Iteration {}".format(_))

    conv_layers = np.random.choice(n_conv_layers)
    filters = np.random.choice(n_filters)
    size_kernel = np.random.choice(kernel_size)
    global_pooling = np.random.choice(final_global_pooling)
    learning_rate = np.random.choice(learning_rates)
    hidden_layers = np.random.choice(n_hidden_layers)
    units = np.random.choice(n_units)
    dropout = np.random.choice(dropout_rate)
    pooling_multiplier = np.random.choice(pooling_size_multiplier)

    print("N. of conv layers  : {}".format(conv_layers))
    print("N. of filters      : {}".format(filters))
    print("Kernel size        : {}".format(size_kernel))
    print("Pooling multiplier : {}".format(pooling_multiplier))
    print("Global pooling     : {}".format(global_pooling))
    print("Learning rate      : {}".format(learning_rate))
    print("N. of hidden layers: {}".format(hidden_layers))
    print("N. of hidden units : {}".format(units))
    print("Dropout rate       : {}".format(dropout))

    # Define log directory for TensorBoard. The directory name contains all the
    # parameters of the model.
    logdir = [str(conv_layers),
              str(filters),
              str(size_kernel),
              str(pooling_multiplier),
              str(global_pooling),
              str(learning_rate),
              str(hidden_layers),
              str(units),
              str(dropout),
              str(sequence_length),
              str(n_classes)]

    logdir = 'tb_logs/cnn/' + '_'.join(logdir)
    tensorboard = [
        TensorBoard(log_dir=logdir,
                    batch_size=batch_size,
                    write_graph=False)
    ]

    model = cnn_model(conv_layers,
                      filters,
                      size_kernel,
                      pooling_multiplier,
                      global_pooling,
                      learning_rate,
                      hidden_layers,
                      units,
                      dropout,
                      sequence_length,
                      n_classes)

    model.fit(x=[x1_tr, x2_tr], y=y_tr,
              batch_size=batch_size,
              epochs=50,
              callbacks=tensorboard,
              validation_split=0.05)
