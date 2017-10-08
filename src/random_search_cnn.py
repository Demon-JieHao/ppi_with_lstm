import numpy as np
import h5py
import keras.layers
from keras.models import Model


n_conv_layers = np.arange(1, 5)
n_filters = np.array([16, 32, 64])
kernel_size = np.array([4, 8, 16])
# The pooling size is chosen to be half of the kernel size
learning_rates = np.logspace(-5, -3, 3)
n_hidden_layers = np.arange(1, 5)
n_units = np.array([128, 256, 512])
dropout_rate = np.array([0.3, 0.4, 0.5])
activation_fun = 'relu'
embedding_dim = np.array([8, 16, 32])


f = h5py.File('output/create_tokenized_dataset_500.hdf5', 'r')
x1_tr = f['train/x1']
x2_tr = f['train/x2']
y_tr = f['train/y']

input_dim = x1_tr.shape[1]


def cnn_model(conv_layers,
              filters,
              size_kernel,
              learning_rate,
              hidden_layers,
              units,
              dropout,
              dim_embedding):

    keras.backend.clear_session()

    # These two parameters must be integer 1-tuples.
    pool_size = (int(size_kernel / 2),)
    size_kernel = (int(size_kernel),)
    
    embedding = keras.layers.Embedding(input_dim=21, output_dim=dim_embedding)
    convolutions = {}
    maxpoolings = {}
    input1 = keras.layers.Input(shape=(input_dim,), name='input1')
    input2 = keras.layers.Input(shape=(input_dim,), name='input2')

    embedding1 = embedding(input1)
    embedding2 = embedding(input2)

    for k in range(conv_layers):
        convolutions[k] = keras.layers.Conv1D(filters, size_kernel,
                                              activation='relu')
        maxpoolings[k] = keras.layers.MaxPooling1D(pool_size)
        embedding1 = convolutions[k](embedding1)
        embedding1 = maxpoolings[k](embedding1)
        embedding2 = convolutions[k](embedding2)
        embedding2 = maxpoolings[k](embedding2)

    embedding1 = keras.layers.Flatten()(embedding1)
    embedding2 = keras.layers.Flatten()(embedding2)

    hidden = keras.layers.concatenate([embedding1, embedding2], axis=-1)

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

    conv_layers = np.random.choice(n_conv_layers)
    filters = np.random.choice(n_filters)
    size_kernel = np.random.choice(kernel_size)
    learning_rate = np.random.choice(learning_rates)
    hidden_layers = np.random.choice(n_hidden_layers)
    units = np.random.choice(n_units)
    dropout = np.random.choice(dropout_rate)
    dim_embedding = np.random.choice(embedding_dim)

    print("N. of conv layers  : {}".format(conv_layers))
    print("N. of filters      : {}".format(filters))
    print("Kernel size        : {}".format(size_kernel))
    print("Learning rate      : {}".format(learning_rate))
    print("N. of hidden layers: {}".format(hidden_layers))
    print("N. of hidden units : {}". format(units))
    print("Dropout rate       : {}".format(dropout))
    print("Embedding dimension: {}".format(dim_embedding))

    # Define log directory for TensorBoard. The directory name contains all the
    # parameters of the model.
    logdir = [str(conv_layers),
              str(filters),
              str(size_kernel),
              str(learning_rate),
              str(hidden_layers),
              str(units),
              str(dropout),
              str(dim_embedding)]
    logdir = 'tb_logs/cnn/' + '_'.join(logdir)
    tensorboard = [
        keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=0.1)
        ]

    model = cnn_model(conv_layers,
                      filters,
                      size_kernel,
                      learning_rate,
                      hidden_layers,
                      units,
                      dropout,
                      dim_embedding)

    model.fit(x=[x1_tr, x2_tr], y=y_tr,
              batch_size=128,
              epochs=40,
              validation_split=0.05,
              callbacks=tensorboard)
