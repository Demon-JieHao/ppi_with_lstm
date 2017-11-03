from __future__ import print_function, division, absolute_import

import keras.backend
import keras.layers
import keras.optimizers
from keras.models import Model
from keras.constraints import max_norm
import keras.backend as K


def cnn_model(conv_layers,
              filters,
              size_kernel,
              pooling_multiplier,
              global_pooling,
              learning_rate,
              hidden_layers,
              units,
              dropout,
              sequence_length,
              embedding_dim):

    """ Create a siamese CNN model with or without a final global pooling
    layer.

    This function generates siamese CNN models to be used in a random search.

    Args:
        conv_layers (int)    : The number of convolutional layers in
                               each layer.
        filters (int)        : Number of filter maps. Doubles at each layer.
        size_kernel (int)    : The width of the convolutional layer.
        pooling_multiplier (float): The multiplicative factor by which we must
                                    multiply the size_kernel to obtain the
                                    width of the pooling window.
        global_pooling (bool): Shall we add a global max pooling layer as the
                               last layer?
        learning_rate (float): The learning rate used by the optimizer.
        hidden_layers (int)  : The number of fully connected layers after the
                               concatenation.
        units (int) :          The number of hidden units in the fully
                               connected layers. It's the same for each
                               fully connected layer.
        dropout (float)      : The dropout to apply to each fully connected
                               layer. It's the same for each fully connected
                               layer.
        sequence_length (int): The length of the sequences.
        embedding_dim (int)  : The embedding dimension.

    Returns:
           A keras model (type `keras.engine.training.Model`)
    """
    keras.backend.clear_session()

    # These two parameters must be integer 1-tuples.
    pool_size = (int(size_kernel * pooling_multiplier),)
    size_kernel = (int(size_kernel),)

    input_layer = {}
    embedding = {}
    convolutions = {}
    maxpoolings = {}

    embedding_layer = keras.layers.Embedding(
        input_dim=21, output_dim=embedding_dim)
    # Initialize the input layeres and one-hot-encode them
    for i in range(2):
        input_layer[i] = keras.layers.Input(shape=(sequence_length,),
                                            dtype='int32',
                                            name='input' + str(i))
        # Shared embeddings for the two inputs
        embedding[i] = embedding_layer(input_layer[i])

    # Create one or more convolutional and maxpooling layers
    for k in range(conv_layers - 1):
        convolutions[k] = keras.layers.Conv1D(filters,
                                              size_kernel,
                                              activation='relu')
        maxpoolings[k] = keras.layers.MaxPooling1D(pool_size)
        # Double the number of filters at each new convolution
        filters *= 2

    # Create the last convolutional layer
    convolutions[conv_layers - 1] = keras.layers.Conv1D(filters,
                                                        size_kernel,
                                                        activation='relu')

    # import ipdb; ipdb.set_trace()

    # Apply the convolution and the maxpooling to the two inputs
    for k in range(conv_layers - 1):
        for i in range(2):
            embedding[i] = convolutions[k](embedding[i])
            embedding[i] = maxpoolings[k](embedding[i])

    # Apply the last convolutional layer, and the type of maxpooling selected
    # with the `global_pooling` option (global or local)
    for i in range(2):
        embedding[i] = convolutions[conv_layers - 1](embedding[i])
        if global_pooling:
            maxpoolings[conv_layers - 1] = keras.layers.GlobalMaxPooling1D()
            embedding[i] = maxpoolings[conv_layers - 1](embedding[i])
        else:
            maxpoolings[conv_layers - 1] = keras.layers.MaxPooling1D()
            embedding[i] = maxpoolings[conv_layers - 1](embedding[i])
            embedding[i] = keras.layers.Flatten()(embedding[i])

    hidden = keras.layers.concatenate([embedding[0], embedding[1]], axis=-1)

    for _ in range(hidden_layers):
        hidden = keras.layers.Dense(units, activation='relu')(hidden)
        hidden = keras.layers.Dropout(dropout)(hidden)

    output = keras.layers.Dense(units=1, activation='sigmoid')(hidden)

    model = Model([input_layer[0], input_layer[1]], output)

    adam = keras.optimizers.adam(lr=learning_rate)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['acc'])
    return model


def cnn_gru_model(conv_layers,
                  filters,
                  size_kernel,
                  pool_size,
                  gru_states,
                  learning_rate,
                  hidden_layers,
                  units,
                  dropout,
                  sequence_length,
                  n_classes):
    """

    :param conv_layers: Number of convolutional layers in the two branches.
    :param filters: Number of filters in the convolutional layers.
    :param size_kernel: Size of the convolutional layer.
    :param pool_size: Size of the pooling window.
    :param gru_states:
    :param learning_rate: Learning rate.
    :param hidden_layers: Number of hidden layers.
    :param units: Number of units in the hidden layers.
    :param dropout: Dropout rate in the hidden layers.
    :param sequence_length: Maximum length of the input sequences.
    :param n_classes: Number of classes.
    :return:
    """

    keras.backend.clear_session()

    output_shape = (sequence_length, n_classes)

    # These two parameters must be integer 1-tuples.
    size_kernel = (int(size_kernel),)

    one_hot_encoder = keras.layers.Lambda(
        K.one_hot, arguments={'num_classes': n_classes},
        output_shape=output_shape
    )

    input_layer = {}
    embedding = {}
    convolutions = {}

    # Create one or more convolutional layers
    for k in range(conv_layers):
        convolutions[k] = keras.layers.Conv1D(filters,
                                              size_kernel,
                                              activation='relu')
        # Double the number of filters at each new convolution
        filters *= 2

    # Create one single local max pooling layer
    maxpooling = keras.layers.MaxPooling1D(pool_size)

    # Shared Bidirectional GRU Layer
    bi_gru = keras.layers.Bidirectional(
        keras.layers.GRU(gru_states,
                         kernel_constraint=max_norm(1.0)))

    for i in range(2):
        input_layer[i] = keras.layers.Input(shape=(sequence_length,),
                                            dtype='int32',
                                            name='input' + str(i))
        embedding[i] = one_hot_encoder(input_layer[i])
        for k in range(conv_layers):
            embedding[i] = convolutions[k](embedding[i])

    # Add the max-pooling and flatten
    for i in range(2):
        embedding[i] = maxpooling(embedding[i])
        embedding[i] = bi_gru(embedding[i])
        # embedding[i] = maxpooling(embedding[i])
        # embedding[i] = keras.layers.Flatten()(embedding[i])

    hidden = keras.layers.concatenate([embedding[0], embedding[1]], axis=-1)

    for _ in range(hidden_layers):
        hidden = keras.layers.Dense(units, activation='relu')(hidden)
        hidden = keras.layers.Dropout(dropout)(hidden)

    output = keras.layers.Dense(units=1, activation='sigmoid')(hidden)

    model = Model([input_layer[0], input_layer[1]], output)

    adam = keras.optimizers.Adam(lr=learning_rate)

    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['acc'])
    return model


def mlp_model(n_shared_layers,
              shared_units,
              learning_rate,
              hidden_layers,
              units,
              dropout,
              embedding_dim,
              input_length=500,
              input_dim=21):
    """

    :param n_shared_layers: Number of shared hidden layer in the two branches.
    :param shared_units: Number of units in the shared hidden layers.
    :param learning_rate: Learning rate.
    :param hidden_layers: Number of dense layers after concatenation.
    :param units: Number of units in the hidden layers.
    :param dropout: Dropout rate.
    :param embedding_dim: Embedding dimension.
    :param input_length: Maximum length of the input sequences.
    :param input_dim: Number of classes (+ 1).
    :return: A Keras model.
    """

    keras.backend.clear_session()

    embedding = keras.layers.Embedding(input_dim=input_dim,
                                       output_dim=embedding_dim,
                                       input_length=input_length)
    shared_layers = {}

    input1 = keras.layers.Input(shape=(input_length,), name='input1')
    input2 = keras.layers.Input(shape=(input_length,), name='input2')

    embedding1 = embedding(input1)
    embedding2 = embedding(input2)

    embedding1 = keras.layers.Flatten()(embedding1)
    embedding2 = keras.layers.Flatten()(embedding2)

    for k in range(n_shared_layers):
        shared_layers[k] = keras.layers.Dense(units=shared_units,
                                              activation='relu')
        embedding1 = shared_layers[k](embedding1)
        embedding1 = keras.layers.Dropout(dropout)(embedding1)
        embedding2 = shared_layers[k](embedding2)
        embedding2 = keras.layers.Dropout(dropout)(embedding2)

    hidden = keras.layers.concatenate([embedding1, embedding2],
                                      axis=-1)
    for _ in range(hidden_layers):
        hidden = keras.layers.Dense(units, activation='relu')(hidden)
        hidden = keras.layers.Dropout(dropout)(hidden)

    output = keras.layers.Dense(units=1, activation='sigmoid')(hidden)

    model = Model([input1, input2], output)

    adam = keras.optimizers.adam(lr=learning_rate)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['acc'])
    return model


def mlp_fp_model(n_shared_layers,
                 shared_units,
                 learning_rate,
                 hidden_layers,
                 units,
                 dropout,
                 input_dim):
    """
    Model constructor for performing random searches on the ProtR FPs.

    :param n_shared_layers: Integer. Number of layers with shared weights in
                            the two branches.
    :param shared_units: Integer. Number of units in the shared layers.
    :param learning_rate: Float. The learning rate.
    :param hidden_layers: Integer. The number of hidden layers.
    :param units: Integer. The number of units in the hidden layers.
    :param dropout: Float. The dropout rate.
    :param input_dim: Integer. The number of input features.
    :return:
    """
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
