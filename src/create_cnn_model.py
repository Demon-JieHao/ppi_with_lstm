from __future__ import print_function, division, absolute_import
import keras.layers
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
              n_classes):

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
        n_classes (int):       The number of classes (AAs).

    Returns:
           A keras model (type `keras.engine.training.Model`)
    """
    keras.backend.clear_session()

    output_shape = (sequence_length, n_classes)

    # These two parameters must be integer 1-tuples.
    pool_size = (int(size_kernel * pooling_multiplier),)
    size_kernel = (int(size_kernel),)

    one_hot_encoder = keras.layers.Lambda(
        K.one_hot, arguments={'num_classes': n_classes},
        output_shape=output_shape
    )

    input_layer = {}
    embedding = {}
    convolutions = {}
    maxpoolings = {}

    # Initialize the input layeres and one-hot-encode them
    for i in range(2):
        input_layer[i] = keras.layers.Input(shape=(sequence_length,),
                                            dtype='int32',
                                            name='input' + str(i))
        # Shared embeddings for the two inputs
        embedding[i] = one_hot_encoder(input_layer[i])

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
