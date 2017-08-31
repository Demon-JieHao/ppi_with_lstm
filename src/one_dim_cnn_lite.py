import numpy as np
import gzip
import pickle
import tensorflow as tf
from datetime import datetime


def encoder(protein, how, prot_max_len, num_aa):
    z = np.zeros((prot_max_len, num_aa), dtype='float32')
    z[np.arange(len(protein)), protein] = 1
    if how == 'cnn':
        z = z.T
    return z


def batch_encoder(protein_list, how, prot_max_len, num_aa):
    num_proteins = len(protein_list)
    if how == 'ohe':
        z = np.zeros((num_proteins, prot_max_len, num_aa), dtype='float32')
    else:
        z = np.zeros((num_proteins, num_aa, prot_max_len), dtype='float32')
    for i in np.arange(num_proteins):
        z[i] = encoder(protein_list[i], how, prot_max_len, num_aa)
    return z


def batch_generator(x1, x2, y, batch_size, how,
                    prot_max_len, num_aa):

    num_iterations = int(np.ceil(len(x1) / batch_size))
    start = 0
    for iteration in range(num_iterations):
        end = start + batch_size
        input_batch1 = x1[start:end]
        input_batch2 = x2[start:end]
        ybatch = y[start:end]
        batch1 = batch_encoder(input_batch1, how, prot_max_len, num_aa)
        batch2 = batch_encoder(input_batch2, how, prot_max_len, num_aa)
        yield batch1, batch2, ybatch
        start += batch_size


now = datetime.utcnow().strftime('%Y%m%d%H%M%S')
log_dir = 'tf_logs/cnn_1d' + now

max_protein_length = 500
n_aas = 21

conv1_fmaps = 16
conv1_ksize = 5
conv1_stride = 1
conv1_pad = 'VALID'

conv2_fmaps = 16
conv2_ksize = 10
conv2_stride = 2
conv2_pad = 'VALID'

n_fc1 = 128

n_epochs = 5
learning_rate = 0.01
batch_size = 128

pool_size = 10
max_pool_strides = 10

dropout_rate1 = 0.5
dropout_rate2 = 0.4

he_init = tf.contrib.layers.variance_scaling_initializer()

# Load the dataset still in a list form
X1, X2, Y = pickle.load(gzip.open('../output/create_dataset.pkl.gzip', 'r'))
Y = Y.reshape(Y.shape[0], 1)

# Create training, dev and test set
X1_train, X2_train, Y_train = X1[:-20000], X2[:-20000], Y[:-20000]
X1_dev, X2_dev, Y_dev = X1[-20000:-10000], X2[-20000:-10000], Y[-20000:-10000]
X1_test, X2_test, Y_test = X1[-10000:], X2[-10000:], Y[-10000:]

# We create placeholders for the 1D CNN, where each sequence has length equal
# to the maximum protein length and a number of channels equal to the number
# of allowed aminoacids
tf.reset_default_graph()

x1 = tf.placeholder(dtype=tf.float32,
                    shape=(None, max_protein_length, n_aas),
                    name='x1')
x2 = tf.placeholder(dtype=tf.float32,
                    shape=(None, max_protein_length, n_aas),
                    name='x2')
y = tf.placeholder(tf.int64, shape=(None, 1), name='y')
training = tf.placeholder(tf.bool, shape=(), name='dropout')

# We create two convolutional layers that are forced to have the same weights
with tf.name_scope('first_conv'):
    conv1 = tf.layers.conv1d(x1, filters=conv1_fmaps, kernel_size=conv1_ksize,
                             strides=conv1_stride, padding=conv1_pad,
                             kernel_initializer=he_init,
                             activation=tf.nn.relu, name='conv1')

    conv2 = tf.layers.conv1d(x2, filters=conv1_fmaps, kernel_size=conv1_ksize,
                             strides=conv1_stride, padding=conv1_pad,
                             kernel_initializer=he_init,
                             activation=tf.nn.relu, name='conv1', reuse=True)

with tf.name_scope('second_conv'):
    conv3 = tf.layers.conv1d(conv1, filters=conv1_fmaps,
                             kernel_size=conv2_ksize,
                             strides=conv1_stride, padding=conv1_pad,
                             kernel_initializer=he_init,
                             activation=tf.nn.relu, name='conv3')

    conv4 = tf.layers.conv1d(conv2, filters=conv1_fmaps,
                             kernel_size=conv2_ksize,
                             strides=conv1_stride, padding=conv1_pad,
                             kernel_initializer=he_init,
                             activation=tf.nn.relu, name='conv3', reuse=True)

with tf.name_scope('merge'):
    merged = tf.reduce_mean([conv3, conv4], axis=0)

with tf.name_scope('max_pool'):
    max_pool = tf.layers.max_pooling1d(merged, pool_size=pool_size,
                                       strides=max_pool_strides,
                                       name='max_pool')
    max_pool_shape = max_pool.get_shape()
    max_pool_flat = tf.reshape(
        max_pool,
        (-1, max_pool_shape[1].value * max_pool_shape[2].value))

with tf.name_scope('fully_connected'):
    hidden1 = tf.layers.dense(max_pool_flat, units=128, activation=tf.nn.relu,
                              kernel_initializer=he_init, name='hidden1')
    dropout1 = tf.layers.dropout(hidden1, rate=dropout_rate1,
                                 name='dropout1', training=training)
    hidden2 = tf.layers.dense(dropout1, units=64, activation=tf.nn.relu,
                              kernel_initializer=he_init, name='hidden2')
    dropout2 = tf.layers.dropout(hidden2, rate=dropout_rate2,
                                 name='dropout2', training=training)
    logits = tf.layers.dense(dropout2, units=1,
                             kernel_initializer=he_init, name='logits')

with tf.name_scope('loss'):
    xentropy = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.cast(y, tf.float32), logits=logits)
    loss = tf.reduce_mean(xentropy)

with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)
    prediction = tf.cast(tf.greater(logits, 0.5), dtype=tf.int64)
    correct = tf.equal(prediction, y)
    accuracy = tf.reduce_mean(tf.cast(correct, dtype=tf.float32))

init = tf.global_variables_initializer()
accuracy_summary = tf.summary.scalar('accuracy', accuracy)
train_writer = tf.summary.FileWriter(log_dir + '_train',
                                     tf.get_default_graph())


with tf.Session() as sess:
    sess.run(init)
    n_batches = int(np.ceil(len(Y_train) / batch_size))
    step = 0

    for epoch in range(n_epochs):
        print('epoch: {}'.format(epoch))
        batchgen = batch_generator(X1_train, X2_train,
                                   Y_train, batch_size,
                                   'ohe', 500, 21)
        for x1b, x2b, yb in batchgen:
            if step % 100 == 0:
                # train_acc = accuracy_summary.eval(
                #     feed_dict={x1: x1b, x2: x2b, y: yb, training: True})
                # train_writer.add_summary(train_acc, step)
                acc, current_loss = sess.run(
                    [accuracy, loss],
                    feed_dict={x1: x1b, x2: x2b, y: yb, training: True})
                print('Step=', step, 'Training accuracy=', acc,
                      'Training loss=', current_loss)
            step += 1
            sess.run(training_op,
                     feed_dict={x1: x1b, x2: x2b, y: yb, training: True})

        # Full pass over the dev set
        n_dev_batches = int(np.ceil(len(Y_dev) / batch_size))
        dev_batchgen = batch_generator(X1_dev, X2_dev, Y_dev, batch_size,
                                       'ohe', 500, 21)
        mean_dev_acc = 0
        for x1d, x2d, yd in dev_batchgen:
            dev_acc = sess.run(accuracy,
                               feed_dict={x1: x1d, x2: x2d, y: yd,
                                          training: False})
            mean_dev_acc += dev_acc
        print('Test accuracy =', mean_dev_acc/n_dev_batches)
