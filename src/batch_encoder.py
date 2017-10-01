import numpy as np


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


def batch_generator(x1, x2, y, seq_len1, seq_len2, batch_size, how,
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
        seq_len1_batch = seq_len1[start:end]
        seq_len2_batch = seq_len2[start:end]
        yield batch1, batch2, ybatch, seq_len1_batch, seq_len2_batch
        start += batch_size


def create_datasets(X1, X2, Y, len_X1, len_X2, n_dev_test=10000):
    X1_train, X2_train, Y_train = (X1[:-2*n_dev_test],
                                   X2[:-2*n_dev_test],
                                   Y[:-2*n_dev_test])
    X1_dev, X2_dev, Y_dev = (X1[-2*n_dev_test:-n_dev_test],
                             X2[-2*n_dev_test:-n_dev_test],
                             Y[-2*n_dev_test:-n_dev_test])
    X1_test, X2_test, Y_test = (X1[-n_dev_test:],
                                X2[-n_dev_test:],
                                Y[-n_dev_test:])
    len_X1_train, len_X2_train = (len_X1[:-2*n_dev_test],
                                  len_X2[:-2*n_dev_test])
    len_X1_dev, len_X2_dev = (len_X1[-2*n_dev_test:-n_dev_test],
                              len_X2[-2*n_dev_test:-n_dev_test])
    len_X1_test, len_X2_test = (len_X1[-n_dev_test:],
                                len_X2[-n_dev_test:])

    return (X1_train, X2_train, Y_train,
            X1_dev, X2_dev, Y_dev,
            X1_test, X2_test, Y_test,
            len_X1_train, len_X2_train,
            len_X1_dev, len_X2_dev,
            len_X1_test, len_X2_test)
