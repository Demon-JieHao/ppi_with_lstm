import numpy as np


def encoder(protein, how, prot_max_len, num_aa):
    z = np.zeros((prot_max_len, num_aa))
    z[np.arange(len(protein)), protein] = 1
    if how == 'cnn':
        z = z.T
    return z



def batch_encoder(protein_list, how, prot_max_len, num_aa):
    num_proteins = len(protein_list)
    z = np.zeros((num_proteins, prot_max_len, num_aa))
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
        yield [batch1, batch2], ybatch
        start += batch_size


# def batch_encoder(protein_list, how, prot_max_len, num_aa):
#     results = np.array([encoder(protein, how, prot_max_len, num_aa)
#                         for protein in protein_list])
#     return results


# def batch_generator(x1, x2, y, batch_size, how,
#                     prot_max_len, num_aa):

#     num_iterations = int(np.ceil(len(x1) / batch_size))
#     start = 0
#     for iteration in range(num_iterations):
#         end = start + batch_size
#         input_batch1 = x1[start:end]
#         input_batch2 = x2[start:end]
#         ybatch = y[start:end]
#         batch1 = batch_encoder(input_batch1, how, prot_max_len, num_aa)
#         batch2 = batch_encoder(input_batch2, how, prot_max_len, num_aa)
#         yield [batch1, batch2], ybatch
#         start += batch_size
