# README

## Dataset Generation

The `create_dataset.py` file creates a minimal dataset containing only the protein ids of the interacting proteins in each pair. The ids are mapped to a one-hot-encoded numpy array by the batch generator, and not before. This makes the dataset extremely light-weight, since the processing is taken care by the batch generator.

The `create_tokenized_dataset.py` does the opposite, and preprocess the whole dataset at once, returning a numpy array where each row represents a protein sequence encoded as an array of integers. So, if we have only three "letters" in our vocabulary, {'A', 'B', 'C'} and a sequence 'ABCCBA' the representation will be [1, 2, 3, 3, 2, 1] (given a max-length of 6). Note that the first letter is mapped to 1 and not to 0.
