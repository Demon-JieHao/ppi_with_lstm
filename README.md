# Experimenting with Deep Learning for PPI prediction #

Florian Kiefer has put together a balanced dataset of protein pairs with a binary variable indicating whether the two proteins are interacting or not. In this project I compare the performance of Multi-Layer Perceptrons (MLP), one-dimensional Convolutional Neural Networks (CNN) and Recurrent Neural Networks (RNN) on this prediction task. The steps of the analysis are summarized in the `Makefile`. In brief, they are:

1. The creation of the "full dataset", i.e., all the protein pairs are stored in a single pandas data frame in HDF5 format, containing the uniprot IDs, the sequences, and the interaction variable.
2. Filtering of the dataset. For convenience, especially with RNNs, we consider a subset of the original dataset containing only the protein pairs where each protein has a length 50 <= l <= 500, and contains only the 20 canonical aminoacid. The output of this step is another, smaller, pandas data frame in HDF5 format.
3. Tokenization. The 20 aminoacids are associated with integer values from 1 to 20. A protein is then represented as a list of integers. The proteins that are shorter than 500 are padded with zeros. After shuffling the dataset, we create a training-dev set and a test set with the last 10,000 pairs, which is stored in an HDF5 file. The dataset contans a `train` and a `test` subsets.
4. Calculate the protein predictors (NOT IN THE MAKEFILE YET). The `protr.R` file reads all the sequences and computes ~2800 protein predictors that are stored in a TSV file.
5. Normalization of the protein predictors: the predictors are on different scales and need to be standardized.
6. Creation of a dataset based on the protein predictors. This dataset is also stored in an HDF5 file, and has the same structure as the tokenized one.
7. Random search of the hyper-parameters for a model based on the protein predictors. This script fits 16 models with random combinations of various hiperparameters and stored the TensorBoard logs in a folder for later inspection.
8. Random search on the hyper-parameters for a model based on the tokinization. Same as above, but using the integer indices rather than the protein predictors.

### How do I get set up? ###

* You need the protein sequences and the PPI dataset from Florian Kiefer.
* These scripts have been tested on the latest versions of Keras, using TensorFlow as a backend.
* You need the h5py library for the creation of the HDF5 training and test datasets. 
* NOTE: Running the current Makefile will not work because the generation of the protein predictors is not covered (it takes ~10 hours), and it has been tested on a different hardware (explicit calls to CUDA devices).

### Contribution guidelines ###

* Feel free to write comments in the wiki of open issues.
* If you need more information, drop me a line.

### Who do I talk to? ###

* Giovanni d'Ario <giovanni.dario@gmail.com>
