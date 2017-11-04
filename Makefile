.PHONY: all clean

DIR=/home/giovenko/DeepLearning/ppi_with_lstm
SEQLEN=500
MERGED=False

vpath %.py src

%.stmp: %.py
	python $< $(SEQLEN) $(DIR)
	touch $@

all:    create_tokenized_dataset.stmp\
	    create_protein_fp_dataset.stmp


# Create an HDF5 dataset similar to the tokenized dataset 500
# but using the protein predictors rather than the AA indices.
create_protein_fp_dataset.stmp: normalized_protein_fp.stmp
    python $< $(SEQLEN) $(DIR) $(MERGED)

# Using the output of protr.R, normalize the protein predictors
# to zero mean and unit variance.
normalized_protein_fp.stmp:

# Create an HDF5 dataset compososed of a training and test set.
# Each set contains x1, x2 and y.
create_tokenized_dataset.stmp: filtered_ppi_dataset.stmp

# Filter the dataset based on min and max length and keep only the
# canonical aminoacids. Here the max length is 500.
filtered_ppi_dataset.stmp: full_ppi_dataset.stmp

# Create a pandas data frame with uniprot IDs and sequences for each
# member of a pair.
full_ppi_dataset.stmp: 

clean:
	rm -f *.stmp
	rm -f output/*.hdf5
