.PHONY: all clean

vpath %.py src

%.stmp: %.py
	python $<
	touch $@

all: 	create_tokenized_dataset_500.stmp\
	create_protein_fp_dataset_500.stmp


# Create an HDF5 dataset similar to the tokenized dataset 500
# but using the protein predictors rather than the AA indices.
create_protein_fp_dataset_500.stmp: normalized_protein_fp.stmp

# Using the output of protr.R, normalize the protein predictors
# to zero mean and unit variance.
normalized_protein_fp.stmp:

# Create an HDF5 dataset compososed of a training and test set.
# Each set contains x1, x2 and y.
create_tokenized_dataset_500.stmp: filtered_ppi_dataset_500.stmp

# Filter the dataset based on min and max length and keep only the
# canonical aminoacids. Here the max length is 500.
filtered_ppi_dataset_500.stmp: full_ppi_dataset.stmp

# Create a pandas data frame with uniprot IDs and sequences for each
# member of a pair.
full_ppi_dataset.stmp: 

clean:
	rm -f *.stmp
	rm -f output/*.hdf5
