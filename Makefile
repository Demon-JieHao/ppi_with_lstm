.PHONY: all clean

vpath %.py src

%.stmp: %.py
	python $<
	touch $@

all:	random_search_cnn.done\
	random_forest_on_protein_fps.stmp\
	create_tokenized_dataset_500.stmp\
	create_protein_fp_dataset_500.stmp


# Random search on the 1D CNN models
random_search_cnn.done: random_search_cnn.py
	CUDA_VISIBLE_DEVICES=0 python $<
	touch $@

# Run a RF classifier
random_forest_on_protein_fps.stmp: create_random_forest_dataset.stmp

# Create a protein FP dataset suitable for a random forest classifier
create_random_forest_dataset.stmp: normalized_protein_fp.stmp

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
	rm -f output/*
