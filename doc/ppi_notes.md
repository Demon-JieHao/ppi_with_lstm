# Notes on the PPI Analysis

## Embedding and LSTMs

I spent some time trying to understand whether I should use a one-hot encoding
or an embedding layer. In the end it seems to me that the two things are very
similar. In principle one could hard-code an embedding layer that maps each
index to a one-hot encoded vector, and make it non-trainable. This should 
obtain the same results as a one-hot encoded input. If we decide to go for an 
embedding layer, the question is what dimension should it have. We may decide 
to preserve the original dimension, compress it, or even expand it.

## Summary of results

We want to collect the results from the various models tested so far, and thi