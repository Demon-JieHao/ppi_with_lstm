from __future__ import print_function, division, absolute_import

import pandas as pd
from keras.models import Model, load_model
from sklearn.manifold import TSNE
import numpy as np

sequences = pd.read_hdf('output/filtered_tokenized_ppi_dataset_500_.hdf5')
model_emb = load_model('models/best_embedding_model')

# Define a new model that reads an input tensor and retunrns the embeddings.
embedding = Model(inputs=model_emb.input,
                  outputs=model_emb.get_layer('dropout_3').output)

x = sequences.values
protein_embeddings = embedding.predict([x, x], batch_size=128)

tsne = TSNE()
x_tsne = tsne.fit_transform(x)
np.save('output/run_tsne_on_embeddings.npz', x_tsne)
