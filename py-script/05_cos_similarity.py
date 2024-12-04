# Packages
import transformers
from sentence_transformers import SentenceTransformer
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
import os
import numpy as np

# Loading calculated embeddings
risvig_embeddings = np.load('/work/Ccp-MePSDA/modelling/embeddings/embeddings.npy')
keyword_embeddings = np.load('/work/Ccp-MePSDA/modelling/embeddings/list_embeddings.npy')

mean_embedding = np.mean(keyword_embeddings, axis=0)

cos_sim = sentence_transformers.util.cos_sim(mean_embedding, risvig_embeddings)




# Reshape mean_embedding to be 2D to match the input requirements of cosine_similarity
mean_embedding_reshaped = mean_embedding.reshape(1, -1)

# Calculate cosine similarities between mean_embedding and risvig_embeddings
cossims = cosine_similarity(mean_embedding_reshaped, risvig_embeddings)

# Flatten for 1D array for similarity purposes
cossims = cossims.flatten()