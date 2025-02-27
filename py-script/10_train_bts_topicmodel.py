# importing packages

# Basic python packages
import pandas as pd
import numpy as np
import bertopic
import os
from os.path import join

# BERTopic related
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer

# Transformers packages
from sentence_transformers import SentenceTransformer
import transformers
# Handle parallelism for tokenizer
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# SpaCy
import spacy
from spacy.lang.da import Danish

# Viz packages
import matplotlib.pyplot as plt

# DIRS and PATHS
output_dir = '/work/Ccp-MePSDA/output/bts_topics'
os.makedirs(output_dir, exist_ok=True)

# Importing Spacy Stopwords
nlp = Danish()
stop_words = list(nlp.Defaults.stop_words)
stop_words.extend(['originalartiklen', 'originalartikel',
'ea670633','Originalartiklen',
'øh','øhm',
'sådan','ehm',
'æhm', 'Æhm',
'Ehm', 'Sådan', 'mmh', 'Mmh'])

# Loading data
bts_df = pd.read_csv('/work/Ccp-MePSDA/output/collected_data/bts_df.csv')
#mepsda_df.drop(columns='index', inplace=True)
bts_df['chunked'] = bts_df['chunked'].astype(str)

# Select columns
bts_df = bts_df[['source', 'title', 'chunk_index', 'chunked']]

# Defining embedding model
embedding_model = SentenceTransformer('intfloat/multilingual-e5-small')
# Loading pre-trained embeddings
embeddings = np.load('/work/Ccp-MePSDA/modelling/embeddings/bts_embeddings.npy')

# Define Umap cluster parameters
umap_model = UMAP(n_neighbors=2,
n_components=20,
metric='cosine',
min_dist=0.04,
low_memory=False,
random_state=666)

# Defining hierarchical density based clustering model
hdbscan_model = HDBSCAN(
    min_cluster_size=12,
    cluster_selection_method='leaf',
    metric='euclidean',
    prediction_data=True)

# Define representation model
representation_model = KeyBERTInspired()

# Define CountVectorizer model
vectorizer_model = CountVectorizer(
    stop_words=stop_words, 
    min_df=0.01,
    max_df=2, 
    ngram_range=(1, 2))

# Iniate model
topic_model = BERTopic(
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    representation_model=representation_model,
    embedding_model=embedding_model,
    vectorizer_model=vectorizer_model,
    top_n_words=20,
    verbose=True)

# Run model on text column
topics, probs = topic_model.fit_transform(bts_df['chunked'], embeddings)

# Add topics, probs to data
data_topics = topic_model.get_document_info(bts_df['chunked'], bts_df)
data_topics['topic_prob'] = probs

# Topic info
topics_info = topic_model.get_topic_info()

# Save model
topic_model.save("/work/Ccp-MePSDA/modelling/model/bts_bertopic", serialization="safetensors", save_ctfidf=True, save_embedding_model=embedding_model)

# save data with topics
outn_data = join(output_dir, 'bts_chunk_topics.csv')
data_topics.to_csv(outn_data, index = False)

# save topic info
outn_topics = join(output_dir, 'bts_topic_info.csv')
topics_info.to_csv(outn_topics, index = False)