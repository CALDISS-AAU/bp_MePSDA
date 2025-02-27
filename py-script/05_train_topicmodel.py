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
# handle parallelism for tokenizer
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# SpaCy
import spacy
from spacy.lang.da import Danish

# Viz packages
import matplotlib.pyplot as plt

# Output dir
output_dir = '/work/Ccp-MePSDA/output/topics'

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
mepsda_df = pd.read_csv('/work/Ccp-MePSDA/output/collected_data/mepsda_df.csv')
#mepsda_df.drop(columns='index', inplace=True)
mepsda_df['chunked'] = mepsda_df['chunked'].astype(str)

# Select columns
mepsda_df = mepsda_df[['source', 'title', 'chunk_index', 'chunked']]

# Defining embedding model
embedding_model = SentenceTransformer('intfloat/multilingual-e5-small')
# Loading pre-trained embeddings
embeddings = np.load('/work/Ccp-MePSDA/modelling/embeddings/embeddings.npy')

# Define Umap cluster parameters
umap_model = UMAP(n_neighbors=3,
n_components=20,
metric='cosine',
min_dist=0.04,
low_memory=False,
random_state=420)

# Defining hierarchical density based clustering model
hdbscan_model = HDBSCAN(
    min_cluster_size=100,
    cluster_selection_method='leaf',
    metric='euclidean',
    prediction_data=True)

# Define representation model
representation_model = KeyBERTInspired()

# Define CountVectorizer model
vectorizer_model = CountVectorizer(
    stop_words=stop_words, 
    min_df=2, 
    max_df=0.85, 
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
topics, probs = topic_model.fit_transform(mepsda_df['chunked'], embeddings)

# Add topics, probs to data
data_topics = topic_model.get_document_info(mepsda_df['chunked'], mepsda_df)
data_topics['topic_prob'] = probs

# Topic info
topics_info = topic_model.get_topic_info()

# Save model
topic_model.save("/work/Ccp-MePSDA/modelling/model/mepsda_bertopic", serialization="safetensors", save_ctfidf=True, save_embedding_model=embedding_model)

# Save data with topics
outn_data = join(output_dir, 'mepsda_chunk_topics.csv')
data_topics.to_csv(outn_data, index = False)

# Save topic info
outn_topics = join(output_dir, 'mepsda_topic_info.csv')
topics_info.to_csv(outn_topics, index = False)