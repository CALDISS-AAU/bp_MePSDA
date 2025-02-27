#!/usr/bin/env python
# coding: utf-8

# importing packages

# Basic python packages
import pandas as pd
import numpy as np
import bertopic
import os

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
import topicwizard
from topicwizard.compatibility import BERTopicWrapper
from topicwizard.figures import topic_map
import plotly.express as px
from topicwizard.figures import *



# Loading data
mepsda_df = pd.read_csv('/work/Ccp-MePSDA/output/collected_data/mepsda_df.csv')
#mepsda_df.drop(columns='index', inplace=True)
mepsda_df['chunked'] = mepsda_df['chunked'].astype(str)
# select columns
mepsda_df = mepsda_df[['source', 'title', 'chunk_index', 'chunked']]


# Load from directory
# Defining embedding model
embedding_model = SentenceTransformer('intfloat/multilingual-e5-small')
topic_model = BERTopic.load("/work/Ccp-MePSDA/modelling/model/mepsda_bertopic", embedding_model=embedding_model)

# Creating wrapper for topicwizard
wrapped_model = BERTopicWrapper(topic_model)

# Calculating hierarchy for topics
hierarchical_topics = topic_model.hierarchical_topics(mepsda_df['chunked'])
# Creating hiearchy plot
topic_hierarchy = topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics)

# Show
topic_hierarchy.show()

# Saving to folder
topic_hierarchy.write_html('/work/Ccp-MePSDA/output/plots/topic_hierarchy.html')

# Loading embeddings
embeddings = np.load('/work/Ccp-MePSDA/modelling/embeddings/embeddings.npy')

# Reduce dimensionality of embeddings, this step is optional but much faster to perform iteratively:
reduced_embeddings = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)
hierarchy_doc_topic = topic_model.visualize_hierarchical_documents(mepsda_df['chunked'], hierarchical_topics, reduced_embeddings=reduced_embeddings)

# Saving to folder
hierarchy_doc_topic.write_html('/work/Ccp-MePSDA/output/plots/hierarchy_doc_topic.html')


# Creating term rank plot
term_rank = topic_model.visualize_term_rank()
# Saving to folder
term_rank.write_html('/work/Ccp-MePSDA/output/plots/term_rank.html')


# Creating heatmap
topic_model.visualize_heatmap(n_clusters=40, width=1000, height=1000)


# # Topic-wizard viz

# Produce a TopicData object for persistance or figures.
topic_data = wrapped_model.prepare_topic_data(mepsda_df['chunked'])


wizard_bar = topic_barcharts(topic_data, top_n=10)
# Add a title
wizard_bar.update_layout(
    title="Top Terms Across Topics",
    title_font_size=26
)

# Change chart size
wizard_bar.update_layout(
    height=1300,
    width=1500
)

# Adjust bar width
wizard_bar.update_traces(
    width=0.95
)
# show and write
wizard_bar.show()
wizard_bar.write_html('/work/Ccp-MePSDA/output/plots/wizard_bar.html')

word_map = word_map(topic_data)  # Generate the figure

word_map.show()  # Display the plot

word_map.write_html('/work/Ccp-MePSDA/output/plots/word_map.html')

# Document map plot
document_map = document_map(topic_data)
document_map.show()

document_map.write_html('/work/Ccp-MePSDA/output/plots/document_map.html')

# Keyword similarity histogram
keyword_sim = pd.read_csv('/work/Ccp-MePSDA/output/keyword_search/mepsda_chunk_keywordsimilarity.csv')
fig = go.Figure()

# Add the histogram for theme 1
fig.add_trace(go.Histogram(
    x=keyword_sim['theme1_similarity'],
    opacity=0.6,
    name='Theme 1'
))

# Add the histogram for theme 2
fig.add_trace(go.Histogram(
    x=keyword_sim['theme2_similarity'],
    opacity=0.6,
    name='Theme 2'
))

# Add the histogram for theme 3
fig.add_trace(go.Histogram(
    x=keyword_sim['theme3_similarity'],
    opacity=0.6,
    name='Theme 3'
))

# Update the layout
fig.update_layout(
    title='Distribution of Similarity Scores by Keyword Theme',
    title_font_size=22,
    xaxis_title='Cosine Similarity',
    yaxis_title='Frequency',
    legend_title_text='Theme'
)

# Show the plot
fig.show()
fig.write_html('/work/Ccp-MePSDA/output/plots/histogram.html')
