#!/usr/bin/env python
# coding: utf-8

# Importing packages

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
import topicwizard
from topicwizard.compatibility import BERTopicWrapper
from topicwizard.figures import topic_map
import plotly.express as px
import plotly.graph_objects as go
from topicwizard.figures import *


output_dir= '/work/Ccp-MePSDA/output/plots'

# Loading data
bts_df = pd.read_csv('/work/Ccp-MePSDA/output/collected_data/bts_df.csv')
bts_df['chunked'] = bts_df['chunked'].astype(str)
# Select columns
bts_df = bts_df[['source', 'title', 'chunk_index', 'chunked']]


# Load from directory
# Defining embedding model
embedding_model = SentenceTransformer('intfloat/multilingual-e5-small')
topic_model = BERTopic.load("/work/Ccp-MePSDA/modelling/model/bts_bertopic", embedding_model=embedding_model)

# creating wrapper for topicwizard
wrapped_model = BERTopicWrapper(topic_model)

topic_model.get_topic_info()


# Further reduce topics
topic_model.reduce_topics(bts_df['chunked'], nr_topics=21)

# Saving once again after merging
topic_model.save("/work/Ccp-MePSDA/modelling/model/bts_bertopic", serialization="safetensors", save_ctfidf=True, save_embedding_model=embedding_model)

# Creating hierarchical topics
hierarchical_topics = topic_model.hierarchical_topics(bts_df['chunked'])

topic_hierarchy = topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics)

topic_hierarchy.update_layout(
    title="Hierarchical Clusering BTS-material",
    title_font_size=26
)
topic_hierarchy.show()
topic_hierarchy.write_html(join(output_dir, 'bts_topic_hiearchy.html'))


# Loading embeddings
embeddings = np.load('/work/Ccp-MePSDA/modelling/embeddings/bts_embeddings.npy')
# Run the visualization with the original embeddings

# Reduce dimensionality of embeddings, this step is optional but much faster to perform iteratively:
reduced_embeddings = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)
hierarchy_doc_topic = topic_model.visualize_hierarchical_documents(bts_df['chunked'], hierarchical_topics, reduced_embeddings=reduced_embeddings)
hierarchy_doc_topic.update_layout(
    title='hierarchical doc mapping for BTS-material',
    title_font_size=26
)
hierarchy_doc_topic.write_html('/work/Ccp-MePSDA/output/plots/hierarchy_doc_bts.html')


term_rank = topic_model.visualize_term_rank()

term_rank.write_html('/work/Ccp-MePSDA/output/plots/bts_term_rank.html')


# Topic Wizard plots

# Produce a TopicData object for persistance or figures.
topic_data = wrapped_model.prepare_topic_data(bts_df['chunked'])

wizard_bar = topic_barcharts(topic_data, top_n=8)
# Add a title
wizard_bar.update_layout(
    title="Top Terms Across Topics for BTS-material",
    title_font_size=26
)

# Change chart size
wizard_bar.update_layout(
    height=1500,
    width=2000
)

# Adjust bar width
wizard_bar.update_traces(
   width=1
)

wizard_bar.show()
wizard_bar.write_html('/work/Ccp-MePSDA/output/plots/bts_wizard_bar.html')

# Create wordmap
word_map = word_map(topic_data)  # Generate the figure
word_map.show()  # Display the plot

word_map.write_html('/work/Ccp-MePSDA/output/plots/bts_word_map.html')

# Document map
document_map = document_map(topic_data)
document_map.show()


document_map.write_html('/work/Ccp-MePSDA/output/plots/bts_document_map.html')


keyword_sim = pd.read_csv('/work/Ccp-MePSDA/output/keyword_search/bts_chunk_keywordsimilarity.csv')

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
    title='Distribution of BTS Similarity Scores by Keyword Theme',
    title_font_size=22,
    xaxis_title='Cosine Similarity',
    yaxis_title='Frequency',
    legend_title_text='Theme'
)

# Show the plot
fig.show()
fig.write_html('/work/Ccp-MePSDA/output/plots/bts_histogram.html')

