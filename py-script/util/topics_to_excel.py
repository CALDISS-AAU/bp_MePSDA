#!/usr/bin/env python
# coding: utf-8

import os
from os.path import join
import pandas as pd

# dirs and pats
output_dir = '/work/Ccp-MePSDA/output/topics'
output_bts = '/work/Ccp-MePSDA/output/bts_topics'

# read csvs
chunks_topics_df = pd.read_csv(join(output_dir, 'mepsda_chunk_topics.csv'))
bts_chunk_topics_df = pd.read_csv(join(output_bts, 'bts_chunk_topics.csv'))
topics_info_df = pd.read_csv(join(output_dir, 'mepsda_topic_info.csv'))
bts_topics_info = pd.read_csv(join(output_bts, 'bts_topic_info.csv'))

# filter representative docs
keep_cols = ['source', 'title', 'chunk_index', 'chunked', 'Topic', 'Name', 'Top_n_words', 'topic_prob']
topic_docs = chunks_topics_df.loc[chunks_topics_df['Representative_document'], keep_cols].rename(columns = {'chunked':'text'})
bts_topics_info = bts_chunk_topics_df.loc[chunks_topics_df['Representative_document'], keep_cols].rename(columns = {'chunked':'text'})

# write to excel
topic_docs.to_excel(join(output_dir, 'mepsda_topic_representative-texts.xlsx'), index=False)
bts_topics_info.to_excel(join(output_dir, 'bts_topic_representative-texts.xlsx'), index=False)
topics_info_df.to_excel(join(output_dir, 'mepsda_topic_info.xlsx'), index=False)
bts_topics_info.to_excel(join(output_bts,'bts_topic_info.xlsx'), index=False)