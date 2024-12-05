import os
from os.path import join
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# dirs and paths
output_dir = '/work/Ccp-MePSDA/output/keyword_search'

# Loading data
mepsda_df = pd.read_csv('/work/Ccp-MePSDA/output/collected_data/mepsda_df.csv')
mepsda_df = mepsda_df[['source', 'title', 'chunk_index', 'chunked']]
#mepsda_df.drop(columns='index', inplace=True)
mepsda_df['chunked'] = mepsda_df['chunked'].astype(str)

# Loading calculated embeddings
risvig_embeddings = np.load('/work/Ccp-MePSDA/modelling/embeddings/embeddings.npy')

## add to df
mepsda_df['chunk_embedding'] = list(risvig_embeddings)

# Load sentence transformer
model = SentenceTransformer('intfloat/multilingual-e5-small')  

# Keywords
keywords = {
    'theme1': [
        'pige',
        'dreng*',
        'tøj*',
        'ven*',
        'diversitet',
        'ung*',
        'seksu*',
        'sex',
        'stereotyp*',
        'mand*',
        'kvinde*',
        'maskulin*',
        'feminin*'
        ],
    'theme2': [
        'improvis*',
        'cast*',
        'skuespil*',
        'skabe*',
        'målgruppe',
        'kritik*',
        'drøm*',
        'film*',
        'serie*',
        'drama*',
        'leg*',
        'værktøj',
        'håndværk',
        'inspir*',
        'lær*'
    ],
    'theme3': [
        'location',
        'Silkeborg',
        'Jylland',
        'Nordsjælland',
        'verden',
        'univers',
        'by',
        'provins*',
        'gymnasie*',
        'miljø',
        'virkelig*',
        'autenti*',
        'filter'
    ]
}

# Function to score text based on number of keywords (including wildcards)
def score_text(text, keywords):
    score = 0
    for keyword in keywords:
        if '*' in keyword:
            base = keyword.replace('*', '')
            score += sum(1 for word in text.split() if word.startswith(base))
        else:
            score += text.split().count(keyword)
    return score

# Function to calculate the mean embedding of a list of strings (sentences)
def calculate_mean_embedding(sentences, model = model):
    embeddings = model.encode(sentences)
    return np.mean(embeddings, axis=0)

# Keyword scores
mepsda_df['theme1_score'] = mepsda_df['chunked'].apply(lambda text: score_text(text, keywords.get('theme1')))
mepsda_df['theme2_score'] = mepsda_df['chunked'].apply(lambda text: score_text(text, keywords.get('theme2')))
mepsda_df['theme3_score'] = mepsda_df['chunked'].apply(lambda text: score_text(text, keywords.get('theme3')))

# top 5 representation chunks
theme1_top = mepsda_df.sort_values('theme1_score', ascending=False)['chunked'].tolist()[:5]
theme2_top = mepsda_df.sort_values('theme2_score', ascending=False)['chunked'].tolist()[:5]
theme3_top = mepsda_df.sort_values('theme3_score', ascending=False)['chunked'].tolist()[:5]

# theme embeddings
theme1_embedding = calculate_mean_embedding(theme1_top)
theme2_embedding = calculate_mean_embedding(theme2_top)
theme3_embedding = calculate_mean_embedding(theme3_top)

# theme similarity
mepsda_df['theme1_similarity'] = mepsda_df['chunk_embedding'].apply(lambda emb: cosine_similarity([emb], [theme1_embedding])[0][0])
mepsda_df['theme2_similarity'] = mepsda_df['chunk_embedding'].apply(lambda emb: cosine_similarity([emb], [theme2_embedding])[0][0])
mepsda_df['theme3_similarity'] = mepsda_df['chunk_embedding'].apply(lambda emb: cosine_similarity([emb], [theme3_embedding])[0][0])

# output df with embeddings
mepsda_df.to_csv(join(output_dir, 'mepsda_chunk_keywordsimilarity.csv'), index = False)

# top chunks
theme1_top = mepsda_df.sort_values('theme1_similarity', ascending = False).reset_index(drop = True).loc[:500, ['source', 'title', 'chunk_index', 'chunked', 'theme1_score', 'theme1_similarity']].rename(columns = {'chunked': 'text'})
theme2_top = mepsda_df.sort_values('theme2_similarity', ascending = False).reset_index(drop = True).loc[:500, ['source', 'title', 'chunk_index', 'chunked', 'theme2_score', 'theme2_similarity']].rename(columns = {'chunked': 'text'})
theme3_top = mepsda_df.sort_values('theme3_similarity', ascending = False).reset_index(drop = True).loc[:500, ['source', 'title', 'chunk_index', 'chunked', 'theme3_score', 'theme3_similarity']].rename(columns = {'chunked': 'text'})

# export
theme1_top.to_excel(join(output_dir, 'theme1_texts_top500.xlsx'), index = False)
theme2_top.to_excel(join(output_dir, 'theme2_texts_top500.xlsx'), index = False)
theme3_top.to_excel(join(output_dir, 'theme3_texts_top500.xlsx'), index = False)