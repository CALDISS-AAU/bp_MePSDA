# Packages
import os
from os.path import join
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import openpyxl

# Defining output path
output_dir = '/work/Ccp-MePSDA/output/keyword_search'

# Loading data
bts_df = pd.read_csv('/work/Ccp-MePSDA/output/collected_data/bts_df.csv')
bts_df = bts_df[['source', 'title', 'chunk_index', 'chunked']]
bts_df['chunked'] = bts_df['chunked'].astype(str)

# Loading calculated embeddings
bts_embeddings = np.load('/work/Ccp-MePSDA/modelling/embeddings/bts_embeddings.npy')

## Add to df
bts_df['chunk_embedding'] = list(bts_embeddings)

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
bts_df['theme1_score'] = bts_df['chunked'].apply(lambda text: score_text(text, keywords.get('theme1')))
bts_df['theme2_score'] = bts_df['chunked'].apply(lambda text: score_text(text, keywords.get('theme2')))
bts_df['theme3_score'] = bts_df['chunked'].apply(lambda text: score_text(text, keywords.get('theme3')))

# Top 5 representation chunks
theme1_top = bts_df.sort_values('theme1_score', ascending=False)['chunked'].tolist()[:5]
theme2_top = bts_df.sort_values('theme2_score', ascending=False)['chunked'].tolist()[:5]
theme3_top = bts_df.sort_values('theme3_score', ascending=False)['chunked'].tolist()[:5]

# Theme embeddings
theme1_embedding = calculate_mean_embedding(theme1_top)
theme2_embedding = calculate_mean_embedding(theme2_top)
theme3_embedding = calculate_mean_embedding(theme3_top)

# Theme similarity
bts_df['theme1_similarity'] = bts_df['chunk_embedding'].apply(lambda emb: cosine_similarity([emb], [theme1_embedding])[0][0])
bts_df['theme2_similarity'] = bts_df['chunk_embedding'].apply(lambda emb: cosine_similarity([emb], [theme2_embedding])[0][0])
bts_df['theme3_similarity'] = bts_df['chunk_embedding'].apply(lambda emb: cosine_similarity([emb], [theme3_embedding])[0][0])

# Output df with embeddings
bts_df.to_csv(join(output_dir, 'bts_chunk_keywordsimilarity.csv'), index = False)

# Top chunks
theme1_top = bts_df.sort_values('theme1_similarity', ascending = False).reset_index(drop = True).loc[:500, ['source', 'title', 'chunk_index', 'chunked', 'theme1_score', 'theme1_similarity']].rename(columns = {'chunked': 'text'})
theme2_top = bts_df.sort_values('theme2_similarity', ascending = False).reset_index(drop = True).loc[:500, ['source', 'title', 'chunk_index', 'chunked', 'theme2_score', 'theme2_similarity']].rename(columns = {'chunked': 'text'})
theme3_top = bts_df.sort_values('theme3_similarity', ascending = False).reset_index(drop = True).loc[:500, ['source', 'title', 'chunk_index', 'chunked', 'theme3_score', 'theme3_similarity']].rename(columns = {'chunked': 'text'})

# Export
theme1_top.to_excel(join(output_dir, 'theme1_bts_top500.xlsx'), index = False)
theme2_top.to_excel(join(output_dir, 'theme2_bts_top500.xlsx'), index = False)
theme3_top.to_excel(join(output_dir, 'theme3_bts_top500.xlsx'), index = False)


# Extracting each topics representative sentences and separating into excel files
bts_top_rep = pd.read_excel('/work/Ccp-MePSDA/output/topics/bts_topic_representative-texts.xlsx')

df_dict = {g: d for g, d in bts_top_rep.groupby('Topic')}

df_dict = {f'df{i}': d for i, (g, d) in enumerate(bts_top_rep.groupby('Topic'))}

output_path='/work/Ccp-MePSDA/output/topics/bts_separated_topic'
os.makedirs(output_path, exist_ok=True)
for name, df, in df_dict.items():
    file_path = os.path.join(output_path, f"{name}_bts.csv")
    df.to_csv(file_path)