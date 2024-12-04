# Transformers packages
import transformers
from sentence_transformers import SentenceTransformer
import torch
# handle parallelism for tokenizer
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np
import pandas as pd
import os
# Loading data
mepsda_df = pd.read_csv('/work/Ccp-MePSDA/output/collected_data/mepsda_df.csv')
#mepsda_df.drop(columns='index', inplace=True)
mepsda_df['chunked'] = mepsda_df['chunked'].astype(str)

# Pre-calculate embeddings
embedding_model = SentenceTransformer("Maltehb/danish-bert-botxo")
embeddings = embedding_model.encode(mepsda_df['chunked'].tolist(), show_progress_bar=True)

embeddings_tensor = torch.tensor(embeddings)

# Saving Embeddings
torch.save(embeddings,'/work/Ccp-MePSDA/modelling/embeddings/data_embeddings.pt')

# Calculations for keyword list

with open('/work/Ccp-MePSDA/data/keyword.txt') as f:
    seed_words = f.readlines()

list_embed = embedding_model.encode(seed_words, show_progress_bar=True)

list_tensor = torch.tensor(list_embed)

torch.save(list_embed,'/work/Ccp-MePSDA/modelling/embeddings/list_embeddings.pt')
