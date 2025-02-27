# Packages
import numpy as np
import pandas as pd
import os
# Handle parallelism for tokenizer
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Transformers packages
import transformers
from sentence_transformers import SentenceTransformer
import torch

# Loading data
bts_df = pd.read_csv('/work/Ccp-MePSDA/output/collected_data/bts_df.csv')
# Mepsda_df.drop(columns='index', inplace=True)
bts_df['chunked'] = bts_df['chunked'].astype(str)

# Pre-calculate embeddings
embedding_model = SentenceTransformer("intfloat/multilingual-e5-small")
embeddings = embedding_model.encode(bts_df['chunked'].tolist(), show_progress_bar=True)

embeddings_tensor = torch.tensor(embeddings)
embeddings = embeddings_tensor.numpy()

# Saving Embeddings
np.save("/work/Ccp-MePSDA/modelling/embeddings/bts_embeddings.npy", embeddings)