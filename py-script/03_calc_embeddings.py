# Packages
import transformers
from sentence_transformers import SentenceTransformer
# Handle parallelism for tokenizer
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np
import pandas as pd
import os

# Loading data
mepsda_df = pd.read_csv('/work/Ccp-MePSDA/data/collected_data/mepsda_df.csv')
#mepsda_df.drop(columns='index', inplace=True)
mepsda_df['chunked'] = mepsda_df['chunked'].astype(str)

# Pre-calculate embeddings
embedding_model = SentenceTransformer("intfloat/multilingual-e5-small")
embeddings = embedding_model.encode(mepsda_df['chunked'].tolist(), show_progress_bar=True)

# Saving Embeddings
np.save("/work/Ccp-MePSDA/modelling/embeddings/embeddings.npy", embeddings)