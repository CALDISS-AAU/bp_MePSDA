# Packages
import pandas as pd
import re
import os
import sys
from os.path import join
import glob
import numpy as np
import pysbd
from pysbd import Segmenter
from tqdm import tqdm

# DIRS AND PATHS
project_dir = join('/work', 'Ccp-MePSDA')
data_dir = join(project_dir, 'data')
modelling_dir = join(project_dir, 'modelling')
modules_dir = join(project_dir, 'modules')
sys.path.append(modules_dir)

from mepsda_funs import * # Reading project-functions

logs_dir = join(modelling_dir, 'logs')
output_dir = join(project_dir, 'output')
model_dir = join(modelling_dir, 'models')

data_transcribed = join(output_dir, 'transcript')

# Enable tqdm for pandas
tqdm.pandas()

# Set the minimum character limit for segmenter/chunking function
min_chars = 200

# Directory path containing kontra files
print('processing kontra...')

# Directory path containing the files
kontra_files = glob.glob('/work/Ccp-MePSDA/output/transcript/kontra_transcribed/*.csv')

# Create empty list
dataframes = []

# Iterate over each file and read it, skipping the first row (header)
for filename in kontra_files:
    df = pd.read_csv(filename)
    df_concat = pd.DataFrame({
        'full_text': [df['text'].str.cat(sep='. ')],
        'title': [os.path.basename(filename)], # creaing title based on filename
        'source': ['kontra'] # Type of interview
    })
    
    dataframes.append(df_concat)

# Concatenate all dataframes
kontra_df = pd.concat(dataframes, ignore_index=True)

# Converting to string
kontra_df['full_text'] = kontra_df['full_text'].astype('str')

# Apply the function to the text column and explode the result
kontra_df['chunked'] = kontra_df['full_text'].progress_apply(lambda x: split_text_into_chunks(x, min_chars))
kontra_df = kontra_df.explode("chunked").reset_index(drop=True)

# create chunk index
kontra_df['chunk_index'] = kontra_df.groupby('title').cumcount()

# To CSV
kontra_df.to_csv('/work/Ccp-MePSDA/output/collected_data/df_kontra.csv', index=False)


# Evigt files
print('processing evigt files...')

# Directory path containing the interview files
evigt_files = glob.glob('/work/Ccp-MePSDA/output/transcript/evigt_transcribed/*.csv')

# Create empty list
dataframes = []

# Iterate over each file and read it, skipping the first row (header)
for filename in evigt_files:
    df = pd.read_csv(filename)
    df_concat = pd.DataFrame({
        'full_text': [df['text'].str.cat(sep='. ')],
        'title': [os.path.basename(filename)], # creaing title based on filename
        'source': ['evigt'] # Type of interview
    })
    
    dataframes.append(df_concat)

# Concatenate all dataframes
evigt_df = pd.concat(dataframes, ignore_index=True)

# Converting to string
evigt_df['full_text'] = evigt_df['full_text'].astype('str')

# Apply the function to the text column and explode the result
evigt_df['chunked'] = evigt_df['full_text'].progress_apply(lambda x: split_text_into_chunks(x, min_chars))
evigt_df = evigt_df.explode("chunked").reset_index(drop=True)

# create chunk index
evigt_df['chunk_index'] = evigt_df.groupby('title').cumcount()

# To CSV
evigt_df.to_csv('/work/Ccp-MePSDA/output/collected_data/df_evigt.csv', index=False)


# Zusa files 
print('processing zusa...')

zusa_files = glob.glob('/work/Ccp-MePSDA/output/transcript/zusa_transcribed/*.csv')
# Empty list for files
dataframes = []
# For looping files
for filename in zusa_files:
    df = pd.read_csv(filename)
    df_concat = pd.DataFrame({
        'full_text': [df['text'].str.cat(sep='. ')],
        'title': [os.path.basename(filename)], # creaing title based on filename
        'source': ['zusa'] # Type of interview
    })

    dataframes.append(df_concat)

# Concatenate
zusa_df = pd.concat(dataframes, ignore_index=True)


# Apply the function to the text column and explode the result
zusa_df['chunked'] = zusa_df['full_text'].progress_apply(lambda x: split_text_into_chunks(x, min_chars))
zusa_df = zusa_df.explode("chunked").reset_index(drop=True)
# chunk index
zusa_df['chunk_index'] = zusa_df.groupby('title').cumcount()

# To CSV
zusa_df.to_csv('/work/Ccp-MePSDA/output/collected_data/df_zusa.csv', index=False)


# Grænser files
print('processing grænser...')

grænser_files = glob.glob('/work/Ccp-MePSDA/output/transcript/grænser_transcribed/*.csv')

# Empty list for files
dataframes = []
# for loop collecting the csv files in path
for filename in grænser_files:
    df = pd.read_csv(filename)
    df_concat = pd.DataFrame({
        'full_text': [df['text'].str.cat(sep='. ')],
        'title': [os.path.basename(filename)], # creaing title based on filename
        'source': ['grænser'] # Type of interview
    })
    
    dataframes.append(df_concat)
# Concatenating interview files
grænser_df = pd.concat(dataframes, ignore_index=True)

# Apply the segmenter function to the text column and explode the result
grænser_df['chunked'] = grænser_df['full_text'].progress_apply(lambda x: split_text_into_chunks(x, min_chars))
grænser_df = grænser_df.explode("chunked").reset_index(drop=True)

grænser_df['chunk_index'] = grænser_df.groupby('title').cumcount()

# Saving to csv
grænser_df.to_csv('/work/Ccp-MePSDA/output/collected_data/df_grænser.csv', index=False)


# Grænser III files
print('processing Grænser III: Return of the Jedi...')

grænser_III_files = glob.glob('/work/Ccp-MePSDA/output/transcript/grænser_III_transcribed/*.csv')

# Empty list for files
dataframes = []
# For loop collecting the csv files in path
for filename in grænser_III_files:
    df = pd.read_csv(filename)
    df_concat = pd.DataFrame({
        'full_text': [df['text'].str.cat(sep='. ')],
        'title': [os.path.basename(filename)], # creaing title based on filename
        'source': ['grænser_III'] # Type of interview
    })
    
    dataframes.append(df_concat)
# Concatenating interview files
grænser_III_df = pd.concat(dataframes, ignore_index=True)

# Apply the segmenter function to the text column and explode the result
grænser_III_df['chunked'] = grænser_III_df['full_text'].progress_apply(lambda x: split_text_into_chunks(x, min_chars))
grænser_III_df = grænser_III_df.explode("chunked").reset_index(drop=True)

grænser_III_df['chunk_index'] = grænser_III_df.groupby('title').cumcount()

# Saving to csv
grænser_III_df.to_csv('/work/Ccp-MePSDA/output/collected_data/df_grænser_III.csv', index=False)


# ________________________ Merging files ________________________
print('merging...')

# Selecting all frames created
frames = [kontra_df, evigt_df, zusa_df, grænser_df, grænser_III_df]

# Concatenating the frames into one
bts_interviews = pd.concat(frames, ignore_index=True)

# Making content as strings
bts_interviews['chunked'] = bts_interviews['chunked'].astype(str)

# Replace NaN with empty string or some placeholder
bts_interviews['chunked'] = bts_interviews['chunked'].replace(np.nan, '')

# Saving to output folder
print('saving to csv...')
bts_interviews.to_csv('/work/Ccp-MePSDA/output/collected_data/bts_df.csv', index=False)

print('Done!')