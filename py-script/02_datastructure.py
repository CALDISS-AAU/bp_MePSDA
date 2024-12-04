import pandas as pd
import re
import os
import sys
from os.path import join
import glob
import numpy as np
import pysbd
from pysbd import Segmenter

# DIRS AND PATHS
project_dir = join('/work', 'Ccp-MePSDA')
data_dir = join(project_dir, 'data')
modelling_dir = join(project_dir, 'modelling')
modules_dir = join(project_dir, 'modules')
sys.path.append(modules_dir)

from mepsda_funs import * # indlæser alle funktioer i mepsda_funs

logs_dir = join(modelling_dir, 'logs')
output_dir = join(project_dir, 'output')
model_dir = join(modelling_dir, 'models')

data_transcribed = join(output_dir, 'transcribed')

# Set the minimum character limit for segmenter/chunking function
min_chars = 200

# Directory path containing the interview files
path = r'/work/Ccp-MePSDA/data/infomedia'

# Create empty list
data = []

# Iterate over each file in the directory
for filename in os.listdir(path):
    # Ensure it's a .txt file
    if filename.endswith(".txt"):
        with open(os.path.join(path, filename), encoding='utf-8') as fh:
            # Read the first line as the title
            title = filename
            # Read the rest of the lines as content
            content = fh.read().strip() 
        # Append the data to the list as a dictionary
        data.append({"Title": title, "Content": content})

# Convert to pandas dataframe
infomedia_df = pd.DataFrame(data)

# preprocess stuff
infomedia_df = infomedia_df.sort_values(by='Title', ascending=True, ignore_index=True)
infomedia_df.rename(columns={'Title':'title', 'Content': 'text'}, inplace=True)
# applying title function
infomedia_df['title'] = infomedia_df['title'].apply(correct_title)
# applying filter functions
infomedia_df['text'] = infomedia_df['text'].apply(info)
infomedia_df['text'] = infomedia_df['text'].apply(info_filter)
infomedia_df['text'] = infomedia_df['text'].apply(org_article)
infomedia_df['text'] = infomedia_df['text'].apply(thumbnail)
infomedia_df['text'] = infomedia_df['text'].apply(article_no_show)
# converting to intergers
infomedia_df['title'] = infomedia_df['title'].astype("int64")

# Applying new line function
infomedia_df['text'] = infomedia_df['text'].apply(remove_line)
# converting to string
infomedia_df['text'] = infomedia_df['text'].astype('str')
infomedia_df['source'] = 'infomedia'

# Segmentation function
def split_text_into_chunks(text, min_chars):
    from pysbd import Segmenter
    segmenter = Segmenter(language="da", clean=True)
    sentences = segmenter.segment(text)
    chunks = []
    current_chunk = []
    current_length = 0
    for sentence in sentences:
        sentence_length = len(sentence)
        if current_length + sentence_length > min_chars and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length
        else:
            current_chunk.append(sentence)
            current_length += sentence_length
    if current_chunk:
        last_chunk = " ".join(current_chunk)
        if len(last_chunk) < min_chars and chunks:
            chunks[-1] += " " + last_chunk
        else:
            chunks.append(last_chunk)
    return chunks


# Apply the function to the text column and explode the result
infomedia_df['chunked'] = infomedia_df['text'].apply(lambda x: split_text_into_chunks(x, min_chars))
infomedia_df = infomedia_df.explode("chunked").reset_index(drop=True)

# create chunk index
infomedia_df['chunk_index'] = infomedia_df.groupby('title').cumcount()
# making sure the chunked column is strings
infomedia_df['chunked'] = infomedia_df['chunked'].astype(str)

# Saving as csv
infomedia_df.to_csv("/work/Ccp-MePSDA/output/infomedia/df_infomedia.csv", index=False)


# ________________________ transcribed files ________________________

# Masterclass files

# Directory path containing the interview files
masterclass_files = glob.glob('/work/Ccp-MePSDA/output/transcript/masterclass_transcribed/*.csv')

# Create empty list
dataframes = []

# Iterate over each file and read it, skipping the first row (header)
for filename in masterclass_files:
    df = pd.read_csv(filename)
    df['title'] =  filename.split('/')[-1] # creaing title based on filename
    df['source'] = 'masterclass' # Type of interview
    dataframes.append(df)

# Concatenate all dataframes
masterclass_df = pd.concat(dataframes, ignore_index=True)

# Converting to string
masterclass_df['text'] = masterclass_df['text'].astype('str')

# Apply the function to the text column and explode the result
masterclass_df['chunked'] = masterclass_df['text'].apply(lambda x: split_text_into_chunks(x, min_chars))
masterclass_df = masterclass_df.explode("chunked").reset_index(drop=True)

# create chunk index
masterclass_df['chunk_index'] = masterclass_df.groupby('title').cumcount()

# dropping 
masterclass_df.drop(columns=['temperature', 'avg_logprob', 'words'], inplace=True)
# To CSV
masterclass_df.to_csv('/work/Ccp-MePSDA/output/collected_data/df_masterclass.csv', index=False)


# misc files 

misc_files = glob.glob('/work/Ccp-MePSDA/output/transcript/miscellaneous_transcribed/*.csv')
# empty list for files
dataframes = []
# For looping files
for filename in misc_files:
    df = pd.read_csv(filename)
    df['title'] =  filename.split('/')[-1] # Creating file name
    df['source'] = 'misc'
    dataframes.append(df)
# concatenate
misc_df = pd.concat(dataframes, ignore_index=True)


# Apply the function to the text column and explode the result
misc_df['chunked'] = misc_df['text'].apply(lambda x: split_text_into_chunks(x, min_chars))
misc_df = misc_df.explode("chunked").reset_index(drop=True)
# chunk index
misc_df['chunk_index'] = misc_df.groupby('title').cumcount()
# dropping columns
misc_df.drop(columns=['temperature', 'avg_logprob', 'words', 'id'], inplace=True)

# To CSV
misc_df.to_csv('/work/Ccp-MePSDA/output/collected_data/df_misc.csv', index=False)


# interview files
interview_files = glob.glob('/work/Ccp-MePSDA/output/transcript/interviews_transcribed/*.csv')

# Empty list for files
dataframe = []
# for loop collecting the csv files in path
for filename in interview_files:
    df = pd.read_csv(filename)
    df['title'] =  filename.split('/')[-1] # Creating file name
    df['source'] = 'interviews' 
    dataframe.append(df)
# Concatenating interview files
interview_df = pd.concat(dataframe, ignore_index=True)

# Apply the segmenter function to the text column and explode the result
interview_df['chunked'] = interview_df['text'].apply(lambda x: split_text_into_chunks(x, min_chars))
interview_df = interview_df.explode("chunked").reset_index(drop=True)

interview_df['chunk_index'] = interview_df.groupby('title').cumcount()

# Dropping columns
interview_df.drop(columns=['temperature', 'avg_logprob', 'words', 'id'], inplace=True)

# Saving to csv
interview_df.to_csv('/work/Ccp-MePSDA/output/collected_data/df_interview.csv', index=False)


# ________________________ Merging files ________________________
# selecting all frames created
frames = [masterclass_df, interview_df, misc_df, infomedia_df]

# concatenating the frames into one
master_df = pd.concat(frames, ignore_index=True)

# aggregating each text under the same title
#master_df = master_df.groupby(['title', 'source']).agg({'chunked':'\n'.join})

# making content as strings
master_df['chunked'] = master_df['chunked'].astype(str)

# Replace NaN with empty string or some placeholder
master_df['chunked'] = master_df['chunked'].replace(np.nan, '')

# Resetting index to flatten the grouped structure
#master_df = master_df.reset_index()

# Saving to output folder
master_df.to_csv('/work/Ccp-MePSDA/output/collected_data/mepsda_df.csv', index=False)