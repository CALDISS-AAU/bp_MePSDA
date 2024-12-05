import pandas as pd
import re
import os
import sys
from os.path import join
import glob

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
txt_dir = join(data_dir, 'txt')

data_transcribed = join(output_dir, 'transcribed')

# FUNCTION FOR CLEANING FILENAME (WINDOWS APPROPRIATE)
def sanitize_filename(filename):
    # Define invalid characters
    invalid_chars = r'[<>:"/\\|?*]'
    # Replace invalid characters with an underscore
    sanitized = re.sub(invalid_chars, '_', filename)
    # Ensure no trailing spaces or periods
    return sanitized.strip().rstrip('.')

# Directory path containing the interview files
masterclass_files = glob.glob('/work/Ccp-MePSDA/output/transcript/masterclass_transcribed/*.csv') # masterclass

misc_files = glob.glob('/work/Ccp-MePSDA/output/transcript/miscellaneous_transcribed/*.csv') # misc

interview_files = glob.glob('/work/Ccp-MePSDA/output/transcript/interviews_transcribed/*.csv') # interviews

## MASTERCLASS TO TXT
outdir = join(txt_dir, 'masterclass')
for filename in masterclass_files:
    df = pd.read_csv(filename)

    text = df['text'].str.cat(sep='.\n')
    name = sanitize_filename(os.path.basename(filename)).replace('.csv', '')
    
    outname = f'{os.path.basename(name)}.txt'

    outp = join(outdir, outname)

    with open(outp, 'w', encoding='utf-8') as f:
        f.write(text)

## MISC TO TXT
outdir = join(txt_dir, 'misc')
for filename in misc_files:
    df = pd.read_csv(filename)

    text = df['text'].str.cat(sep='.\n')
    name = sanitize_filename(os.path.basename(filename)).replace('.csv', '')
    
    outname = f'{os.path.basename(name)}.txt'

    outp = join(outdir, outname)

    with open(outp, 'w', encoding='utf-8') as f:
        f.write(text)


## INTERVIEWS TO TXT
outdir = join(txt_dir, 'interviews')
for filename in interview_files:
    df = pd.read_csv(filename)

    text = df['text'].str.cat(sep='.\n')
    name = sanitize_filename(os.path.basename(filename)).replace('.csv', '')
    
    outname = f'{os.path.basename(name)}.txt'
    
    outp = join(outdir, outname)

    with open(outp, 'w', encoding='utf-8') as f:
        f.write(text)
