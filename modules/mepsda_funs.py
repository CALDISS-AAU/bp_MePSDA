#!/usr/bin/env python
# coding: utf-8

import re

def correct_title(text):
    '''
    removing filetype from Title column
    '''
    pattern = r'.txt'
    return re.sub(pattern, '', text)

def remove_line(text):
    '''
    removing new line
    '''
    pattern = r'\n'
    return re.sub(pattern, '', text)

def info(text):
    '''
    infomedia type needs removing
    '''
    pattern=r'Originalartiklen kan ikke vises i dette vindue\n'
    return re.sub(pattern, ' ', text)

def info_filter(text):
    pattern=r'Åbn originalartiklen i et nyt vindue\n'
    return re.sub(pattern, '', text)

def org_article(text):
    pattern=r'Åbn originalartikel\n'
    return re.sub(pattern, '', text)

def thumbnail(text):
    pattern=r'thumbnail.*\n'
    return re.sub(pattern, '', text)

def article_no_show(text):
    pattern=r'Originalvisning er ikke tilgængelig for denne artikel\n'
    return re.sub(pattern, '', text)

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
