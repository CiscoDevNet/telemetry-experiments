"""
Copyright (c) 2024 Cisco and/or its affiliates.
This software is licensed to you under the terms of the Cisco Sample
Code License, Version 1.1 (the "License"). You may obtain a copy of the
License at
               https://developer.cisco.com/docs/licenses
All use of the material herein must be in accordance with the terms of
the License. All rights not expressly granted by the License are
reserved. Unless required by applicable law or agreed to separately in
writing, software distributed under the License is distributed on an "AS
IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
or implied.
"""

import nltk

from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
nltk.download('punkt')

def get_stop_words(stop_file_path):
    with open(stop_file_path, 'r', encoding="utf-8") as f:
        stopwords = f.readlines()
        stop_set = set(m.strip() for m in stopwords)
        return list(stop_set)
    
stemmer = PorterStemmer()
def get_stemmed_tokens(tokens, stemmer):
    stemmed_tokens = []
    for token in tokens:
        if token.isalpha():
            stemmed_tokens.append(stemmer.stem(token))
    return stemmed_tokens

def get_tokens(string):
    tokens = word_tokenize(string)
    stemmed_tokens = get_stemmed_tokens(tokens, stemmer)
    return stemmed_tokens