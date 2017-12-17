# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 20:05:06 2017

@author: Antoi
"""
 

from helper_functions.textTokenizer import stem_news_set
from helper_functions.buildWordDictionnary import build_dict
from helper_functions.vectorizeTokens import vectorize_tokens
import numpy as np
from nltk.corpus import reuters
##Build the input for the selected dataset (train or test) of Reuters news
##Input:
##-dataset: input to build (train or test)
##- max_sequence_length: number of words to use
##- Minimum frequency of a word, word with lower frequency will be discarded
##Return: a numpy array of dimension number of text by sequence length
def build_input(dataset='train',max_sequence_length=200,min_voc_frequency=0.5):
    documents = reuters.fileids()
    docs_id = list(filter(lambda doc: doc.startswith(dataset),
                            documents))
    docs = [reuters.raw(doc_id) for doc_id in docs_id]
    stemmed_set=stem_news_set(docs)
    word_dict,word_freq=build_dict(stemmed_set,min_freq=min_voc_frequency)
    if dataset=='test':
            docs_id_train = list(filter(lambda doc: doc.startswith('train'),
                            documents))
            docs_train = [reuters.raw(doc_id) for doc_id in docs_id_train]
            stemmed_train_set=stem_news_set(docs_train)
            word_dict,word_freq=build_dict(stemmed_train_set,min_freq=min_voc_frequency)
    vectorized_text=vectorize_tokens(stemmed_set,word_dict,max_sequence_length)
    return(vectorized_text)

##Build the output/target for the selected dataset (train or test) of Reuters news
##Input:
##-dataset: input to build (train or test)
##Return:
##- a one-hot encoded numpy array of dimension number of text by numbr of categories
def build_output(dataset='train'):
    documents = reuters.fileids()
    docs_id = list(filter(lambda doc: doc.startswith(dataset),
                            documents))
    output=np.zeros((len(docs_id),len(reuters.categories())))
    reuters_categories=reuters.categories()
    i=0
    for docs in docs_id:
        if i%100==0:
            print(i)
        for category in reuters.categories(docs):
            output[i,reuters_categories.index(category)]=1
        i+=1
    return output

