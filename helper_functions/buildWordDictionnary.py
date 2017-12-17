# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 18:04:09 2017

@author: Antoi
"""
import pandas as pd
##Build the dictionnary of a list of token (each token is a text from the corpus)
##Input:
##- tokens_list: list of list of words.Each list correspond to a text
##- min_freq/max_freq: words which appear less/more than the freq in the differents documents are removed.
##Output:
##- two pandas dataframe, one containing the words and their index and one containing the frequency of the words.
def build_dict(tokens_list,min_freq=None,max_freq=None):
    word_dict={}
    word_frequency={}
    for tokens in tokens_list:
        word_in_tokens={}
        for token in tokens:           
            if token in word_dict.keys():
                if token not in word_in_tokens.keys():
                    word_frequency[token]=word_frequency[token]+1
            else:
                word_dict[token]=len(word_dict)-1
                word_frequency[token]=1
            word_in_tokens[token]=1
    df_word=pd.DataFrame.from_dict(word_dict,'index')
    df_word['word']=list(df_word.index)
    df_freq=pd.DataFrame.from_dict(word_frequency,'index')
    df_freq['word']=list(df_freq.index)
    df_freq['frequency']=df_freq.iloc[:,0]/len(tokens_list)*100
    if min_freq!=None:
        df_word=df_word[df_freq['frequency']>min_freq]
        df_freq=df_freq[df_freq['frequency']>min_freq]
    if max_freq!=None:
        df_word=df_word[df_freq['frequency']<max_freq]
        df_freq=df_freq[df_freq['frequency']<max_freq]
    df_word['word_index']=range(len(df_word))
    df_word=df_word.drop(0,1)
    return df_word, df_freq