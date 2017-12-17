# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 18:38:58 2017

@author: Antoi
"""
import numpy as np


##Function to transform the different texts into their corresponding sequence of integer
##Input: 
#tokens_list: List of tokens, each token is the list of the words in a text
#word_dict: the dictionnary to use, the dictionnary is a pandas dataframe mapping a word to a number
#max_sequence_length: maximum length of a token. If a token is longer, it is cut, if it is shorter, 0 are added
##Ex:If "snow" is the word n°10 of the dictionnary and "cat" the number 15,  ["cat","snow"] 
## will return [15,10] 
def vectorize_tokens(tokens_list,word_dict,max_sequence_length=None):
    if max_sequence_length==None:
        sequence_length=max(list(map(len,tokens_list)))
    else:
        sequence_length=max_sequence_length
    n_text=len(tokens_list)
    result=np.zeros((n_text,sequence_length))
    text_number=0
    for tokens in tokens_list:
        if text_number%100==0:
            print(text_number)
        word_position=0
        for token in tokens:
            try:
                word_index=word_dict[word_dict.word==token]['word_index'][0]
                result[text_number,word_position]=word_index+1
                word_position+=1
            except IndexError:
                pass
            if word_position>=sequence_length:
                break
        text_number+=1
    return result