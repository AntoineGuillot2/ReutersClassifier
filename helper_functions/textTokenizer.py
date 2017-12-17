# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 17:25:01 2017

@author: Antoi
"""
import string
import re    
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from dateutil.parser import parse


###Removing stop word and punctuation

##Remove a list of words from a tokenized text. Mainly used to remove punctuation and stop words
def remove_words(tokenised_text):
    stop_words=stopwords.words('english')
    punctuation=string.punctuation
    char_to_remove=stop_words+list(punctuation)+['lt']+["'s"]+["''"]
    return [word.lower() for word in tokenised_text if word.lower() not in char_to_remove]

###Replace date and number by a placeholder
def replace_date_number(string):
    try: 
        parse(string)
        return 'datetime_number'
    except ValueError:
        if any(char.isdigit() for char in string):
            return 'datetime_number'
        else:
            return string
  
##Remove +',; from the string
def remove_special_char(string):
    return re.sub("[^A-Za-z0-9]+", '', string)

##stem a list of tokens
def stem_list(tokens):
    stemmer = SnowballStemmer("english")
    return list(map(stemmer.stem,tokens))

##return the stemmed and cleaned news set
def stem_news_set(news_set):
    ##Defining stop words and punctuation
    ##Tokenisation
    tokenised_train_set= list(map(lambda x: x.replace('/',' '),news_set))
    tokenised_train_set = list(map(word_tokenize, tokenised_train_set))
    ##Remove stop words and punctuation
    cleaned_train_set = list(map(remove_words,tokenised_train_set))
    ##Remove dates, number and some special caracters
    cleaned_train_set = list(map(lambda x: list(map(replace_date_number,x)),cleaned_train_set))
    cleaned_train_set = list(map(lambda x: list(map(remove_special_char,x)),cleaned_train_set))
    ###stemming
    return list(map(stem_list,cleaned_train_set ))
