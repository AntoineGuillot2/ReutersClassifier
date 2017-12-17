# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 09:52:50 2017

@author: Antoi
"""
import numpy as np
def cosine_similarity(x,y):
    return(x.dot(np.transpose(y))/np.sqrt(np.sum(x**2)*np.sum(y**2)))


def article_sim(article_1,article_2,embedding_model):
    embedding_1=embedding_model.predict(article_1.reshape((1,-1)))
    embedding_2=embedding_model.predict(article_2.reshape((1,-1)))
    print(np.shape(embedding_1))
    print(np.shape(embedding_2))
    return cosine_similarity(embedding_1,embedding_2)
    
def closest_articles(article_1,article_set,embedding_model,k=5):
    embedding_1=embedding_model.predict(article_1.reshape((1,-1)))
    embedding_set=embedding_model.predict(article_set)
    similarities=list(map(lambda x:cosine_similarity(embedding_1,x),embedding_set))
    similarities=np.reshape(similarities,(-1,))
    indices=np.argsort(similarities)
    return indices[-k:], np.sort(similarities)[-k:]