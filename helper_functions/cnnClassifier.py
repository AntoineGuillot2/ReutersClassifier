# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 10:55:15 2017

@author: Antoi
"""
##In this script we want to classify the reuters dataset
##The training set contains around 7000 news and the test set around 3000 obs
##Each new can belong to one or several of the 90 categories

from keras.models import Model, Sequential, load_model
from keras.layers import LSTM, GRU,Dense, Reshape, Masking, Input
from keras.layers.embeddings import Embedding
from keras import metrics
from buildData import build_output,build_input
from articleSimilarity import closest_articles, article_sim
import numpy as np


##Boolean to chose whether to load or build the input dataset (takes some time)
create_input=False
save_rnn=False
##Loading data

if create_input==False:
    create_input=np.load('vectorized_883_400.npy')
    input_data_test=np.load('vectorized_883_400_test.npy')
else:
    input_data=build_input('train',400)
    input_data_test=build_input('test',400)

output_data=build_output('train')
output_data_test=build_output('test')


##  CNN classifier ######
from keras.layers import Conv1D,MaxPooling1D, Flatten, Dropout

##Building CNN
model = Sequential()
model.add(Embedding(int(np.max(input_data)), 200, input_length=input_data.shape[1]))
model.add(Reshape((200,input_data.shape[1])))
model.add(Conv1D(50,10))
model.add(MaxPooling1D(5))
model.add(Dropout(0.3))
model.add(Conv1D(50,10))
model.add(MaxPooling1D(5))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(100,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(100,activation='relu'))
model.add(Dropout(0.3))
model.add((Dense(output_data.shape[1],activation='sigmoid')))

##Compilation and training
model.compile(loss='binary_crossentropy',
              optimizer='nadam')
model.fit(input_data,output_data, batch_size=100, epochs=12,validation_data=[input_data_test,output_data_test])

##Prediction on test set
pred=model.predict(input_data_test)
pred_vs_real=np.abs(output_data_test-pred)
