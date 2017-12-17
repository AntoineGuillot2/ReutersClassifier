# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 19:21:21 2017

@author: Antoi
"""
##In this script we want to classify the reuters dataset
##The training set contains around 7000 news and the test set around 3000 obs
##Each new can belong to one or several of the 90 categories

from keras.models import Model, Sequential, load_model
from keras.layers import LSTM, GRU,Dense, Reshape, Masking, Input
from keras.layers.embeddings import Embedding
from keras import metrics
from helper_functions.buildData import build_output,build_input
from helper_functions.articleSimilarity import closest_articles, article_sim
import numpy as np

##Boolean to chose whether to load or build the input dataset (takes some time)
create_input=False
save_rnn=False
##Loading data

if create_input==False:
    input_data=np.load('data/vectorized_883_400.npy')
    input_data_test=np.load('data/vectorized_883_400_test.npy')
else:
    input_data=build_input('train',400)
    input_data_test=build_input('test',400)

output_data=build_output('train')
output_data_test=build_output('test')

########### TRAINING ###########
if save_rnn==True:
    ######  RNN Classifier #####
    
    ##Network achitecture
    word_input=Input(shape=(input_data.shape[1],))
    x=Embedding(input_dim=int(np.max(input_data)), output_dim=100, input_length=input_data.shape[1], mask_zero=True)(word_input)
    article_embedding=GRU(100)(x)
    x=Dense(100,activation='relu')(article_embedding)
    x=Dense(100,activation='relu')(x)
    category_prediction=Dense(output_data.shape[1],activation='sigmoid')(x)
    
    ##Compiling and training model
    global_model=Model(word_input,category_prediction)
    global_model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',metrics=[metrics.top_k_categorical_accuracy,'accuracy'])
    hist=global_model.fit(input_data,output_data, batch_size=64, epochs=14,validation_data=[input_data_test,output_data_test])

##Saving or loading RNN
if save_rnn==True:
    global_model.save('models/global_model.h5')
    article_embedding_model=Model(word_input,article_embedding)
    article_embedding_model.save('models/article_embedding_model.h5')
else:
    global_model=load_model('models/global_model.h5')
    article_embedding_model=load_model('models/article_embedding_model.h5')
    

##################################################################
########### Model evaluation ####################################

from sklearn.metrics import roc_auc_score, precision_recall_curve,average_precision_score
import matplotlib.pyplot as plt

##Evaluation of the model on the train set with ROC AUC
prediction_train=global_model.predict(input_data)
ROC_AUC=list(map(lambda x: roc_auc_score(output_data[:,x],prediction_train[:,x]),range(90)))
plt.hist(ROC_AUC, normed=True, bins=30)
plt.ylabel('Probability')
plt.xlabel('ROC AUC')
print('MEAN TRAIN AUC ', np.mean(ROC_AUC))
##Weighted PR
PR_Score=list(map(lambda x: average_precision_score(output_data[:,x],prediction_train[:,x]),range(90)))
weighted_PR_increase=PR_Score/np.mean(output_data_test,0)
np.median(weighted_PR_increase)
plt.hist(np.log(weighted_PR_increase)/np.log(10), normed=True, bins=30)
plt.ylabel('Probability')



##Evaluation of the model on the testset with ROC AUC
prediction_test=global_model.predict(input_data_test)
ROC_AUC=list(map(lambda x: roc_auc_score(output_data_test[:,x],prediction_test[:,x]),range(90)))
plt.hist(ROC_AUC, normed=True, bins=30)
plt.ylabel('Probability')
plt.xlabel('ROC AUC')
print('MEAN TEST AUC ', np.mean(ROC_AUC))
##Weighted PR on test set
PR_Score=list(map(lambda x: average_precision_score(output_data_test[:,x],prediction_test[:,x]),range(90)))
weighted_PR_increase=PR_Score/np.mean(output_data_test,0)
np.median(weighted_PR_increase)
import matplotlib.pyplot as plt
plt.hist(np.log(weighted_PR_increase)/np.log(10), normed=True, bins=30)
plt.ylabel('Probability')



#######################################################
##Visualisation of the output layer ###########
from seaborn import heatmap, clustermap
import pandas as pd
from nltk.corpus import reuters
##Visualisation of the test output layer
prediction_test_df=pd.DataFrame(prediction_test)
prediction_test_df.columns=reuters.categories()
heatmap(prediction_test_df, cmap="YlGnBu")

error_test_df=pd.DataFrame(prediction_test-output_data_test)
error_test_df.columns=reuters.categories()
clustermap(error_test_df, cmap="RdBu")

##Visualisation of the train output layer
prediction_train_df=pd.DataFrame(prediction_train)
prediction_train_df.columns=reuters.categories()
heatmap(prediction_train_df, cmap="YlGnBu")
error_train_df=pd.DataFrame(prediction_train-output_data)
error_train_df.columns=reuters.categories()
clustermap(error_train_df, cmap="RdBu")



##################################################################
###K closest document to a given document######################
documents = reuters.fileids()
docs_id = list(filter(lambda doc: doc.startswith('test'),
                            documents))
docs = [reuters.raw(doc_id) for doc_id in docs_id]
closest_id_1,closest_sim_1=closest_articles(input_data_test[112],input_data_test,article_embedding_model)
print(docs[112])
for x in closest_id_1[1:4]:
    print(docs[x])

