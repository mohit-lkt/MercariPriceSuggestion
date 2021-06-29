#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 07:42:39 2020

@author: mohit
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
#Load the data
merdata = pd.read_csv('train.tsv', sep = '\t')
test = pd.read_csv('test.tsv', sep = '\t')

#Data Preprocessing

#filling missing values
def fill_missing_data(data):
        data.category_name.fillna(value = "Others",inplace = True)
        data.brand_name.fillna(value = "Not known",inplace = True)
        data.item_description.fillna(value = "No description given",inplace = True)
        return data
merdata = fill_missing_data(merdata)
test = fill_missing_data(test)

#Splitting category name into 3 sub categories
def cat_split(row):
    try:
        text = row
        text1,text2,text3 = text.split('/')
        return text1,text2,text3
    except:
        return ("Label not given","Label not given","Label not given")
merdata['general_cat'],merdata['subcat_1'],merdata['subcat_2'] = zip(*merdata['category_name'].apply(lambda x : cat_split(x)))
test['general_cat'],test['subcat_1'],test['subcat_2'] = zip(*test['category_name'].apply(lambda x : cat_split(x)))

#finding item description length
def wordCount(text):
        text = text.lower()
        regex = re.compile('[' +re.escape(string.punctuation) + '0-9\\r\\t\\n]')
        txt = regex.sub(" ", text)
        words = [w for w in txt.split(" ")
                 if not w in stopwords.words('english') and len(w)>3]
        return len(words)
merdata['desc_len'] = merdata['item_description'].apply(lambda x: wordCount(x))
test['desc_len'] = test['item_description'].apply(lambda x: wordCount(x))


#Encoding on name column
from sklearn.feature_extraction.text import CountVectorizer
vectorizer1 = CountVectorizer(min_df = 10)
vectorizer1.fit(merdata['name'].values)
merdata_name = vectorizer1.transform(merdata['name'].values)
test_name = vectorizer1.transform(test['name'].values)

#normalizing item description length
from sklearn.preprocessing import Normalizer
normalizer = Normalizer()
normalizer.fit(merdata['desc_len'].values.reshape(-1,1))
x_train_words_des_norm = normalizer.transform(merdata['desc_len'].values.reshape(-1,1))
x_test_words_des_norm = normalizer.transform(test['desc_len'].values.reshape(-1,1))

#‘get_dummies’ method will recognize the categorical features and will obtain the one-hot encoded values automatically. 
from scipy.sparse import csr_matrix
x_dummies = csr_matrix(pd.get_dummies(merdata[['item_condition_id','shipping']],sparse=True).values)
x_test_dummies = csr_matrix(pd.get_dummies(test[['item_condition_id','shipping']],sparse=True).values)

#Brand name Encoding
vectorizer2 = CountVectorizer(min_df = 10)
vectorizer2.fit(merdata['brand_name'].values)
merdata_brandname = vectorizer2.transform(merdata['brand_name'].values)
test_brandname = vectorizer2.transform(test['brand_name'].values)

#General category Encoding
vectorizer3 = CountVectorizer(min_df = 10)
vectorizer3.fit(merdata['general_cat'].values)
merdata_gencat = vectorizer3.transform(merdata['general_cat'].values)
test_gencat = vectorizer3.transform(test['general_cat'].values)

#Sub category1 Encoding
vectorizer4 = CountVectorizer(min_df = 10)
vectorizer4.fit(merdata['subcat_1'].values)
merdata_subcat1 = vectorizer4.transform(merdata['subcat_1'].values)
test_subcat1 = vectorizer4.transform(test['subcat_1'].values)

#Sub category2 Encoding
vectorizer5 = CountVectorizer(min_df = 10)
vectorizer5.fit(merdata['subcat_2'].values)
merdata_subcat2 = vectorizer5.transform(merdata['subcat_2'].values)
test_subcat2 = vectorizer5.transform(test['subcat_2'].values)

import re

def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

# we are removing the words from the stop words list for ex: 'no', 'nor', 'not'
stopwords= ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",
            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 
            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 
            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 
            's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 
            've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',
            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',
            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 
            'won', "won't", 'wouldn', "wouldn't"]

from tqdm import tqdm
preprocessed_train_des = []
# tqdm is for printing the status bar
for sentance in tqdm(merdata['item_description'].values):
    sent = decontracted(sentance)
    sent = sent.replace('\\r', ' ')
    sent = sent.replace('\\"', ' ')
    sent = sent.replace('\\n', ' ')
    sent = re.sub('[^A-Za-z0-9]+', ' ', sent)
    # https://gist.github.com/sebleier/554280
    sent = ' '.join(e for e in sent.split() if e.lower() not in stopwords)
    preprocessed_train_des.append(sent.lower().strip())

# after preprocesing
preprocessed_train_des[20000]


preprocessed_test_des = []
# tqdm is for printing the status bar
for sentance in tqdm(test['item_description'].values):
    sent = decontracted(sentance)
    sent = sent.replace('\\r', ' ')
    sent = sent.replace('\\"', ' ')
    sent = sent.replace('\\n', ' ')
    sent = re.sub('[^A-Za-z0-9]+', ' ', sent)
    # https://gist.github.com/sebleier/554280
    sent = ' '.join(e for e in sent.split() if e.lower() not in stopwords)
    preprocessed_test_des.append(sent.lower().strip())

#Calculating sentiment score on item description as a feature
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()
train_sentiment = []; 
for sentence in tqdm(preprocessed_train_des):
    for_sentiment = sentence
    ss = sid.polarity_scores(for_sentiment)
    train_sentiment.append(ss)
negative=[]
neutral=[]
positive=[]
compounding=[]
for i in train_sentiment:
    for polarity,score in i.items():
        if(polarity=='neg'):
            negative.append(score)
        if(polarity=='neu'):
            neutral.append(score)
        if(polarity=='pos'):
            positive.append(score)
        if(polarity=='compound'):
            compounding.append(score)
#creating new features as each of the 4 sentiments.
merdata['negative']=negative
merdata['neutral']=neutral
merdata['positive']=positive
merdata['compound']=compounding

#Same as above sentiment analysis to be performed on test data.
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

sid = SentimentIntensityAnalyzer()

X_test_sentiment = []; 
for sentence in tqdm(preprocessed_test_des):
    for_sentiment = sentence
    ss = sid.polarity_scores(for_sentiment)
    X_test_sentiment.append(ss)
negative=[]
neutral=[]
positive=[]
compounding=[]
for i in X_test_sentiment:
    
    for polarity,score in i.items():
        if(polarity=='neg'):
            negative.append(score)
        if(polarity=='neu'):
            neutral.append(score)
        if(polarity=='pos'):
            positive.append(score)
        if(polarity=='compound'):
            compounding.append(score)
test['negative']=negative
test['neutral']=neutral
test['positive']=positive
test['compound']=compounding

#Normalizing values for every sentiment feature

#For negative
from sklearn.preprocessing import Normalizer
normalizer = Normalizer()

normalizer.fit(merdata['negative'].values.reshape(-1,1))

X_train_neg_norm = normalizer.transform(merdata['negative'].values.reshape(-1,1))

X_test_neg_norm = normalizer.transform(test['negative'].values.reshape(-1,1))

print("After normalizations")
print(X_train_neg_norm.shape, y_train.shape)

#For Neutral
from sklearn.preprocessing import Normalizer
normalizer = Normalizer()

normalizer.fit(merdata['neutral'].values.reshape(-1,1))

X_train_neu_norm = normalizer.transform(merdata['neutral'].values.reshape(-1,1))

X_test_neu_norm = normalizer.transform(test['neutral'].values.reshape(-1,1))

print("After normalizations")
print(X_train_neu_norm.shape, y_train.shape)

#For Positive
from sklearn.preprocessing import Normalizer
normalizer = Normalizer()

normalizer.fit(merdata['positive'].values.reshape(-1,1))

X_train_pos_norm = normalizer.transform(merdata['positive'].values.reshape(-1,1))

X_test_pos_norm = normalizer.transform(test['positive'].values.reshape(-1,1))

print("After normalizations")
print(X_train_pos_norm.shape, y_train.shape)

#For Compound
from sklearn.preprocessing import Normalizer
normalizer = Normalizer()

normalizer.fit(merdata['compound'].values.reshape(-1,1))

#X_train_price_norm = normalizer.transform(X_train['price'].values.reshape(-1,1))
X_train_com_norm = normalizer.transform(merdata['compound'].values.reshape(-1,1))

X_test_com_norm = normalizer.transform(test['compound'].values.reshape(-1,1))

print("After normalizations")
print(X_train_com_norm.shape, y_train.shape)
#Applying TF IDF vectorizer on preprocessed item description
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=10,max_features=5000)
vectorizer.fit(preprocessed_train_des)
X_train_itemdes = vectorizer.transform(preprocessed_train_des)
X_test_itemdes = vectorizer.transform(preprocessed_test_des)

# after preprocessing
#combining all the preprocessed features using horizontal stacking. This will give us final X_train and y_train and X_test.
from scipy.sparse import hstack
X_train = hstack((merdata_name , merdata_brandname,merdata_gencat,merdata_subcat1,merdata_subcat2,X_train_itemdes,x_dummies,x_train_words_des_norm)).tocsr()
X_test = hstack((test_name , test_brandname,test_gencat,test_subcat1,test_subcat2,X_test_itemdes,x_test_dummies,x_test_words_des_norm)).tocsr()
y_train = merdata.price


#Applying Models

#Linear Regression
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,y_train)
y_test_lr = lr.predict(X_test)

#Ridge Regression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
parameters = {"alpha":[0.01,0.1,0,1,10,100]}
ridgeReg = Ridge(solver = "lsqr", fit_intercept=False)
lr_reg = GridSearchCV(ridgeReg,param_grid =parameters,n_jobs=-1)
lr_reg.fit(X_train, y_train)
y_test_rr = lr_reg.predict(X_test)

#Stochastic Gradient Descent(SGD) Regression
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GridSearchCV
parameters = {"alpha": [0.0001,0.001,0.01,0.1,0,1,10,100] , "l1_ratio":[0.2,0.3,0,4,0.5,0.6,0.7,0.8,0.9]}

model = SGDRegressor(loss = "squared_loss", learning_rate = "invscaling",max_iter = 200, penalty = "l2",fit_intercept = False)

lr_reg = GridSearchCV(model,param_grid = parameters , n_jobs = -1)
lr_reg.fit(X_train , y_train)
y_test_sgdr = lr_reg.predict(X_test)

#Applying LGBM Regression
from lightgbm import LGBMRegressor
lgbm_params ={'subsample': 0.9, 'colsample_bytree': 0.8, 'min_child_samples': 50, 'objective': 'regression','boosting_type': 'gbdt','learning_rate': 0.5,'max_depth': 8,'n_estimators': 500,'num_leaves': 80 }
model = LGBMRegressor(**lgbm_params)
model.fit(X_train, y_train, early_stopping_rounds=100,verbose=True)
y_test_lgbmr = model.predict(X_test)

#Ensembling above 4 results
y_test = (y_test_lgbmr * 0.4 + y_test_lr * 0.2 + y_test_rr * 0.3 +y_test_sgdr *0.1)

#Forming Submission file.
test['price'] = y_test
Z = test[['id','price']]
Z.to_csv('submission.csv',index = False) 

