import json
f = open('userDefinedParameters.json','r')
param = json.load(f)
f.close()
# will come from json file later
vocabSize=param['vocabSize']
sequence_length=param['sequence_length']
#end
train_path = "../train/" # source data
test_path = "../test/" # test data for evaluation.

#Creating "imdb_train.csv","imdb_test.csv"
from nltk.tokenize import word_tokenize
def tokenize_data(data):
    words=word_tokenize(data)
    return words

def get_stopwords():
    from homebrewStopwords import stopwords
    return stopwords

'''
CLEAN_DATA takes a sentence and the stopwords as inputs
returns the sentence without any stopwords, html tags and punctuations. Also performs lemmatization and stemming
data - The input from which the stopwords have to be removed
stop_words_list - A list of stopwords
'''
from nltk.corpus import wordnet

treebank_to_wordnet_dict = {
    'J' : wordnet.ADJ, 'V' : wordnet.VERB, 'N' : wordnet.NOUN, 'R' : wordnet.ADV
    }

import string
import re
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
def clean_data(data,stop_words_list = get_stopwords()):
    data = data.replace('\n', '')
    # removes HTML tags
    data=re.sub('<(\d|\w|\s|/)*>', '', data)
    # removes punctuations
    #data=re.sub(r'[^\w\s]','', data)
    originalData = data
    data = data.translate(str.maketrans(".", " "))
    data = data.translate(str.maketrans('', '', string.punctuation + '_'))

    words=tokenize_data(data)
    words = [words.lower() for words in words]

    wn=WordNetLemmatizer()
    stemmed_words = [wn.lemmatize(w, (treebank_to_wordnet_dict.get(pos[0]) if treebank_to_wordnet_dict.get(pos[0]) is not None else wordnet.NOUN)) for w, pos in pos_tag(words)]

    useful_stemmed_words = [w for w in stemmed_words if w not in stop_words_list]

    result=' '.join(useful_stemmed_words)

    return result


'''
IMDB_DATA_PREPROCESS explores the neg and pos folders from aclImdb/train and creates a output_file in the required format
input_dir - Path of the training samples
output_dir - Path where the file has to be saved
Name  - Name with which the file has to be saved
Mix - Used for shuffling the data
'''
def performancePrint(i,type,name):
    if i%500 ==0 :
        print(i , " {} and for {} ".format(type,name))

import pandas as pd
import os
import numpy as np
def imdb_data_preprocess(input_dir, output_dir="./Dataset/", name="imdb_train.csv", mix=False):
    # from pandas import DataFrame, read_csv
    # import csv
    indices = []
    text = []
    rating = []

    i =  0
    # positive review are present in pos folder and labelled as 1
    for filename in os.listdir(input_dir+"pos"):
        data = open(input_dir+"pos/"+filename, 'r' , encoding="ISO-8859-1").read()
        data = clean_data(data)
        indices.append(i)
        text.append(data)
        rating.append("1")
        i = i + 1
        performancePrint(i,"pos",name)


    for filename in os.listdir(input_dir+"neg"):
        data = open(input_dir+"neg/"+filename, 'r' , encoding="ISO-8859-1").read()
        data = clean_data(data)
        indices.append(i)
        text.append(data)
        rating.append("0")
        i = i + 1
        performancePrint(i,"neg",name)

    Dataset = list(zip(indices,text,rating))

    if mix:
        np.random.shuffle(Dataset)

    df = pd.DataFrame(data = Dataset, columns=['row_Number', 'text', 'polarity'])
    df.to_csv(output_dir+name, index=False, header=True)


'''
RETRIEVE_DATA takes a CSV file as the input and returns the corresponding arrays of labels and data as output.
Name - Name of the csv file
Train - If train is True, both the data and labels are returned. Else only the data is returned
'''
import pandas as pd
def retrieve_data(input_dir='./Dataset/',name="imdb_train.csv"):
    data_dir = input_dir + name
    data = pd.read_csv(data_dir,header=0, encoding = 'ISO-8859-1')
    X = data['text']
    Y = data['polarity']
    return X, Y

'''
TFIDF_PROCESS takes the data to be fit as the input and returns a vectorizer of the tfidf as output
Data - The data for which the bigram model has to be fit
'''

from sklearn.feature_extraction.text import TfidfVectorizer
def tfidf_process(data,max_features=vocabSize):
    vectorizer = TfidfVectorizer(max_features=max_features, sublinear_tf = True)#, min_df = 0.02, max_df = 0.97)
    vectorizer.fit(data)
    return vectorizer

# Padding the sequences to a fixed length
from tensorflow.keras.preprocessing.sequence import pad_sequences
def add_padding_to_Xdata(xTrain,xTest, sequence_length):
    xTrain = pad_sequences( xTrain , maxlen=sequence_length , padding='pre', value=0 )
    xTest = pad_sequences( xTest , maxlen=sequence_length , padding='pre', value=0 )
    return xTrain,xTest

def sanityEmbeddings(processedText, vectorizer, tokenizer):
    documentTermMatrix = vectorizer.transform(processedText).toarray()
    vocabDictionary = vectorizer.vocabulary_
    intEmbeddings = []
    temp = []
    i = 0

    for document in processedText:
        topTfidfEmbeddings = {vocabDictionary.get(token) : documentTermMatrix[i][vocabDictionary.get(token)] for token in tokenizer(document) if vocabDictionary.get(token) is not None}

        #take middle 90% of the sorted embeddings based on tfidf score
        topTfidfEmbeddings = dict(sorted(topTfidfEmbeddings.items(), key = lambda item: item[1], reverse = False)
        [round(len(topTfidfEmbeddings) * 0.05):
        round(len(topTfidfEmbeddings) * 0.95)])

        for token in tokenizer(document):
            embedding = vocabDictionary.get(token)

            if embedding in topTfidfEmbeddings:
                temp.append(embedding)

        i += 1

        intEmbeddings.append(temp)
        temp = []

    return intEmbeddings


import time
import pickle
def load_data_self_preprocess(processData=True):
    start = time.time()

    if processData is True:
        print ("Preprocessing the training_data--")
        imdb_data_preprocess(input_dir=train_path,output_dir=train_path,name="imdb_train.csv",mix=True)
        print ("Preprocessing the testing_data--")
        imdb_data_preprocess(input_dir=test_path,output_dir=test_path,name="imdb_test.csv",mix=False)
        print ("Done with preprocessing in. Now, will retreieve the training data in the required format")

    (xTrain_text, yTrain) = retrieve_data(input_dir=train_path,name="imdb_train.csv")
    print ("Retrieved the training data. Now will retrieve the test data in the required format")
    (xTest_text,yTest) = retrieve_data(input_dir=test_path,name="imdb_test.csv")
    print ("Retrieved the test data. Now will initialize the model \n\n")
    print("As per choice we will use vocabulary size as {}".format(vocabSize))
    print('We will try to fit our train data usinf tfidf_vectorizer')

    tfidf_vectorizer = tfidf_process(xTrain_text,max_features=vocabSize)
    tokenizer = tfidf_vectorizer.build_tokenizer()

    with open('vocab.pkl', 'wb') as pklFile:
        pickle.dump(tfidf_vectorizer.vocabulary_, pklFile)

    with open('vectorizer.pkl', 'wb') as pklFile:
        pickle.dump(tfidf_vectorizer, pklFile)

    xTrain = sanityEmbeddings(xTrain_text, tfidf_vectorizer, tokenizer)
    xTest = sanityEmbeddings(xTest_text, tfidf_vectorizer, tokenizer)
    xTrain,xTest=add_padding_to_Xdata(xTrain,xTest, sequence_length)
    end=time.time()
    print('The data preparation took {} ms'.format(end-start))
    return (xTrain,yTrain),(xTest,yTest)


from tensorflow.keras.datasets import imdb
def load_data_keras_preproccesed(processData=False):
    (xTrain,yTrain),(xTest,yTest) = imdb.load_data( num_words=vocabSize)
    xTrain,xTest=add_padding_to_Xdata(xTrain,xTest, sequence_length)
    return (xTrain,yTrain),(xTest,yTest)


if __name__=='__main__':
    print("This file is for preprocessing the IMDB Dataset")
    print("Not meant to be run directly")
