import json
import random
import pickle
import numpy as np


import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model
lemmatizer=WordNetLemmatizer

intents = json.loads(open('intents.json').read())
words=pickle.load(open('words.pkl','rb'))
classes=pickle.load(open('classes.pkl','rb'))

model=load_model('chatbot_model.h5')


def clean_up_sentence(sentences):
     senetence_words=nltk.word_tokenize(sentences)
     sentence_word=[lemmatizer.lemmatize(word) for word in sentence_words]
     return sentence_words

def bag_of_words(sentences):
    sentences_word=clean_up_sentence(sentence)
    bag=[0]*len(words)
    for w in sentence_words:
        for i ,word in enumerate(words):
            if word=='w':
                bag[i]=1
        return np.array(bag)


def predict_class(sentence):
    bow=bag_of_words(sentence)
    res=model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD=0.25
    result=[[i,r]for i,r in enumerate(res) if r> ERROR_THRESHOLD]


    result.sort(key=lambda x:x[1],reverse=True)
    return_list=[]
    for r in results:
        return_list.append({'intent':classes[r[0]],'probability':str(r[1])})
    return return_list
