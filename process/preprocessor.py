import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from string import punctuation
import string
import os
import pickle
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('punkt')

def remove_text_special (text:str)->str:
    text = text.replace('\\t',"").replace('\\n',"").replace('\\u',"").replace('\\',"")
    text = text.encode('ascii', 'replace').decode('ascii')
    return text.replace("http://"," ").replace("https://", " ")

"""
    CASE FOLDING 
"""

def casefolding(Comment:str)->str:
    Comment = Comment.lower()
    return Comment

"""
    PUNCTUATION
"""

def cleansing(text:str)->str:
    text = re.sub("@[A-Za-z0-9_]+", "", text) #clenasing mention
    text = re.sub("#[A-Za-z0-9_]+", "", text) #clenasing hashtag
    text = re.sub(r'https?://\S+|www\.\S+', "", text) #cleansing url link
    text = re.sub("[^a-zA-Zï ]+", " ", text) #cleansing character
    text = re.sub(r'[-+]?[0-9]+', "", text) #cleansing number
    return text

"""
    TOKENIZING
"""

def tokenizing(text):
    return nltk.word_tokenize(text)

"""
    STOPWARD REMOVAL
"""  

stop_words = set(stopwords.words('indonesian'))
def stopwords_removal(filtering):
    # # Menambah kata dalam stopword
    # more_stopword = ['btw','dll','anw']
    # stop_words.update(more_stopword)
    # Proses
    filtering = [word for word in filtering if word not in stop_words]
    return filtering

"""
    STEMMING
"""

# Inisialisasi stemmer untuk bahasa Indonesia
factory = StemmerFactory()
stemmer = factory.create_stemmer()
def stemming (document:list)->list:   
    # Menjalankan stemming
    result = []
    for term in document:
        result.append(stemmer.stem(term))
    return result

def preprocessing(text:str)->list:
    text = remove_text_special(text)
    text = casefolding(text)
    text = cleansing (text)
    tokenize = tokenizing(text)
    tokenize = stopwords_removal(tokenize)
    tokenize = stemming(tokenize)

    return tokenize

with open("./model/tfidf11.pickle", 'rb') as file:
    vectorizer11 = pickle.load(file)
with open("./model/tfidf12.pickle", 'rb') as file:
    vectorizer12 = pickle.load(file)
with open("./model/tfidf.pickle", 'rb') as file:
    vectorizer2 = pickle.load(file)
with open("./model/tfidf31.pickle", 'rb') as file:
    vectorizer31 = pickle.load(file)