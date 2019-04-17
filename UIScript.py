import os
import sys
import re
import csv
import pickle
import itertools
import numpy as np
import math as m
import datetime
from pprint import pprint
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import backend
from tensorflow.python.client import device_lib
from tensorflow.keras.models import model_from_json
from tensorflow.keras.layers import Dense, Input, Flatten, Reshape, concatenate, Dropout
from tensorflow.keras.layers import Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Embedding
from tensorflow.keras.layers import LSTM, Bidirectional
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
nltk.download('stopwords')
nltk.download('punkt')
tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('english'))


# MAX_SONG_LENGTH = 2500
# SONG_PER_GENRE = 4500
DATA_KEYS = ['lyrics','lyrics_labels','unique_words_set','longest_song','genre_index']
SAVE = True

#CONSTANTS
VALIDATION_SPLIT = 0.33
TEST_SPLIT = 0.2
learning_rate = .001
max_grad_norm = 1.
DROPOUT = 0.5
EMBEDDING_DIM = 100
epochs=5


# PATH CONSTANTS
PICKLE_ROOT = 'data/lyrics/'
PICKLE_INPUT = 'CNN_input.pickle' 
CSV_TEST = 'recent_testdata_'+str(epochs)+'.csv' 
PICKLE_TEST = 'recent_testdata_'+str(epochs)+'.pickle' 


MODEL_DIR = 'saved_models/'
MODEL_LOAD_FILE = MODEL_DIR + 'cnn_model_1.1_'+str(epochs)+'.json'
MODEL_LOAD_WEIGHTS_FILE = MODEL_DIR + 'best_weights5.hdf5'

# Default values - changed later
# MAX_SONG_LENGTH = 2500
# MAX_UNIQUE_WORDS = 20000


def remove_stop_and_punct(article):
    final_article = []
    word_tokens = tokenizer.tokenize(article)
    filtered_article = [w for w in word_tokens if not w in stop_words]
    filtered_article = [x.lower() for x in filtered_article]
    filtered_song = ' '.join(filtered_article)
    return filtered_song, filtered_article

def clean_genre(data):
    song_list = []
    unique_words_list = []
    count = 0
    max_length_song = 0
    # for key, song_info in data['lyrics'].items():
    #     title, artist = key
    #     inner_title = song_info['title']
    #     if count%1000==0:
    #         print('iter - %d: song - %s' %(count, inner_title))
    #     inner_artist = song_info['artist']
    song_lyrics = data
    
#       song_lyrics_norm = re.sub(r'[^a-zA-Z0-9-\']', ' ', song_lyrics).strip()
#       song_lyrics_split = song_lyrics_norm.lower().split()
    song_lyrics_norm, song_lyrics_split = remove_stop_and_punct(song_lyrics)
    # unique_words_list = list(set(unique_words_list + song_lyrics_split))
    # count+=1
        
    # if count >= SONG_PER_GENRE:
    #     print('hit max songs: %d' %(SONG_PER_GENRE))
    #     print('songs left out: %d' %(len(data['lyrics'])-SONG_PER_GENRE))
    #     print('longest song length: %d' %(max_length_song))
    #     return song_list, unique_words_list, max_length_song
       
    return song_lyrics_norm,song_lyrics_split

def clean_data(p_lyrics,genre_index):
    lyrics = []
    # lyrics_labels = []
    # unique_words_set = []
    # longest_song = 0
    # data = p_lyrics
    # genre = data['genre']
    #print('cleaning: %s' %(genre))
    norm,split = clean_genre(p_lyrics)
    # unique_words_set = list(set(unique_words_set+unique_words))
    # #song_labels = [genre_index[genre]]*len(song_list)
    # if longest_song < g_longest_song:
    #     longest_song = g_longest_song
    # lyrics = lyrics + song_list
    #lyrics_labels = lyrics_labels + song_labels
    return [norm]


def load_model(filename,weights_filename):
    # load json and create model
    json_file = open(filename, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(weights_filename)
    print("Loaded model from disk")
    return loaded_model

def main(argv):
    model = load_model(MODEL_LOAD_FILE,MODEL_LOAD_WEIGHTS_FILE)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    #x_test = clean input
    pickle_data = pickle.load( open(PICKLE_ROOT + "genres.pickle" , "rb" ))
    genre_index = pickle_data
    #inputlyrics = "Cardinal in the white snow They got too high to fly home For the winter, and I had a roommate once He got so high, couldn't go out at night So he found love on the internetAnd it is freezing in Pennsylvania"
    with open('song_test.txt', 'r') as file:
        inputlyrics = file.read().replace('\n', '')
    new_data = clean_data(inputlyrics,genre_index)
    print(new_data)
    token = pickle.load(open("data/test/token.pickle","rb"))
    MAX_SONG_LENGTH = round(pickle.load(open(PICKLE_ROOT+"CNN_input.pickle","rb"))["longest_song"],-2)
    print(MAX_SONG_LENGTH)
    print('tokenizing')
    sequences = token.texts_to_sequences(new_data)
    data = keras.preprocessing.sequence.pad_sequences(sequences, maxlen=MAX_SONG_LENGTH,padding='post')
    print(len(data[0]))
    print('finished tokenizing')
    scores= model.predict(data,verbose=0)
    genreNumber = scores.argmax(axis = 1)
    for g in genre_index.keys():
        if(genre_index[g] == genreNumber):
            print(g)


if __name__ == '__main__':
    main(sys.argv)