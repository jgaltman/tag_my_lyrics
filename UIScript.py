import sys
import re
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.models import model_from_json
from tkinter import *
from tkinter import ttk  
from tkinter import scrolledtext
from tkinter import messagebox
from PIL import ImageTk, Image
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
nltk.download('stopwords')
nltk.download('punkt')
tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('english'))

#CONSTANTS
DATA_KEYS = ['lyrics','lyrics_labels','unique_words_set','longest_song','genre_index']
SAVE = True
VERSION = '2.0'
VALIDATION_SPLIT = 0.33
TEST_SPLIT = 0.2
learning_rate = .001
max_grad_norm = 1.
DROPOUT = 0.5
EMBEDDING_DIM = 100
epochs=5

# PATH CONSTANTS
PICKLE_ROOT = 'data/lyrics/'
PICKLE_INPUT = PICKLE_ROOT+'CNN_input.pickle' 

MODEL_DIR = 'saved_models/'
MODEL_LOAD_FILE = MODEL_DIR+'cnn_model_'+VERSION+'.json'
MODEL_LOAD_WEIGHTS_FILE = MODEL_DIR+'best_weights_'+VERSION+'.hdf5'

TEST_DIR = 'data/test/'
TOCKENIZER_PATH = TEST_DIR+'token.pickle'
GENRE_FILE = TEST_DIR+'genres.pickle'


def clean_data(song_lyrics):    
    word_tokens = tokenizer.tokenize(song_lyrics)
    filtered_song = [w for w in word_tokens if not w in stop_words]
    filtered_song = [x.lower() for x in filtered_song]
    filtered_song = ' '.join(filtered_song)
    return [filtered_song]

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
    def clicked():
        inputlyrics = (txt.get("1.0",'end-1c'))
        if(inputlyrics == ""):
            messagebox.showwarning('Error', 'Please enter lyrics')
        else:
            new_data = clean_data(inputlyrics)
            print(new_data)
            token = pickle.load(open(TOCKENIZER_PATH,"rb"))
            MAX_SONG_LENGTH = round(pickle.load(open(PICKLE_INPUT,"rb"))["longest_song"],-2)
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
                    lbl2.configure(text=g)
    def clicked_reset():
        txt.delete(1.0,END)
        lbl2.configure(text="")

    model = load_model(MODEL_LOAD_FILE,MODEL_LOAD_WEIGHTS_FILE)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    #x_test = clean input
    genre_index = pickle.load( open(GENRE_FILE, "rb" ))
    #inputlyrics = "Cardinal in the white snow They got too high to fly home For the winter, and I had a roommate once He got so high, couldn't go out at night So he found love on the internetAnd it is freezing in Pennsylvania"
    window = Tk()
    window.title("Tag My Lyrics")
    lbl = Label(window, text="♬ Classify Lyrics to Genre ♬", font=("Arial Bold", 50))
    lbl.grid(column=0, row=0)
    lbl3 = Label(window,text = "Paste your lyrics below",font=("Arial Bold", 25))
    lbl3.grid(column=0,row =1)
    txt = scrolledtext.ScrolledText(window,width=60,height=25)
    txt.grid(column=0,row=2)
    btn = Button(window, text="Classify", command=clicked,font=("Arial Bold", 25)) 
    btn.grid(column=0, row=3)
    #btn.config( height = 6, width = 12 )
    lbl2 = Label(window, text="", font=("Arial Bold", 50))
    lbl2.grid(column = 0, row = 4)
    btn2 = Button(window, text="Reset", command=clicked_reset,font=("Arial Bold", 15)) 
    btn2.grid(column=0, row=5)
    window.configure(background='LightBlue1')
    lbl.configure(background = 'LightBlue1')
    lbl2.configure(background = 'LightBlue1')
    lbl3.configure(background = 'LightBlue1')
    window.geometry('900x600')
    window.mainloop()


if __name__ == '__main__':
    main(sys.argv)


 
