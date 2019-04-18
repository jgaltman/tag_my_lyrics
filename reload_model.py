import os
import sys
import re
import pickle
import itertools
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.models import model_from_json

import config as cfg

#CONSTANTS
TEST_SPLIT = cfg.TEST_SPLIT
SAVE_RELOAD = cfg.SAVE_RELOAD

# PATH CONSTANTS
CNN_INPUT = cfg.CNN_INPUT

TOCKENIZER_PATH = cfg.TOCKENIZER_PATH
TEST_INDECIES_FILE = cfg.TEST_INDECIES_FILE

MODEL_LOAD_FILE = cfg.MODEL_FILE
MODEL_LOAD_WEIGHTS_FILE = cfg.BEST_WEIGHTS_FILE

CONFUSION_MATRIX_FILE = cfg.CONFUSION_MATRIX_RELOAD_FILE

# Default values - changed later
MAX_SONG_LENGTH = cfg.MAX_SONG_LENGTH

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

def plot_confusion_matrix(cm, g_index, error):
    cmap=plt.cm.Blues
    print('Confusion matrix, without normalization')
    print(cm)
    classes = ['' for i in range(len(g_index))]
    print(g_index)
    for key,val in g_index.items():
        classes[val] = key
    print(classes)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title('Average Recognition Accuracy: %.02f' %(error))
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

def check_accuracy(model2,x_test,y_test):
    print('%d,%d'%(len(x_test),len(y_test)))
    correct=0.0
    y_pred = model2.predict(x_test,verbose=0)
    matrix = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
    accuracy = np.sum(np.identity(len(genre_index))*matrix)/len(y_test)
    print('Accuracy: %.2f' %(accuracy))
    plt.figure()
    plot_confusion_matrix(matrix,genre_index,accuracy)
    if SAVE_RELOAD:
        plt.savefig(CONFUSION_MATRIX_FILE)
        print('saved confusion matrix')
    else:
        plt.show()
    plt.clf()


print('loading pickles')
pickle_data = pickle.load( open(CNN_INPUT , "rb" ))
lyrics = pickle_data['lyrics']
lyrics_labels = pickle_data['lyrics_labels']
unique_words_set = pickle_data['unique_words_set']
genre_index = pickle_data['genre_index']
MAX_SONG_LENGTH = round(pickle_data['longest_song'],-2)
print('number of songs: %d' %(len(lyrics)))
print('number of genres: %d' %(len(genre_index)))
print('number of lyrics: %d' %(len(lyrics_labels)))
print('number of unique words: %d' %(len(unique_words_set)))
print('longest song: %d' %(MAX_SONG_LENGTH))
print('finished loading pickles')

print('tokenizing')
tokenizer = pickle.load(open(TOCKENIZER_PATH,"rb"))

sequences = tokenizer.texts_to_sequences(lyrics)

data = keras.preprocessing.sequence.pad_sequences(sequences, maxlen=MAX_SONG_LENGTH,padding='post')
labels = keras.utils.to_categorical(np.asarray(lyrics_labels))

print('finished tokenizing')


# split the data into a training set and a validation set
pickle_test_index = pickle.load( open(TEST_INDECIES_FILE, "rb" ))

indices = pickle_test_index['indices']
print(indices)
data = data[indices]
labels = labels[indices]
shuffled_lyrics = np.array(lyrics)[indices]


num_test_samples = int(TEST_SPLIT * data.shape[0])

test_lyrics = shuffled_lyrics[:num_test_samples]
x_test = data[:num_test_samples]
y_test = labels[:num_test_samples]
x_train = data[num_test_samples:]
y_train = labels[num_test_samples:]

    
print('data tensor:', data.shape)
print('test tensor:', x_test.shape)
print('train tensor:', x_train.shape)
print('valid splits: ', x_test.shape[0]+x_train.shape[0] == data.shape[0])
print('label tensor:', labels.shape)
print('test tensor:', y_test.shape)
print('train tensor:', y_train.shape)
print('valid splits: ', y_test.shape[0]+y_train.shape[0] == data.shape[0])



model = load_model(MODEL_LOAD_FILE,MODEL_LOAD_WEIGHTS_FILE)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

scores= model.evaluate(x_test,y_test,verbose=0)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

check_accuracy(model,x_test,y_test)


