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

#CONSTANTS
VALIDATION_SPLIT = 0.33
TEST_SPLIT = 0.2
learning_rate = .001
max_grad_norm = 1.
DROPOUT = 0.5
EMBEDDING_DIM = 100
epochs=10


# PATH CONSTANTS
PICKLE_ROOT = 'data/lyrics/'
PICKLE_INPUT = 'CNN_input.pickle' 
CSV_TEST = 'recent_testdata_'+str(epochs)+'.csv' 
PICKLE_TEST = 'recent_testdata_'+str(epochs)+'.pickle' 


MODEL_DIR = 'saved_models/'
MODEL_LOAD_FILE = MODEL_DIR + 'cnn_model_1.1_'+str(epochs)+'.json'
MODEL_LOAD_WEIGHTS_FILE = MODEL_DIR + 'best_weights.hdf5'

# Default values - changed later
MAX_SONG_LENGTH = 2500
MAX_UNIQUE_WORDS = 20000


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
    print('saved confusion matrix')

def check_accuracy(model2,x_test,y_test):
    print('%d,%d'%(len(x_test),len(y_test)))
    correct=0.0
    # for x,y in zip(x_test, y_test):
    # x = np.reshape(x,(1,-1))
    # y = np.reshape(y,(1,-1))
    y_pred = model2.predict(x_test,verbose=0)
    # print(scores)
    # print(y)
    # print(np.argmax(scores))
    # print(np.argmax(y))
    # if np.argmax(scores)==np.argmax(y):
    #     correct+=1.0
    matrix = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
    accuracy = np.sum(np.identity(len(genre_index))*matrix)/len(y_test)
    print('Accuracy: %.2f' %(accuracy))
    plt.figure()
    plot_confusion_matrix(matrix,genre_index,accuracy)
    plt.savefig('reloaded_confusion_matrix.png')
    plt.clf()
    #     print('Test loss:', scores[0])
    #     print('Test accuracy:', scores[1])




print('loading pickles')
pickle_data = pickle.load( open(PICKLE_ROOT + PICKLE_INPUT , "rb" ))
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
tokenizer = keras.preprocessing.text.Tokenizer(num_words=MAX_UNIQUE_WORDS)
tokenizer.fit_on_texts(lyrics)
sequences = tokenizer.texts_to_sequences(lyrics)

word_index = tokenizer.word_index
print('Unique words tokens %d' % (len(word_index)))

data = keras.preprocessing.sequence.pad_sequences(sequences, maxlen=MAX_SONG_LENGTH,padding='post')
labels = keras.utils.to_categorical(np.asarray(lyrics_labels))

print('finished tokenizing')


# split the data into a training set and a validation set
pickle_test_index = pickle.load( open(PICKLE_ROOT + PICKLE_TEST , "rb" ))
# indices = []
# with open(PICKLE_ROOT + CSV_TEST, 'r') as f:
#   reader = csv.reader(f)
#   indices = list(reader)

indices = pickle_test_index['indices']
print(indices)
# indices = np.arange(data.shape[0])
# np.random.shuffle(indices)
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


def get_partial_output(model,x_input):
    layer_outputs = [layer.output for layer in model.layers[:12]] 
    # Extracts the outputs of the top 12 layers
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs) # Creates a model that will return these outputs, given the model input
    activations = activation_model.predict(x_input) 
    print(activations)


