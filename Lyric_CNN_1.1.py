#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
import gensim 
from gensim.models import doc2vec
from gensim.models.doc2vec import Doc2Vec

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
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
from tensorflow.keras.callbacks import ModelCheckpoint


# In[2]:


# SERVER VARIABLES
#### Uncomment on server
# device = device_lib.list_local_devices()
# print(device)
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

# backend.set_session(sess)

# NO GPU so must downsize
CPU=True
epochs = 1


# In[3]:


#CONSTANTS
VALIDATION_SPLIT = 0.33
TEST_SPLIT = 0.2
learning_rate = .001
max_grad_norm = 1.
DROPOUT = 0.5
EMBEDDING_DIM = 100

# PATH CONSTANTS
PICKLE_ROOT = 'data/lyrics/'
PICKLE_INPUT = 'CNN_input.pickle' 

EMBEDDING_PATH = 'data/glove_embeddings/'
EMBEDDING_FILE = 'glove.6B.'+str(EMBEDDING_DIM)+'d.txt'

MODEL_DIR = 'saved_models/'
MODEL_SAVE_FILE = 'cnn_model_1.1_'+str(epochs)+'.json'
MODEL_SAVE_WEIGHTS_FILE = 'cnn_model_1.1_'+str(epochs)+'.h5'
BEST_WEIGHTS_FILE = 'best_weights'+str(epochs)+'.hdf5'

GRAPHS_DIR = 'graphs_out/'

DOC2VEC_PATH = MODEL_DIR + 'doc2vec/'
DOC2VEC_FILE = 'd2v.model'

# Default values - changed later
MAX_SONG_LENGTH = 2500
MAX_UNIQUE_WORDS = 20000


# In[4]:


# Embedding
# Elmo could improve the word embeddings - need more research
# elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
print('loading embedding')
try:
    if not os.path.exists(EMBEDDING_PATH+EMBEDDING_FILE):
        print('Embeddings not found, downloading now')
        try:
            print(os.system('pwd'))
            os.system(' cd ' + DATA_PATH)
            os.system(' mkdir ' + EMBEDDING_DIR)
            os.system(' cd ' + EMBEDDING_DIR)
            os.system(' wget http://nlp.stanford.edu/data/glove.6B.zip')
            os.system(' unzip glove.6B.zip')
            os.system(' cd ../..')
        except:
            print('not optimized for this operating system.')
            print('please download: ')
            print('http://nlp.stanford.edu/data/glove.6B.zip')
            print('Note: this may take a while')
            sys.exit()
except:
    print('Do you have the word embeddings?')

glove_embeddings = {}
with open(EMBEDDING_PATH+EMBEDDING_FILE, encoding='utf-8') as emb_f:
    for line in emb_f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype='float32')
        glove_embeddings[word] = vector
print('finished loading embedding')


# In[5]:



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


# In[6]:


# MAX_UNIQUE_WORDS = len(unique_words_set)
# MAX_SONG_LENGTH = 1000
# data preparing
print('tokenizing')
tokenizer = keras.preprocessing.text.Tokenizer(num_words=MAX_UNIQUE_WORDS)
tokenizer.fit_on_texts(lyrics)
sequences = tokenizer.texts_to_sequences(lyrics)

word_index = tokenizer.word_index
print('Unique words tokens %d' % (len(word_index)))

data = keras.preprocessing.sequence.pad_sequences(sequences, maxlen=MAX_SONG_LENGTH,padding='post')
labels = keras.utils.to_categorical(np.asarray(lyrics_labels))

print('finished tokenizing')


# In[7]:


# save model
def save_model(nn_model,filename,weights_filename):
    # serialize model to JSON
    model_json = nn_model.to_json()
    with open(filename, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    nn_model.save_weights(weights_filename)
    print("Saved model to disk")


# In[8]:



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


# In[9]:


def save_test_data(ind):
    filename = 'recent_testdata_'+str(epochs)+'.pickle'
    data_ind = {}
    data_ind['indices'] = ind
    pickle.dump( data_ind, open(PICKLE_ROOT+filename, "wb" ) )
    print('saved test data to %s%s' %(PICKLE_ROOT,filename))


# In[10]:


# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
save_test_data(indices)

t_data = data[:int(data.shape[0]*.1)]
t_labels = labels[:int(data.shape[0]*.1)]

if not CPU:
    num_test_samples = int(TEST_SPLIT * data.shape[0])
    x_test = data[:num_test_samples]
    y_test = labels[:num_test_samples]
    x_train = data[num_test_samples:]
    y_train = labels[num_test_samples:]
    
else:
    num_test_samples = int(TEST_SPLIT * t_data.shape[0])

    x_test = t_data[:num_test_samples]
    y_test = t_labels[:num_test_samples]
    x_train = t_data[num_test_samples:]
    y_train = t_labels[num_test_samples:]

print('data tensor:', data.shape)
print('test tensor:', x_test.shape)
print('train tensor:', x_train.shape)
print('valid splits: ', x_test.shape[0]+x_train.shape[0] == data.shape[0])
print('label tensor:', labels.shape)
print('test tensor:', y_test.shape)
print('train tensor:', y_train.shape)
print('valid splits: ', y_test.shape[0]+y_train.shape[0] == data.shape[0])

print('Preparing embedding matrix.')
# prepare embedding matrix
unique_words_count = min(MAX_UNIQUE_WORDS, len(word_index))
embedding_matrix = np.zeros((unique_words_count, EMBEDDING_DIM))

for word, i in word_index.items():
    if i >= MAX_UNIQUE_WORDS:
        continue
    embedding_vector = glove_embeddings.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        # potentially can improve if OOV words are handled differently        
        embedding_matrix[i] = embedding_vector


# In[11]:


# loading doc2vec model
# train_lyrics = np.array(lyrics)[indices]
# print(train_lyrics[0])
# def read_corpus(_data, tokens_only=False):
#     i = 0
#     for key,line in _data.items():
#         if tokens_only:
#             yield gensim.utils.simple_preprocess(line)
#         else:
#             # For training data, add tags
#             yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i])
#         i+=1
        
# train_corpus = list(read_corpus(train_lyrics))
# model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=40)
# model.build_vocab(train_corpus)
# %time model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

# def lyric2vec():
    
    

# d2v_model = Doc2Vec.load(DOC2VEC_PATH + DOC2VEC_FILE)
# print(d2v_model.infer_vector(train_lyrics[0]))


# In[12]:


def create_2dconv_model():
    embedding_layer = keras.layers.Embedding(unique_words_count,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SONG_LENGTH,
                                trainable=True)
    
    sequence_input = Input(shape=(MAX_SONG_LENGTH,))
    embedded_sequences = embedding_layer(sequence_input)
#     print(embedded_sequences.shape)
    # add first conv filter
    embedded_sequences = Reshape((MAX_SONG_LENGTH, EMBEDDING_DIM, 1))(embedded_sequences)
    x = Conv2D(100, (5, EMBEDDING_DIM), activation='relu')(embedded_sequences)
    x = MaxPooling2D((MAX_SONG_LENGTH - 10 + 1, 1))(x)
    # add second conv filter.
    y = Conv2D(100, (4, EMBEDDING_DIM), activation='relu')(embedded_sequences)
    y = MaxPooling2D((MAX_SONG_LENGTH - 8 + 1, 1))(y)
    # add third conv filter.
    z = Conv2D(100, (3, EMBEDDING_DIM), activation='relu')(embedded_sequences)
    z = MaxPooling2D((MAX_SONG_LENGTH - 6 + 1, 1))(z)
    # concate the conv layers
    alpha = concatenate([x,y,z])

#     alpha = Dropout(0.5)(alpha)
    
    # flatted the pooled features.
    alpha = Flatten()(alpha)
    
    # dropout
    alpha = Dropout(0.5)(alpha)
    
#   alpha = Dense(50, activation='relu')(alpha)
    
    # predictions
    preds = Dense(len(genre_index), activation='softmax')(alpha)
    # build model
    model = Model(sequence_input, preds)
#     adadelta = optimizers.Adam()
        
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    
    return model


# In[13]:


print('Building model')
# opt = tf.keras.optimizers.Adam(lr=learning_rate, clipnorm=max_grad_norm)
# model = create_basic_cnn_model()
# model = create_complex_cnn_model(False)
checkpointer = ModelCheckpoint(filepath=BEST_WEIGHTS_FILE, 
                               monitor = 'val_acc',
                               verbose=1, 
                               save_best_only=True)
model = create_2dconv_model()
# model = create_conv_lstm_model()
model.summary()


# In[14]:


print('Training Model')
model_details = model.fit(x_train, y_train,
            batch_size=128,
            epochs=epochs,
            shuffle=True,
            callbacks=[checkpointer],
            verbose=1,
            validation_split=VALIDATION_SPLIT)

scores= model.evaluate(x_test,y_test,verbose=0)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])


# In[26]:


# # layer_outputs = [layer.output for layer in model.layers]
# print(x_train[0].shape)
# print(type(x_train[0]))
# print(x_train.shape)

# inp = model.input                                           # input placeholder
# outputs = [layer.output for layer in model.layers]          # all layer outputs
# functor = backend.function([inp, backend.learning_phase()], outputs )   # evaluation function

# # Testing
# layer_outs = functor([x_train[0], 1.])
# print(layer_outs)



# layer_outputs = [layer.output for layer in model.layers if layer.name == layer_name or layer_name is None][1:]
# activation_model = Model(inputs=model.input, outputs=layer_outputs)

# activations = activation_model.predict(x_train[0].reshape(1,1300))

# def display_activation(activations, col_size, row_size, act_index): 
#     activation = activations[act_index]
#     print(activation)
#     activation_index=0
#     fig, ax = plt.subplots(row_size, col_size, figsize=(row_size*2.5,col_size*1.5))
#     for row in range(0,row_size):
#         for col in range(0,col_size):
#             ax[row][col].imshow(activation[0, :, :, activation_index], cmap='gray')
#             activation_index += 1

# display_activation(activations, 8, 8, 1)


# In[ ]:


def plot_confusion_matrix(cm, classes, error):
    cmap=plt.cm.Blues
    print('Confusion matrix, without normalization')
    print(cm)
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
    y_pred = model2.predict(x_test,verbose=0)
    matrix = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
    accuracy = np.sum(np.identity(len(genre_index))*matrix)/len(y_test)
    print('Accuracy: %.2f' %(accuracy))
    plt.figure()
    plot_confusion_matrix(matrix,list(genre_index.keys()),accuracy)
    plt.savefig(GRAPHS_DIR+'confusion_matrix_'+str(epochs)+'.png')
    print('saved confusion matrix')

    plt.clf()


# In[ ]:


def plot_data(history):
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    # plt.show()
    plt.savefig(GRAPHS_DIR+'accuracy_'+str(epochs)+'.png')
    plt.clf()
    print('saved accuracy')

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    # plt.show()
    plt.savefig(GRAPHS_DIR+'loss_'+str(epochs)+'.png')
    plt.clf()
    print('saved loss')


# In[ ]:


check_accuracy(model,x_test,y_test)
plot_data(model_details)
save_model(model,MODEL_SAVE_FILE, MODEL_SAVE_WEIGHTS_FILE)


# In[ ]:


################## OLD MODELS ############


# In[ ]:


# def create_basic_cnn_model():
#     embedding_layer = keras.layers.Embedding(unique_words_count,
#                                 EMBEDDING_DIM,
#                                 weights=[embedding_matrix],
#                                 input_length=MAX_SONG_LENGTH,
#                                 trainable=False)
#     model = Sequential()
#     model.add(embedding_layer)
#     model.add(keras.layers.Conv1D(128, 5, activation='relu'))
#     model.add(keras.layers.GlobalMaxPooling1D())
#     model.add(keras.layers.Dense(10, activation='relu'))
#     model.add(keras.layers.Dense(len(genre_index), activation='softmax'))
#     model.compile(loss='categorical_crossentropy',
#                   optimizer='adam',
#                   metrics=['acc'])
#     return model


# In[ ]:


# def create_conv_lstm_model():
#     embedding_layer = keras.layers.Embedding(unique_words_count,
#                                 EMBEDDING_DIM,
#                                 weights=[embedding_matrix],
#                                 input_length=MAX_SONG_LENGTH,
#                                 trainable=False)
#     model_conv = Sequential()
#     model_conv.add(embedding_layer)
#     model_conv.add(Dropout(0.2))
#     model_conv.add(Conv1D(64, 5, activation='relu'))
#     model_conv.add(MaxPooling1D(pool_size=4))
#     model_conv.add(LSTM(EMBEDDING_DIM))
#     model_conv.add(Dense(len(genre_index), activation='softmax'))
#     model_conv.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
#     return model_conv


# In[ ]:


# def create_complex_cnn_model(embedding_trained):

#     embedding_layer = keras.layers.Embedding(unique_words_count,
#                                 EMBEDDING_DIM,
#                                 weights=[embedding_matrix],
#                                 input_length=MAX_SONG_LENGTH,
#                                 trainable=embedding_trained)
#     CONV1D_OUT = 256
#     CONV1D_OUT2 = CONV1D_OUT//2
#     CONV1D_OUT3 = CONV1D_OUT2//2
#     OUT = 5
#     MAX_POOLING_1 = 8
#     MAX_POOLING_2 = 8
#     MAX_POOLING_4 = 13 #when MAX_SONG_LENGTH = 1300
#     # MAX_POOLING_3 = 47 #when MAX_SONG_LENGTH = 1300
#     # MAX_POOLING_2 = 35 #when MAX_SONG_LENGTH = 1000
#     # DROPOUT = 0.5

#     # conv1d = single spatial convolution of 2d input (sequence of 1000 - 200 demention vectors)
#     # maxpooling1d = randomly downsizes by pooling val
#     # dropout = randomly zeros at dropout rate to avoid overfitting

#     model2 = Sequential()
#     model2.add(embedding_layer)

#     model2.add(tf.keras.layers.Conv1D(CONV1D_OUT, OUT, activation='relu'))
#     model2.add(tf.keras.layers.MaxPooling1D(MAX_POOLING_1))
#     model2.add(tf.keras.layers.Dropout(DROPOUT))

#     model2.add(tf.keras.layers.Conv1D(CONV1D_OUT2, OUT, activation='relu'))
#     model2.add(tf.keras.layers.MaxPooling1D(MAX_POOLING_2))
#     model2.add(tf.keras.layers.Dropout(DROPOUT))

#     model2.add(tf.keras.layers.Conv1D(CONV1D_OUT2, OUT, activation='relu'))
#     model2.add(tf.keras.layers.MaxPooling1D(MAX_POOLING_3))
#     model2.add(tf.keras.layers.Flatten())

#     model2.add(tf.keras.layers.Dense(CONV1D_OUT3, activation='relu'))
#     model2.add(tf.keras.layers.Dense(len(genre_index), activation='softmax'))

#     # opt = tf.keras.optimizers.RMSprop(lr=learning_rate, clipnorm=max_grad_norm)
#     model2.compile(loss='categorical_crossentropy',
#                   optimizer='adam',
#                   metrics=['acc'])
#     return model2

