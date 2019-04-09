#!/usr/bin/env python
# coding: utf-8

# In[100]:


import os
import sys
import re
import pickle
import numpy as np
from pprint import pprint
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import backend
from tensorflow.python.client import device_lib
from tensorflow.keras.models import model_from_json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#### Uncomment on server
# device = device_lib.list_local_devices()
# print(device)
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

# backend.set_session(sess)


# In[165]:


#CONSTANTS
VALIDATION_SPLIT = 0.3
TEST_SPLIT = 0.2
learning_rate = .001
max_grad_norm = 1.
DROPOUT = 0.5
EMBEDDING_DIM = 200


# In[3]:


# PATH CONSTANTS
PICKLE_ROOT = 'data/lyrics/'
PICKLE_INPUT = 'CNN_input.pickle' 

EMBEDDING_PATH = 'data/glove_embeddings/'
EMBEDDING_FILE = 'glove.6B.'+str(EMBEDDING_DIM)+'d.txt'

MODEL_SAVE_FILE = 'cnn_model_1.0.json'
MODEL_SAVE_WEIGHTS_FILE = 'cnn_model_1.0.h5'


# In[160]:


# Default values - changed later
MAX_SONG_LENGTH = 2500
MAX_UNIQUE_WORDS = 20000


# In[39]:


# Embedding
# Elmo could improve the word embeddings - need more research
# elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
print('loading embedding')
if not os.path.exists(EMBEDDING_PATH+EMBEDDING_FILE):
    print('Embeddings not found, downloading now')
    print(os.system('pwd'))
    os.system(' cd ' + DATA_PATH)
    os.system(' mkdir ' + EMBEDDING_DIR)
    os.system(' cd ' + EMBEDDING_DIR)
    os.system(' wget http://nlp.stanford.edu/data/glove.6B.zip')
    os.system(' unzip glove.6B.zip')
    os.system(' cd ../..')

glove_embeddings = {}
with open(EMBEDDING_PATH+EMBEDDING_FILE, encoding='utf-8') as emb_f:
    for line in emb_f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype='float32')
        glove_embeddings[word] = vector
print('finished loading embedding')


# In[191]:



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


# In[192]:


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


# In[152]:


# save model
def save_model(nn_model,filename,weights_filename):
    # serialize model to JSON
    model_json = nn_model.to_json()
    with open(filename, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    nn_model.save_weights(weights_filename)
    print("Saved model to disk")


# In[153]:



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


# In[193]:


# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]



t_data = data[:int(data.shape[0]*.1)]
t_labels = labels[:int(data.shape[0]*.1)]

# NO GPU so must downsize
CPU=False
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


# In[230]:


print('Building model')
# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = keras.layers.Embedding(unique_words_count,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SONG_LENGTH,
                            trainable=False)

CONV1D_OUT = 128
OUT = 5
MAX_POOLING_1 = 5
MAX_POOLING_2 = 47
DROPOUT = 0.5

# conv1d = single spatial convolution of 2d input (sequence of 1000 - 200 demention vectors)
# maxpooling1d = randomly downsizes by pooling val
# dropout = randomly zeros at dropout rate to avoid overfitting

model2 = Sequential()
model2.add(embedding_layer)
model2.add(tf.keras.layers.Conv1D(CONV1D_OUT, OUT, activation='relu'))
model2.add(tf.keras.layers.MaxPooling1D(MAX_POOLING_1))
model2.add(tf.keras.layers.Dropout(DROPOUT))

model2.add(tf.keras.layers.Conv1D(CONV1D_OUT, OUT, activation='relu'))
model2.add(tf.keras.layers.MaxPooling1D(MAX_POOLING_1))
model2.add(tf.keras.layers.Dropout(DROPOUT))

model2.add(tf.keras.layers.Conv1D(CONV1D_OUT, OUT, activation='relu'))
model2.add(tf.keras.layers.MaxPooling1D(MAX_POOLING_2))
model2.add(tf.keras.layers.Flatten())

model2.add(tf.keras.layers.Dense(CONV1D_OUT, activation='relu'))
model2.add(tf.keras.layers.Dense(len(genre_index), activation='softmax'))
# loss=binary_crossentropy
# model2.compile(loss='categorical_crossentropy',
#               optimizer='adam',
#               metrics=['acc'])
# adam = tf.keras.optimizers.Adam(lr=learning_rate, clipnorm=max_grad_norm)
model2.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])
model2.summary()


# In[231]:


print('Training Model')
model_details = model2.fit(x_train, y_train,
            epochs=100,
            shuffle=True,
            verbose=1,
            validation_split=VALIDATION_SPLIT)

scores= model2.evaluate(x_test,y_test,verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])


# In[232]:


def check_accuracy(model2,x_test,y_test):
    print('%d,%d'%(len(x_test),len(y_test)))
    correct=0.0
    for x,y in zip(x_test, y_test):
        x = np.reshape(x,(1,-1))
        y = np.reshape(y,(1,-1))
        scores = model2.predict(x,verbose=0)
    #     print(scores)
    #     print(y)
    #     print(np.argmax(scores))
    #     print(np.argmax(y))
        if np.argmax(scores)==np.argmax(y):
            correct+=1.0
    print('Accuracy: %.2f' %(100*correct/len(x_test)))
    #     print('Test loss:', scores[0])
    #     print('Test accuracy:', scores[1])


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
#     plt.show()
    plt.savefig('accuracy.png')
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
#     plt.show()
    plt.savefig('loss.png')


# In[ ]:


check_accuracy(model2,x_test,y_test)
plot_data(model_details)
save_model(model2,MODEL_SAVE_FILE, MODEL_SAVE_WEIGHTS_FILE)

