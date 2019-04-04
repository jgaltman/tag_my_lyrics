#!/usr/bin/env python
# coding: utf-8

# In[70]:

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras import backend
import pickle
import matplotlib.pyplot as plt
from pprint import pprint
import re
from tensorflow.python.client import device_lib

device_lib.list_local_devices()

# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

# backend.set_session(sess)

# In[131]:

#CONSTANTS
MAX_SONG_LENGTH = 2500
# MAX_NUM_WORDS = 20000
VALIDATION_SPLIT = 0.3
TEST_SPLIT = 0.2
learning_rate = .001
max_grad_norm = 1.
dropout = 0.5
EMBEDDING_DIM = 200


# In[83]:


# PATH CONSTANTS
PICKLE_ROOT = 'data/lyrics/'
CHRISTIAN_PATH = 'Christian.pickle'
POP_PATH = 'Pop.pickle'
ROCK_PATH = 'Rock.pickle'
COUNTRY_PATH = 'Country.pickle'
RAP_PATH = 'Rap.pickle'

LYRIC_PATHS = [CHRISTIAN_PATH,POP_PATH,ROCK_PATH,COUNTRY_PATH,RAP_PATH]

DATA_PATH = 'data/'
EMBEDDING_DIR = 'glove_embeddings'
EMBEDDING_PATH = DATA_PATH + EMBEDDING_DIR
EMBEDDING_FILE = 'glove.6B.'+str(EMBEDDING_DIM)+'d.txt'


# In[68]:


# Embedding
# Elmo could improve the word embeddings - need more research
# elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
if not os.path.exists(EMBEDDING_PATH+EMBEDDING_FILE):
    print('Embeddings not found, downloading now')
    os.system(' cd ' + DATA_PATH)
    os.system(' mkdir ' + EMBEDDING_DIR)
    os.system(' cd ' + EMBEDDING_DIR)
    os.system(' wget http://nlp.stanford.edu/data/glove.6B.zip')
    os.system(' unzip glove.6B.zip')
    os.system(' cd ../..')

glove_embeddings = {}
with open(EMBEDDING_PATH+EMBEDDING_FILE) as emb_f:
    for line in emb_f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype='float32')
        glove_embeddings[word] = vector


# In[94]:


# Pickle extraction
# pickle looks like -> pickle_lyrics['lyrics'][('song_title', 'artist')]['lyrics']
# or - > pickle_lyrics['genre']
pickle_lyrics = []
genre_index = {}
max_length = 0
for i,l_path in enumerate(LYRIC_PATHS):
    if not os.path.exists(PICKLE_ROOT+l_path):
        print('problem occured looking for %s' %(PICKLE_ROOT+l_path))
        sys.exit()
    print(os.getcwd()+PICKLE_ROOT+l_path)
    loaded_lyrics = pickle.load(open(PICKLE_ROOT+l_path, "rb" ))
    genre_index[loaded_lyrics['genre']] = i
    pickle_lyrics.append(loaded_lyrics)
    print(len(loaded_lyrics['lyrics']))
    for key, song_info in loaded_lyrics['lyrics'].items():
        if len(song_info['lyrics'].split()) > max_length:
            max_length = len(song_info['lyrics'].split())
#             print(key)
#             print(max_length)
#             print(i)
print(len(pickle_lyrics))
print(genre_index)
# print(max_length)
# print(pickle_lyrics[0]['lyrics']['Cabin Essence: Chorus', 'The Beach Boys']['lyrics'])


# In[95]:


def check_validity(data):
    valid_count = 0
    max_len_key = ''
    max_len = 0
    total_words = []
    for key, song_info in data['lyrics'].items():
        title, artist = key
        inner_title = song_info['title']
        inner_artist = song_info['artist']
        song_lyrics = song_info['lyrics']
        song_lyrics_norm = re.sub(r'[^a-zA-Z0-9-\']', ' ', song_lyrics).strip()
        song_lyrics_split = song_lyrics_norm.split()         
        if title == inner_title and artist == inner_artist and len(song_lyrics_split) <= MAX_SONG_LENGTH:
            if len(song_lyrics_split) > max_len:
                max_len = len(song_lyrics_split)
                max_len_key = key
            valid_count+=1
            total_words = list(set(total_words+song_lyrics_split))
    print(max_len_key)
    print(max_len)
    return valid_count, total_words

for data in pickle_lyrics:
    print(data['genre'])
    total_songs = len(data['lyrics'])
    total_words_set = []
    valid, total_words = check_validity(data)
    total_words_set  = list(set(total_words_set+total_words))
    print(total_songs, ' : ', valid)
print(len(total_words_set))


# In[98]:


def clean_data(data):
    song_list = []
    for key, song_info in data['lyrics'].items():
        title, artist = key
        inner_title = song_info['title']
        inner_artist = song_info['artist']
        song_lyrics = song_info['lyrics']
        song_lyrics_norm = re.sub(r'[^a-zA-Z0-9-\']', ' ', song_lyrics).strip()
        song_lyrics_split = song_lyrics_norm.split()         
        if title == inner_title and artist == inner_artist and len(song_lyrics_split) <= MAX_SONG_LENGTH:       
            song_list.append(song_lyrics_norm)
            
    return song_list
# initial data pre-processing
# assuming a list of tokenized data 
# vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_document_len)

lyrics = []
lyrics_labels = []
for data in pickle_lyrics:
    genre = data['genre']
#     for key, song_info in data['lyrics'].items():
#         song_lyrics = song_info['lyrics']
#         song_lyrics_norm = re.sub(r'[^a-zA-Z0-9-\']', ' ', song_lyrics).strip()
#         song_lyrics_split = song_lyrics_norm.split() 
#         print(song_lyrics)
#         print()
#         print(song_lyrics_norm)
#         print()
#         print(song_lyrics_split)
    song_list = clean_data(data)
    
    song_labels = [genre_index[genre]]*len(song_list)
    
    lyrics = lyrics + song_list
    lyrics_labels = lyrics_labels + song_labels
print(len(lyrics))
print(len(lyrics_labels))


# In[118]:


MAX_UNIQUE_WORDS = len(total_words_set)
# data preparing
tokenizer = keras.preprocessing.text.Tokenizer(num_words=MAX_UNIQUE_WORDS)
tokenizer.fit_on_texts(lyrics)
sequences = tokenizer.texts_to_sequences(lyrics)

word_index = tokenizer.word_index
print('Unique words tokens %d' % (len(word_index)))

data = keras.preprocessing.sequence.pad_sequences(sequences, maxlen=MAX_SONG_LENGTH)
labels = keras.utils.to_categorical(np.asarray(lyrics_labels))



# In[126]:


# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

num_test_samples = int(TEST_SPLIT * data.shape[0])
num_validation_samples = int(VALIDATION_SPLIT * (data.shape[0]-num_test_samples))

x_test = data[:num_test_samples]
y_test = labels[:num_test_samples]
x_val = data[num_test_samples:num_test_samples+num_validation_samples]
y_val = labels[num_test_samples:num_test_samples+num_validation_samples]
x_train = data[num_test_samples+num_validation_samples:]
y_train = labels[num_test_samples+num_validation_samples:]

print('data tensor:', data.shape)
print('label tensor:', labels.shape)
print('test tensor:', x_test.shape)
print('validate tensor:', x_val.shape)
print('train tensor:', x_train.shape)
print('valid splits: ', x_test.shape[0]+x_val.shape[0]+x_train.shape[0] == data.shape[0])

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
        
# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = keras.layers.Embedding(unique_words_count,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SONG_LENGTH,
                            trainable=False)


# In[ ]:


print('Training model.')

sequence_input = tf.keras.layers.Input(shape=(MAX_SONG_LENGTH,))
embedded_sequences = embedding_layer(sequence_input)

#Model 1
l_cov1= tf.keras.layers.Conv1D(128, 5, activation='relu')(embedded_sequences)
l_pool1 = tf.keras.layers.MaxPooling1D(5)(l_cov1)
l_drop1= tf.keras.layers.Dropout(0.2)(l_pool1)
l_cov2 = tf.keras.layers.Conv1D(128, 5, activation='relu')(l_drop1)
l_pool2 = tf.keras.layers.MaxPooling1D(5)(l_cov2)
l_drop2 = tf.keras.layers.Dropout(0.2)(l_pool2)
l_cov3 = tf.keras.layers.Conv1D(128, 5, activation='relu')(l_drop2)
l_pool3 = tf.keras.layers.MaxPooling1D(35)(l_cov3)  # global max pooling
l_flat = tf.keras.layers.Flatten()(l_pool3)
l_dense = tf.keras.layers.Dense(128, activation='relu')(l_flat)
preds = tf.keras.layers.Dense(len(genre_index), activation='softmax')(l_dense)

optimizer = tf.keras.optimizers.RMSprop(lr=learning_rate, clipnorm = max_grad_norm)
model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

model_details = model.fit(x_train, y_train,
            epochs=100,
            shuffle=True,
            verbose=1,
            validation_data=(x_val, y_val))

scores = model.evaluate(x_test,y_test, verbose=0)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])


# In[119]:


print(type(labels))
print(len(labels))
print(labels)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)
# print(str(len(lyrics)) +' : '+ str(len(sequences)))
# print(str(len(lyrics[0].split())) +' : '+ str(len(sequences[0])))
# print(sequences[0])
# print(lyrics[0])


# In[ ]:




