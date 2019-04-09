#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import re
import pickle
# import nltk
import numpy as np
from pprint import pprint
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras import backend
from tensorflow.python.client import device_lib
from tensorflow.keras.models import model_from_json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# nltk.download('stopwords')
# nltk.download('punkt')

device = device_lib.list_local_devices()
print(device)
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

backend.set_session(sess)


# In[12]:


#CONSTANTS
MAX_SONG_LENGTH = 2500
# MAX_NUM_WORDS = 20000
VALIDATION_SPLIT = 0.3
TEST_SPLIT = 0.2
SONG_PER_GENRE = 4500
learning_rate = .0000001
max_grad_norm = 1.
dropout = 0.5
EMBEDDING_DIM = 200


# In[3]:


# PATH CONSTANTS
PICKLE_ROOT = 'data/lyrics/'
CHRISTIAN_PATH = 'Christian.pickle'
POP_PATH = 'Pop.pickle'
ROCK_PATH = 'Rock.pickle'
COUNTRY_PATH = 'Country.pickle'
RAP_PATH = 'Rap.pickle'

LYRIC_PATHS = [CHRISTIAN_PATH,POP_PATH,ROCK_PATH,COUNTRY_PATH,RAP_PATH]

EMBEDDING_PATH = 'data/glove_embeddings/'
EMBEDDING_FILE = 'glove.6B.'+str(EMBEDDING_DIM)+'d.txt'


# In[4]:


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


# In[5]:


# Pickle extraction
# pickle looks like -> pickle_lyrics['lyrics'][('song_title', 'artist')]['lyrics']
# or - > pickle_lyrics['genre']
print('loading pickles')

pickle_lyrics = []
genre_index = {}
for i,l_path in enumerate(LYRIC_PATHS):
    if not os.path.exists(PICKLE_ROOT+l_path):
        print('problem occured looking for %s' %(PICKLE_ROOT+l_path))
        sys.exit()
    print(os.getcwd()+PICKLE_ROOT+l_path)
    loaded_lyrics = pickle.load(open(PICKLE_ROOT+l_path, "rb" ))
    genre_index[loaded_lyrics['genre']] = i
    pickle_lyrics.append(loaded_lyrics)
    print(len(loaded_lyrics['lyrics']))

print(len(pickle_lyrics))
print(genre_index)
print('finished loading pickles')


# In[28]:


def clean_data(data):
    song_list = []
    unique_words_list = []
    count = 0
    for key, song_info in data['lyrics'].items():
        title, artist = key
        inner_title = song_info['title']
        if count%1000==0:
            print('%d: %s' %(count, inner_title))
        inner_artist = song_info['artist']
        song_lyrics = song_info['lyrics']
        song_lyrics_norm = re.sub(r'[^a-zA-Z0-9-\']', ' ', song_lyrics).strip()
        song_lyrics_split = song_lyrics_norm.lower().split()        
        if len(song_lyrics_split) <= MAX_SONG_LENGTH:       
            song_list.append(song_lyrics_norm)
            unique_words_list = list(set(unique_words_list + song_lyrics_split))
        count+=1
        if count >= SONG_PER_GENRE:
            print('hit max songs: %d' %(SONG_PER_GENRE))
            print('songs left out: %d' %(len(data['lyrics'])-SONG_PER_GENRE))
            return song_list, unique_words_list
       
    return song_list, unique_words_list
# initial data pre-processing
# assuming a list of tokenized data 
# vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_document_len)
print('cleaning data')
lyrics = []
lyrics_labels = []
unique_words_set = []
for data in pickle_lyrics:
    genre = data['genre']
    print('cleaning: %s' %(genre))
    song_list, unique_words = clean_data(data)
    unique_words_set = list(set(unique_words_set+unique_words))
    song_labels = [genre_index[genre]]*len(song_list)
    
    lyrics = lyrics + song_list
    lyrics_labels = lyrics_labels + song_labels
print('\n\n')
print('number of songs: %d' %(len(lyrics)))
print('number of lyrics: %d' %(len(lyrics_labels)))
print('number of unique words: %d' %(len(unique_words_set)))
print('finished cleaning data')


# In[29]:


# MAX_UNIQUE_WORDS = len(unique_words_set)
MAX_UNIQUE_WORDS = 20000
# data preparing
print('tokenizing')
tokenizer = keras.preprocessing.text.Tokenizer(num_words=MAX_UNIQUE_WORDS)
tokenizer.fit_on_texts(lyrics)
sequences = tokenizer.texts_to_sequences(lyrics)

word_index = tokenizer.word_index
print('Unique words tokens %d' % (len(word_index)))

data = keras.preprocessing.sequence.pad_sequences(sequences, maxlen=MAX_SONG_LENGTH)
labels = keras.utils.to_categorical(np.asarray(lyrics_labels))

print('finished tokenizing')


# In[30]:


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
    num_validation_samples = int(VALIDATION_SPLIT * (data.shape[0]-num_test_samples))

    x_test = data[:num_test_samples]
    y_test = labels[:num_test_samples]
    x_val = data[num_test_samples:num_test_samples+num_validation_samples]
    y_val = labels[num_test_samples:num_test_samples+num_validation_samples]
    x_train = data[num_test_samples+num_validation_samples:]
    y_train = labels[num_test_samples+num_validation_samples:]
    
else:
    num_test_samples = int(TEST_SPLIT * t_data.shape[0])
    num_validation_samples = int(VALIDATION_SPLIT * (t_data.shape[0]-num_test_samples))

    x_test = t_data[:num_test_samples]
    y_test = t_labels[:num_test_samples]
    x_val = t_data[num_test_samples:num_test_samples+num_validation_samples]
    y_val = t_labels[num_test_samples:num_test_samples+num_validation_samples]
    x_train = t_data[num_test_samples+num_validation_samples:]
    y_train = t_labels[num_test_samples+num_validation_samples:]
    
print('data tensor:', data.shape)
print('test tensor:', x_test.shape)
print('validate tensor:', x_val.shape)
print('train tensor:', x_train.shape)
print('valid splits: ', x_test.shape[0]+x_val.shape[0]+x_train.shape[0] == data.shape[0])
print('label tensor:', labels.shape)
print('test tensor:', y_test.shape)
print('validate tensor:', y_val.shape)
print('train tensor:', y_train.shape)
print('valid splits: ', y_test.shape[0]+y_val.shape[0]+y_train.shape[0] == data.shape[0])

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


# In[31]:


# save model
def save_model(filename,weights_filename):
    # serialize model to JSON
    model_json = model.to_json()
    with open(filename, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(weights_filename)
    print("Saved model to disk")


# In[32]:



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


# In[33]:


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


# In[ ]:




