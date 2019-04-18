#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import re
import pickle
from collections import Counter
from pprint import pprint
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
nltk.download('stopwords')
nltk.download('punkt')
tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('english'))


# In[2]:


#CONSTANTS
MAX_SONG_LENGTH = 2500
SONG_PER_GENRE = 4500
THRESH = 2000
REMOVE_COMMON_WORDS = False
DATA_KEYS = ['lyrics','lyrics_labels','unique_words_set','longest_song','genre_index','artist','song_titles']
SAVE = True


# In[3]:


# PATH CONSTANTS
PICKLE_ROOT = '../data/lyrics/original_pickles/'
PICKLE_OUT_PATH = '../data/lyrics/'
CHRISTIAN_PATH = 'Christian.pickle'
POP_PATH = 'Pop.pickle'
ROCK_PATH = 'Rock.pickle'
COUNTRY_PATH = 'Country.pickle'
RAP_PATH = 'Rap.pickle'
OUT_PICKLE = 'CNN_input.pickle' 

LYRIC_PATHS = [CHRISTIAN_PATH,POP_PATH,ROCK_PATH,COUNTRY_PATH,RAP_PATH]


# In[4]:


# Pickle extraction
# pickle looks like -> pickle_lyrics['lyrics'][('song_title', 'artist')]['lyrics']
# or - > pickle_lyrics['genre']
def load_lyrics():
    lyrics_list = []
    genres_map = {}
    for i,l_path in enumerate(LYRIC_PATHS):
        if not os.path.exists(PICKLE_ROOT+l_path):
            print('problem occured looking for %s' %(PICKLE_ROOT+l_path))
            sys.exit()
        print('loading %s%s%s' %(os.getcwd(),PICKLE_ROOT,l_path))
        loaded_lyrics = pickle.load(open(PICKLE_ROOT+l_path, "rb" ))
        genres_map[loaded_lyrics['genre']] = i
        lyrics_list.append(loaded_lyrics)
        print('number of songs: %d' %(len(loaded_lyrics['lyrics'])))
    return lyrics_list, genres_map


# In[5]:


def remove_stop_and_punct(article):
    final_article = []
    word_tokens = tokenizer.tokenize(article)
    filtered_article = [w for w in word_tokens if not w in stop_words]
    filtered_article = [x.lower() for x in filtered_article]
    filtered_song = ' '.join(filtered_article)
    return filtered_song, filtered_article

def remove_similar_word(word_list,word_count):
    new_word_list = []
    for word in word_list:
        keep = False
        for key,count in word_count[word].items():
            if count < THRESH:
                keep = True
        if keep:
            new_word_list.append(word)
    return new_word_list
            
        
def clean_genre(data,word_count):
    song_list = []
    unique_words_list = []
    count = 0
    max_length_song = 0
    for key, song_info in data['lyrics'].items():
        title, artist = key
        inner_title = song_info['title']
        if count%1000==0:
            print('iter - %d: song - %s' %(count, inner_title))
            
        inner_artist = song_info['artist']
        song_lyrics = song_info['lyrics']
        song_lyrics_norm, song_lyrics_split = remove_stop_and_punct(song_lyrics)
        
        
        
        if title == inner_title and artist == inner_artist:
            if len(song_lyrics_split) <= MAX_SONG_LENGTH:
                if REMOVE_COMMON_WORDS:
                    song_lyrics_split = remove_similar_word(song_lyrics_split,word_count)
                    song_lyrics_norm = ' '.join(song_lyrics_split)
                if len(song_lyrics_split) > max_length_song:
                    max_length_song = len(song_lyrics_split)
                song_list.append(song_lyrics_norm)
                unique_words_list = list(set(unique_words_list + song_lyrics_split))
            count+=1
        
        if count >= SONG_PER_GENRE:
            print('hit max songs: %d' %(SONG_PER_GENRE))
            print('songs left out: %d' %(len(data['lyrics'])-SONG_PER_GENRE))
            print('longest song length: %d' %(max_length_song))
            return song_list, unique_words_list, max_length_song
       
    return song_list, unique_words_list, max_length_song

def clean_data(p_lyrics,genre_index,word_count):
    lyrics = []
    lyrics_labels = []
    unique_words_set = []
    longest_song = 0
    for data in p_lyrics:
        genre = data['genre']
        print('cleaning: %s' %(genre))
        song_list, unique_words, g_longest_song = clean_genre(data,word_count)
        unique_words_set = list(set(unique_words_set+unique_words))
        song_labels = [genre_index[genre]]*len(song_list)
        if longest_song < g_longest_song:
            longest_song = g_longest_song
        lyrics = lyrics + song_list
        lyrics_labels = lyrics_labels + song_labels
    return [lyrics, lyrics_labels, unique_words_set, longest_song, genre_index]


# In[6]:


def save_prepared_data(filepath,filename, list_data):
    data = {}
    print('keys match data: %s' %(len(list_data)==len(DATA_KEYS)))
    for key,val in zip(DATA_KEYS,list_data):
        data[key] = val
    pickle.dump( data, open(filepath+filename, "wb" ) )
    print('saved data to: %s%s' %(filepath,filename))
    
def verify_data(filepath,filename, list_data):
    loaded_data = pickle.load( open(filepath + filename , "rb" ) )
    print('keys match loaded data: %s' %(len(loaded_data)==len(list_data)))
    for key,val in zip(DATA_KEYS,list_data):
        try:
            data_check = (len(loaded_data[key])==len(val))
        except:
            data_check = (loaded_data[key]==val)
        if data_check == False:
            print('ERROR: Data saved does not match at:')
            print(key)
            print('please rerun and check paths')
            return False
    return True


# In[7]:


def count_words_per_genre(p_lyrics):
    word_count = {}
    for data in p_lyrics:
        genre = data['genre']
        for key, song_info in data['lyrics'].items():
            song_lyrics = song_info['lyrics']
            song_lyrics_norm, song_lyrics_split = remove_stop_and_punct(song_lyrics)
            song_word_count = Counter(song_lyrics_split)
            for word, count in song_word_count.items():
                if word in word_count:
                    if genre in word_count[word]:
                        word_count[word][genre]+=count
                    else:
                        word_count[word][genre] = count
                else:
                    word_count[word] = {}
                    word_count[word][genre] = count
    return word_count


# In[8]:


def find_most_occurences(word_count):
    common_words = []
    for word,genre in word_count.items():
        keep = False
        for genre_key, count in word_count[word].items():
            if count < THRESH:
                keep = True
        if not keep:
            common_words.append(word)
    return common_words


# In[9]:


def main(argv):
    pickle_lyrics = []
    genre_index = {}
    word_count = {}
    print('loading pickles')
    pickle_lyrics, genre_index = load_lyrics()
    print('genres dict')
    print(genre_index)
    print('finished loading pickles')
    
    print('counting words')
    word_count = count_words_per_genre(pickle_lyrics)
    print('finished counting')
    
    print('cleaning data')
    new_data = clean_data(pickle_lyrics,genre_index,word_count)
    lyrics = new_data[0]
    lyrics_labels = new_data[1]
    unique_words_set = new_data[2]
    longest_song = new_data[3]
    print('\n')
    print('number of songs: %d' %(len(lyrics)))
    print('number of lyrics: %d' %(len(lyrics_labels)))
    print('number of unique words: %d' %(len(unique_words_set)))
    print('number of genres: %d' %(len(genre_index)))
    print('longest song : %d' %(longest_song))
    print('finished cleaning data')
    
    if SAVE:
        
        save_prepared_data(PICKLE_OUT_PATH,OUT_PICKLE,new_data)
        print('verifying data stored')
        verified = verify_data(PICKLE_OUT_PATH,OUT_PICKLE,new_data)
        print('All data verified : %s' %(verified))

if __name__ == '__main__':
    main(sys.argv)    


# In[ ]:




