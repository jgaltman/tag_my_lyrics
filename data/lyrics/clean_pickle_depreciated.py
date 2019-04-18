import os
import sys
import re
import pickle


CHRISTIAN_PATH = 'Christian.pickle'
POP_PATH = 'Pop.pickle'
ROCK_PATH = 'Rock.pickle'
COUNTRY_PATH = 'Country.pickle'
RAP_PATH = 'Rap.pickle'

LYRIC_PATHS = [CHRISTIAN_PATH,POP_PATH,ROCK_PATH,COUNTRY_PATH,RAP_PATH]

MAX_SONG_LENGTH = 2500

def check_validity(data):
    valid_count = 0
    new_data = {}
    for key, song_info in data.items():
        title, artist = key
        inner_title = song_info['title']
        inner_artist = song_info['artist']
        song_lyrics = song_info['lyrics']
        song_lyrics_norm = re.sub(r'[^a-zA-Z0-9-\']', ' ', song_lyrics).strip()
        song_lyrics_split = song_lyrics_norm.split()         
        if title == inner_title and artist == inner_artist and len(song_lyrics_split) <= MAX_SONG_LENGTH:
            valid_count+=1
            new_data[key] = {'title' : title, 'artist' : artist, 'lyrics' : song_lyrics}
    return new_data , valid_count


max_length = 0
for i,l_path in enumerate(LYRIC_PATHS):
    if not os.path.exists(l_path):
        print('problem occured looking for %s' %(+l_path))
        sys.exit()
    print(os.getcwd()+'/'+l_path)
    loaded_lyrics = pickle.load(open(l_path, "rb" ))
    print('genre: %s' % (loaded_lyrics['genre']))
    loaded_lyrics['lyrics'], valid = check_validity(loaded_lyrics['lyrics'])
    print('Found %d valid songs with lyrics' %(valid))
    pickle.dump( loaded_lyrics, open( l_path, "wb" ) )

    new_lyrics = pickle.load(open(l_path, "rb" ))

    print('saved new pickle: %s' %(len(new_lyrics['lyrics'])==len(loaded_lyrics['lyrics'])))


print('finished')