import lyricsgenius
import sys
import re
import pickle

client_access_token = 'hm9nFzpsVkkj12dXzta2-DIR7_fOmYDV2UutS2a1tX7KZK_bVcrhh8Lr-9cj9o5M'
genius = lyricsgenius.Genius(client_access_token)

def docReader(file):
  l = file.readlines()
  return l


def docConvert(q):
  song_set = set([])
  l = docReader(q)
  for song in l:
    s = song.split("|-+-|")
    song_set.add((s[0],s[1].strip()))
  return song_set


def main(genre, path_in, path_out, append):
  if append:
    with open(path_out, 'rb') as pickle_file:
      lyricData = pickle.load(pickle_file)
  else:
    lyricData = {}
    lyricData['genre'] = genre
    lyricData['lyrics'] = {}

  number_tried = 0
  number_found = 0

  file_in = open(path_in, 'r')
  content = docConvert(file_in)

  print('found/attempted')

  for song_info in content:
    #print(song_info)
    if song_info not in lyricData['lyrics']:
      #print('  not in pickle, let\'s see if we can find it...')

      song = genius.search_song(song_info[0], song_info[1])
      number_tried += 1

      if song:
        #print('    found it!')
        number_found += 1

        lyrics = song.lyrics
        lyrics = re.sub('\\[.*?\\]', '', lyrics)
        lyricData['lyrics'][song_info] = {"title":song.title,"artist":song.artist,"lyrics":lyrics}

        print('{}/{}'.format(number_found,number_tried))

      #else:
        #print('    not found :(')

      if(number_found % 25 == 0):
        #print('pickling to save progress...')
        with open(path_out, 'wb') as pickle_file:
          pickle.dump(lyricData, pickle_file)
        #print('done pickling')

  print('done fetching lyrics, pickling...')
  with open(path_out, 'wb') as pickle_file:
    pickle.dump(lyricData, pickle_file)
  print('done!')


if __name__ == '__main__':
  if len(sys.argv) < 4:
    print('Usage: geniusScraper.py [genre name for object] [path to GenreSongs.txt] [path to fileout.pickle] [flags (optional)]')
    print('  flags include:')
    print('  -a / --append      open [fileout.pickle] and append to it')
  else:
    genre = sys.argv[1]
    filein = sys.argv[2]
    fileout = sys.argv[3]
    append = False
    if len(sys.argv) > 4:
      for arg in sys.argv[4:]:
        if arg == '-a' or arg == '--append':
          append = True
        else:
          print('ignoring unknown argument "{}"'.format(arg))
    main(genre, filein, fileout, append)