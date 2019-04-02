import lyricsgenius
import sys
import re
import pickle

client_access_token = 'hm9nFzpsVkkj12dXzta2-DIR7_fOmYDV2UutS2a1tX7KZK_bVcrhh8Lr-9cj9o5M'
genius = lyricsgenius.Genius(client_access_token)
AUTO = True

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


def fetchSong(song_info, attempt = 0):
  try:
    #print('  not in pickle, let\'s see if we can find it...')
    return genius.search_song(song_info[0], song_info[1])

  except KeyboardInterrupt:
    print('KeyboardInterrupt detected.\ntype \'a\' to try again, \'s\' to skip this song and carry on, or send a keyboard interrupt (^C) to quit.')

    try:
      user_input = input()
    except:
      raise

    if user_input == 'a':
      return fetchSong(song_info)
    else:
      return

  except:
    if AUTO:
      if attempt < 5:
        print('trying again...')
        return fetchSong(song_info, attempt = attempt + 1)
      else:
        with open('error.txt', 'w+') as errorfile:
          errorfile.write('error on this song:')
          errorfile.write('  {}'.format(song_info))
        return

    else:

      print('something has gone wrong :(\ntype \'a\' to try again, \'s\' to skip this song and carry on, or send a keyboard interrupt (^C) to quit.')

      try:
        user_input = input()
      except:
        raise

      if user_input == 'a':
        return fetchSong(song_info)
      else:
        return



def main(genre, path_in, path_out, append):

  # TODO: keep track of songs not found on genius

  if append:
    with open(path_out, 'rb') as pickle_file:
      lyric_data = pickle.load(pickle_file)

  else:
    lyric_data = {}
    lyric_data['lyrics'] = {}

  lyric_data['genre'] = genre
  number_tried = 0
  number_found = 0

  file_in = open(path_in, 'r')
  content = docConvert(file_in)

  print(('so far, you\'ve collected {} songs. '
         'Total size of source songs file is {} songs.').format(
         len(lyric_data['lyrics']), len(content)))

  for song_info in content:
    #print(song_info)
    if song_info not in lyric_data['lyrics']:
      song = fetchSong(song_info)
      number_tried += 1
      if song:
        number_found += 1
        lyrics = song.lyrics
        lyrics = re.sub('\\[.*?\\]', '', lyrics)
        lyric_data['lyrics'][song_info] = {"title":song.title,
        "artist":song.artist,"lyrics":lyrics}
        print(('\nGENRE: {g}\n{f}/{t} found in this session, '
          '{e}/{o} found overall.').format(
          g = genre, f = number_found, t = number_tried,
          e = len(lyric_data['lyrics']), o = len(content)))

      if(number_found % 25 == 0):
        # save every 25 songs just in case
        with open(path_out, 'wb') as pickle_file:
          pickle.dump(lyric_data, pickle_file)

    else:
      print('already in pickle!')


  print('done fetching lyrics, pickling...')
  with open(path_out, 'wb') as pickle_file:
    pickle.dump(lyric_data, pickle_file)
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