# open pickle and do some editing

import sys
import pickle

def main(path_in, path_out):
  with open(path_in, 'rb') as pickle_file:
    lyricData = pickle.load(pickle_file)

  print('genre is {}'.format(lyricData['genre']))

  print(len(lyricData))
  print(len(lyricData['lyrics']))

  for songID in lyricData['lyrics']:
    song = lyricData['lyrics'][songID]
    print('\n\n\n\nsongID: {}, artist: {}, first piece of lyrics: {}'.format(
      songID, song['artist'], song['lyrics'][:100]))

if __name__ == '__main__':
  if len(sys.argv) < 2:
    print('Usage: pickleTransform.py [path to filein.pickle] [path to fileout.pickle]')
  else:
    filein = sys.argv[1]
    fileout = sys.argv[2]
    main(filein, fileout)
