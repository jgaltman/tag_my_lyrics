import nltk
import textblob
import lyricsgenius
import sys
import pull_songs_from_charts.py


def main(argv):
  print(argv)
  print('collecting from genius API...')
  client_access_token = 'hm9nFzpsVkkj12dXzta2-DIR7_fOmYDV2UutS2a1tX7KZK_bVcrhh8Lr-9cj9o5M'
  genius = lyricsgenius.Genius(client_access_token)
  artist = genius.search_artist("Andy Shauf", max_songs=3, sort="title")
  print(artist.songs)

if __name__ == '__main__':
  main(sys.argv)

