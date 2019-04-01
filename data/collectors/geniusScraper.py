import requests
from bs4 import BeautifulSoup
import sys
import re
import pickle

base_url = 'http://api.genius.com'
headers = {'Authorization': 'Bearer u0cAPtMZ8V-fhGqs2OGbEWyEe5HsrLb1BL1tMX-j6W6NYowtlq0_d5aWPUnvSYC0'}


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


def lyrics_from_song_api_path(song_api_path):
  song_url = base_url + song_api_path
  response = requests.get(song_url, headers=headers)
  json = response.json()
  path = json['response']['song']['path']

  # Regular HTML scraping
  page_url = 'http://genius.com' + path
  page = requests.get(page_url)
  html = BeautifulSoup(page.text, 'html.parser')
  lyrics = html.find(class_='lyrics').get_text()
  return lyrics


def main(genre, path_in, path_out):
  lyricData = {}
  lyricData['genre'] = genre
  lyricData['lyrics'] = {}

  id = 0
  number_tried = 0
  number_found = 0

  file_in = open(path_in, 'r')
  content = docConvert(file_in)

  print('found/attempted')

  for item in content:
    song_title = item[0]
    artist_name = item[1]
    search_url = base_url + '/search'
    data = {'q': song_title}
    response = requests.get(search_url, params=data, headers=headers)
    song_info = None
    json = response.json()
    number_tried += 1

    for hit in json['response']['hits']:
      if hit['result']['primary_artist']['name'] == artist_name:
        song_info = hit

    if song_info:
      number_found += 1
      song_api_path = song_info['result']['api_path']
      #print(song_info['result']['full_title'],": Found!")
      my_string = (lyrics_from_song_api_path(song_api_path))
      my_string = re.sub("\[.*?\]", "", my_string)
      lyricData['lyrics'][id] = {"title":song_title,"artist":artist_name,"lyrics":my_string}
      id+=1
      print('id: {}'.format(id))

      print('{}/{}'.format(number_found,number_tried))

      # if(id%100 == 0):
      #   with open('ChristianSongsLyrics.pickle', 'wb') as handle:
      #     pickle.dump(ChristianSongs, handle, protocol=pickle.HIGHEST_PROTOCOL)
      #   with open('ChristianSongsLyrics.pickle', 'rb') as handle:
      #     b = pickle.load(handle)

    with open(path_out, 'wb') as pickle_file:
      pickle.dump(lyricData, pickle_file)

  print('done!')


if __name__ == '__main__':
  if len(sys.argv) < 4:
    print('Usage: geniusScraper.py [genre name for object] [path to GenreSongs.txt] [path to fileout.pickle]')
  else:
    genre = sys.argv[1]
    filein = sys.argv[2]
    fileout = sys.argv[3]
    main(genre, filein, fileout)


      # if 'chorus' in my_string or 'Chorus' in my_string or 'CHORUS' in my_string:
      #   print(my_string)
    # else:
    #   print(artist_name + ' - ' + song_title + ': Not found')

    # print('current progress: {}/{}, {}%'
    #   .format(number_found, number_tried, 100*number_found/number_tried))



# client id OtbFtWmkQc3Vf4R9oOKTRhSqrgtfvFOuODDhROhvJf3onFuiHe9hSHwmeAsSCO52
# secret O8eDgg2tuyorMR0PB0S0GFyWoo7OMa1zFhWjfpTSCxE0Lz0ilIqFgerKaoiGKIKG_gJrNHAU_i01cAWPwF1LXg
# access token u0cAPtMZ8V-fhGqs2OGbEWyEe5HsrLb1BL1tMX-j6W6NYowtlq0_d5aWPUnvSYC0