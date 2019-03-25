import requests
from bs4 import BeautifulSoup

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
if __name__ == '__main__':
    input_file = open('ENTERFILEHERE',"r")
    content = docConvert(input_file)
    for item in content:
        song_title = item[0]
        artist_name = item[1]
        search_url = base_url + '/search'
        data = {'q': song_title}
        response = requests.get(search_url, params=data, headers=headers)
        song_info = None
        json = response.json()

        for hit in json['response']['hits']:
            if hit['result']['primary_artist']['name'] == artist_name:
                song_info = hit
        if song_info:
            song_api_path = song_info['result']['api_path']
            print(song_info['result']['full_title'])
            print(lyrics_from_song_api_path(song_api_path))
        else:
            print(artist_name + ' - ' + song_title + ': Not found')


#client id OtbFtWmkQc3Vf4R9oOKTRhSqrgtfvFOuODDhROhvJf3onFuiHe9hSHwmeAsSCO52
# secret O8eDgg2tuyorMR0PB0S0GFyWoo7OMa1zFhWjfpTSCxE0Lz0ilIqFgerKaoiGKIKG_gJrNHAU_i01cAWPwF1LXg
#access token u0cAPtMZ8V-fhGqs2OGbEWyEe5HsrLb1BL1tMX-j6W6NYowtlq0_d5aWPUnvSYC0
