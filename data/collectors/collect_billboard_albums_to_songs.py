import billboard
from PyLyrics import *

def docReader(file):
	l = file.readlines()
	return l

def docConvert(x):
	song_set = set([])
	l = docReader(x)
	for song in l:
		s = song.split("|-+-|")
		song_set.add((s[0],s[1]))

	return song_set

def song2string(tracks):
	final = []
	for t in tracks:
		final.append(t.name)
	return final

f = open("ChristianAlbumSongs.txt", "a")
song_set = {}
year = 2019
month = 3
days = ["01","14","28"]


while (year >1958 and len(song_set) < 20000):

	if month < 10:
		m = "0%s" % (month)
	else:
		m = month
	print(month, year)
	print(len(song_set))
	charts = ["christian-albums"]
	for day in days:
		date = "%s-%s-%s" % (year, m, day)
		for c in charts:
			try:
				chart = billboard.ChartData(c, date=date, fetch=True, timeout=30)
			except:
				print("Can't Find Chart")
			else:
				for album in chart:
					title = album.title
					try:
						albums = PyLyrics.getAlbums(singer=album.artist)
					except:
						print("Can't Find Artist/Album")
					else:
						for myalbum in albums:
							try:
								temp = myalbum.tracks() 
								tracks = song2string(temp)
							except:
								print("Can't find tracks")
							else:
								if title in tracks:
									for track in tracks:
										song_set[(track, album.artist)] = str(year)+ "|-+-|"+ c				
	month = month - 1
	if month == 0:
		month = 12
		year = year - 1


for x in song_set.keys():
	f.write(x[0] + "|-+-|" + x[1] + "|-+-|" + song_set[x] + "\n")