import lyricwikia
import billboard

def docReader(file):
	l = file.readlines()
	return l

def docConvert(x):
	song_set = set([])
	l = docReader(q)
	for song in l:
		s = song.split("|-+-|")
		song_set.add((s[0],s[1].strip()))

	return song_set

##Set file to add songs to
f = open("CountrySongs.txt", "a")

##Optional to read in previous file set to add to
###q = open("demofile.txt", "r")
###song_set = docConvert(q)
song_set = {}

year = 2019
month = 3
days = ["01","14","28"]


while (year >1958): ##1958 is furthest back billboard.com goes

	if month < 10:
		m = "0%s" % (month)
	else:
		m = month

	##Tracks year/month the data is being read on the console to view progress	
	print(month, year)
	print(len(song_set))

	##add charts you want read
	charts = ["country-digital-song-sales", "country-songs", "country-streaming-songs"]
	for day in days:
		date = "%s-%s-%s" % (year, m, day)
		for c in charts:
			try:
				chart = billboard.ChartData(c, date=date, fetch=True, timeout=30)
			except:
				print("Can't Find Chart")
			else:
				for song in chart:
					song_set[(song.title, song.artist)] = str(year)+ "|-+-|"+ c
				
	month = month - 1
	if month == 0:
		month = 12
		year = year - 1

print(len(song_set))

##adds to final document
for x in song_set.keys():
	f.write(x[0] + "|-+-|" + x[1] + "|-+-|" + song_set[x] + "\n")