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

f = open("PopSongs.txt", "a")
##q = open("demofile.txt", "r")
##song_set = docConvert(q)
song_set = {}
year = 2019
month = 3
days = ["01","14","28"]


while (year >1957):

	if month < 10:
		m = "0%s" % (month)
	else:
		m = month
	print(month, year)
	print(len(song_set))
	for day in days:
		date = "%s-%s-%s" % (year, m, day)
		try:
			chart = billboard.ChartData('adult-contemporary', date=date, fetch=True, timeout=30)
		except:
			print("Oops")
		else:
			for song in chart:
				song_set[(song.title, song.artist)] = str(year)+ "|-+-|"+"adult-contemporary"
		try:
			chart = billboard.ChartData('pop-songs', date=date, fetch=True, timeout=30)
		except:
			print("Oops")
		else:
			for song in chart:
				song_set[(song.title, song.artist)] = str(year)+ "|-+-|"+"pop-songs"
		try:
			chart = billboard.ChartData('adult-pop-songs', date=date, fetch=True, timeout=30)
		except:
			print("Oops")
		else:
			for song in chart:
				song_set[(song.title, song.artist)] = str(year)+ "|-+-|"+"adult-pop-songs"
	month = month - 1
	if month == 0:
		month = 12
		year = year - 1

print(len(song_set))
for x in song_set.keys():
	f.write(x[0] + "|-+-|" + x[1] + "|-+-|" + song_set[x] + "\n")