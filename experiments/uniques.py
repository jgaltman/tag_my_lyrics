import csv


christianDict = {}
countryDict = {}
popDict = {}
rapDict = {}
rockDict = {}

with open('.\\csvs\\rock.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
    	rockDict[row[0]] = int(row[1])

with open('.\\csvs\\pop.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
    	popDict[row[0]] = int(row[1])

with open('.\\csvs\\christian.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
    	christianDict[row[0]] = int(row[1])

with open('.\\csvs\\rap.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
    	rapDict[row[0]] = int(row[1])

with open('.\\csvs\\country.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
    	countryDict[row[0]] = int(row[1])

uniques = {}
genre = 'rap'

unique = rapDict

for word in unique:
	if unique[word] > 5000:
		if word not in rockDict or rockDict[word] < 10:
			if word not in countryDict or countryDict[word] < 10:
				if word not in christianDict or christianDict[word] < 10:
					if word not in popDict or popDict[word] < 10:
						uniques[word] = unique[word]


print(uniques)

with open('.\\csvs\\' + genre + 'Uniques.csv', 'w') as f:
    for key in uniques.keys():
        try:
            f.write("%s,%s\n"%(key,uniques[key]))
        except:
            print(key)