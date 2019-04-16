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

allwords = set([])

allwords = allwords | set(rockDict.keys()) | set(countryDict.keys()) | set(popDict.keys()) | set(rapDict.keys()) | set(christianDict.keys())

print(len(allwords))
thresh = 100

common = []

for word in allwords:
    if word in rockDict and rockDict[word] > thresh:
            if word in countryDict and countryDict[word] > thresh:
                if word in christianDict and christianDict[word] > thresh:
                    if word in popDict and popDict[word] > thresh:
                        if word in rapDict and rapDict[word] > thresh:
                            common.append(word)


print(len(common))
