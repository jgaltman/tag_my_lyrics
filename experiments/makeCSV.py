import csv
import json
import string
import re
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize 
from contractions import contractions_dict

stop_words = set(stopwords.words('english')) 

def format(lyrics):
    final = []
    new_one = ""
    for word in lyrics.split(" "):
        #re.sub(r'[^a-zA-Z0-9-\']', ' ', word).strip()
        words = word.split('\n')
        for w in words:
            if w in contractions_dict:
                w = contractions_dict[w]
                for x in w.split(" "):
                    new_one += x + " "
            else:
                if w.lower() == "wanna":
                    w = "want to"
                if w.lower() == "'cause":
                    w = "because"

                new_one += w + " "
    tokens = word_tokenize(lyrics)

    """
            if len(w) > 0 and w not in string.punctuation:
                while len(w) > 0 and (w[0] in string.punctuation or w[-1] in string.punctuation):
                    if w[0] in string.punctuation:
                        w = w[1:]
                    elif w[-1] in string.punctuation:
                        w = w[:-1]
                if w.lower() not in stop_words:
                    final.append(w.lower())
            """
    return tokens


genre = 'christian'
with open('.\\jsons\\ChristianLyrics.json') as json_file:  
    data = json.load(json_file)
    print(len(data[genre]))
    word_dict = {}
    for s in data[genre]:
        lyrics = s['lyrics']
        words = format(lyrics)
        for word in words:
            word = word.lower()
            if word == "wo":
                word = "will"
            elif word == "n't":
                word = "not"
            elif word == "'s":
                word = "is"
            elif word == "'m":
                word = "am"
            elif word == "'re":
                word = "are"
            elif word == "'ll":
                word = "will"
            elif word == "'ve":
                word = "have"
            elif word == "ca":
                word = "can"
            elif word == "'cause":
                word = "because"
            elif word == "'d":
                word = "would"
            elif word == "ai":
                word = "are"
            elif word == "wan":
                word = "want"
            elif word == "gon":
                word = "go"
            elif word == "na":
                word = "to"
            else: 
                if word not in string.punctuation and word not in stop_words and not word.isdigit():
                    dont = False
                    if word[0] in string.punctuation:
                        word = word[1:]
                    for x in word:
                        if x in string.punctuation or x.isdigit():
                            dont = True
                            break
                    if not dont:
                        if word not in word_dict.keys():
                            word_dict[word] = 1
                        else:
                            word_dict[word] += 1

with open('.\\csvs\\' + genre + '.csv', 'w') as f:
    for key in word_dict.keys():
        try:
            f.write("%s,%s\n"%(key,word_dict[key]))
        except:
            print(key)