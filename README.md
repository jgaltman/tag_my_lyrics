# tag_my_lyrics
tag_my_lyrics is a system for genre classification based on song lyrics. It is written in Python 3.x, and it uses a CNN architecture.

Data collection:
  Two methods for obtaining lyrical data:
      Scrape billboard charts for each genre (this is assuming that the billboard genre is the correct genre)
      List of artists that are known for particular genres that we had trouble getting data for (Pop, Rap)
  Lyrics are then found for each song via lyricwikia python library
  Formatted into json files and pickled.  Each genre has a separate pickle
  
Word Embeddings:
   We use pre trained word embeddings pulled from Stanford's GloVe.
   
Preprocessing:
  Songs are:
    Stripped of stop words
    Lowercased
    Formatted using regex and nltk
 Each word is indexed and tokenized, songs are padded with zeros until they match the longest song available rounded to the nearest    hundred (in our case this is 1300)
 After preprocessing, there is one input file, CNN_input.pickle
  
 
Model:
  Our model only takes into account the first 20k unique words from each song as a list of tokenized lyrics
  Steps: Song -> Embedding Layer -> Conv -> Max pooling -> concatanate -> dropout -> flatten -> dropout -> dense layer -> output

