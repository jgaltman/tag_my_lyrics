# tag_my_lyrics
tag_my_lyrics is a system for genre classification based on song lyrics. It is written in Python 3.x, and it uses a CNN architecture.

### Data collection:
  Two methods for obtaining lyrical data:
      Scrape billboard charts for each genre (this is assuming that the billboard genre is the correct genre)
      List of artists that are known for particular genres that we had trouble getting data for (Pop, Rap)
  Lyrics are then found for each song via lyricwikia python library
  Formatted into json files and pickled.  Each genre has a separate pickle

### Word Embeddings:
  We use pre trained word embeddings pulled from Stanford's GloVe at:
    http://nlp.stanford.edu/data/glove.6B.zip
   
### Preprocessing:
  Songs are:
    Stripped of stop words
    Lowercased
    Formatted using regex and nltk
  Each word is indexed and tokenized, songs are padded with zeros until they match the longest song available rounded to the nearest    hundred (in our case this is 1300)
 After preprocessing, there is one input file, CNN_input.pickle
  
 
### Model:
  Our model only takes into account the first 20k unique words from each song as a list of tokenized lyrics
  Steps: Song -> Embedding Layer -> Conv -> Max pooling -> concatanate -> dropout -> flatten -> dropout -> dense layer -> output

## Running:
### Data collection:

### CNN 
  To run this model correctly after collecting the data follow the order below. Initially run preprocessing to format the data properly. This will save CNN_input.pickle file in data/lyrics. Then Run the CNN model. This will train and test the model with the data. This will save numerous files, most important of which are:
    1. The model files
      - best_weights_2.0.hdf5
      - cnn_model_2.0.h5
      - cnn_model_2.0.json
    2. The UI pickles 
      - test/genres.pickle
      - test/token.pickle
  Lastly run the UI to have a fully functional genre classifier that you can implement. Below is a list of these instructions.
  1. CNN_preprocessing_1.0.py or CNN_preprocessing_1.0.ipynb
  2. Lyric_CNN_2.0.py or Lyric_CNN_2.0.ipynb
  3. UIScript.py

### File Structure
Many Data Files are left out of this repository.
The file tree should initially look as below to run properly.
├── Lyric_CNN_2.0.ipynb
├── Lyric_CNN_2.0.py
├── UIScript.py
├── data
│   ├── glove_embeddings
│   │   └── glove.6B.100d.txt
│   ├── lyrics
│   │   └── original_pickles
│   │       ├── Christian.pickle
│   │       ├── Country.pickle
│   │       ├── Pop.pickle
│   │       ├── Rap.pickle
│   │       └── Rock.pickle
│   └── test
├── graphs_out
├── preprocessing
│   ├── CNN_preprocessing_1.0.ipynb
│   └── CNN_preprocessing_1.0.py
└── saved_models

After running Preprocessing and CNN:

├── Lyric_CNN_2.0.ipynb
├── Lyric_CNN_2.0.py
├── UIScript.py
├── data
│   ├── glove_embeddings
│   │   └── glove.6B.100d.txt
│   ├── lyrics
│   │   ├── CNN_input.pickle
│   │   └── original_pickles
│   │       ├── Christian.pickle
│   │       ├── Country.pickle
│   │       ├── Pop.pickle
│   │       ├── Rap.pickle
│   │       └── Rock.pickle
│   └── test
│       ├── genres.pickle
│       ├── recent_testdata_2.0.pickle
│       └── token.pickle
├── graphs_out
│   ├── accuracy_2.0.png
│   ├── confusion_matrix_2.0.png
│   └── loss_2.0.png
├── preprocessing
│   ├── CNN_preprocessing_1.0.ipynb
│   └── CNN_preprocessing_1.0.py
└── saved_models
    ├── best_weights_2.0.hdf5
    ├── cnn_model_2.0.h5
    └── cnn_model_2.0.json

