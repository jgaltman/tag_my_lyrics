# CONFIG FILE FOR CONSTANTS

# VERSION
VERSION = '2.0'

# SERVER VARIABLES
CPU=False
EPOCHS = 10

#CONSTANTS
VALIDATION_SPLIT = 0.33
TEST_SPLIT = 0.2
LEARNING_RATE = .001
MAX_NORM_GRAD = 1.
DROPOUT = 0.5
EMBEDDING_DIM = 100
MAX_UNIQUE_WORDS = 20000

# GENERAL PATH CONSTANTS
LYRIC_ROOT = 'data/lyrics/'
CNN_INPUT = LYRIC_ROOT+'CNN_input.pickle' 

EMBEDDING_DIR = 'data/glove_embeddings/'
EMBEDDING_FILE = EMBEDDING_DIR+'glove.6B.'+str(EMBEDDING_DIM)+'d.txt'

MODEL_DIR = 'saved_models/'
MODEL_FILE = MODEL_DIR+'cnn_model_'+VERSION+'.json'
MODEL_WEIGHTS_FILE = MODEL_DIR+'cnn_model_'+VERSION+'.h5'
BEST_WEIGHTS_FILE = MODEL_DIR+'best_weights_'+VERSION+'.hdf5'

GRAPHS_DIR = 'graphs_out/'
ACCURACY_GRAPH_FILE = GRAPHS_DIR+'accuracy_'+VERSION+'.png'
LOSS_GRAPH_FILE = GRAPHS_DIR+'loss_'+VERSION+'.png'
CONFUSION_MATRIX_FILE = GRAPHS_DIR+'confusion_matrix'+VERSION+'.png'

TEST_DIR = 'data/test/'
TOCKENIZER_PATH = TEST_DIR+'token.pickle'
TEST_INDECIES_FILE = TEST_DIR+'recent_testdata_'+VERSION+'.pickle'
GENRE_FILE = TEST_DIR+'genres.pickle'

#PREPROCESS CONSTANTS
MAX_SONG_LENGTH = 2500
SONG_PER_GENRE = 4500
THRESH = 2000
REMOVE_COMMON_WORDS = False
DATA_KEYS = ['lyrics','lyrics_labels','unique_words_set','longest_song','genre_index','artist','song_titles']
SAVE_PREPROCESS = True

#PREPROCESS PATH CONSTANTS
PICKLE_ROOT = 'data/lyrics/original_pickles/'
PICKLE_OUT_PATH = 'data/lyrics/'
CHRISTIAN_PATH = 'Christian.pickle'
POP_PATH = 'Pop.pickle'
ROCK_PATH = 'Rock.pickle'
COUNTRY_PATH = 'Country.pickle'
RAP_PATH = 'Rap.pickle'
OUT_PICKLE = 'CNN_input.pickle' 

# RELOAD
SAVE_RELOAD = False
# RELOAD PATH CONSTANTS
CONFUSION_MATRIX_RELOAD_FILE = GRAPHS_DIR+'reloaded_confusion_matrix_'+VERSION+'.png'


DOC2VEC_PATH = MODEL_DIR + 'doc2vec/'
DOC2VEC_FILE = 'd2v.model'

