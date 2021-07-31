from nltk.corpus import stopwords
from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS

stopwords = set(stopwords.words('english')).union(set(ENGLISH_STOP_WORDS))

#words to remove from stopwords
removedWords = set([
    "wouldn't", 'hasn', "doesn't", 'weren', 'wasn',
    "weren't", 'didn', 'mightn', "couldn't",
    "that'll", "didn't", "haven't", 'needn',
    "shouldn't", 'haven', "isn't", 'couldn', "it's",
    'not', 'aren', 'isn', 'doesn', "wasn't",
    'mustn', "should've", "shan't", "you'll", 'wouldn',
    "aren't", "won't", 'hadn', 'shouldn', "needn't",
    "hasn't", "mustn't", "hadn't", "mightn't", "you'd", "don't",
    "wouldnt", "doesnt", "werent", "couldnt",
    "thatll", "didnt", "youve", "havent",
    "shouldnt", "isnt", "its", "wasnt",
    "shouldve", "shant", "arent", "wont", "neednt",
    "hasnt", "mustnt", "hadnt", "mightnt", "dont"
    ])

stopwords = stopwords - removedWords
