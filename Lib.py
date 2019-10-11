import re
import gzip
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import csv
import string

# nltk.download('stopwords')
# nltk.download('punkt')

from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

lem = WordNetLemmatizer()
stem = PorterStemmer()

stop_words = stopwords.words('english')

def processText(text):
    # Remove punctuations
    text = re.sub('[^a-zA-Z]', ' ', text)

    # Convert to lowercase
    text = text.lower()

    # remove tags
    text = re.sub("&lt;/?.*?&gt;", " &lt;&gt; ", text)

    #remove special characters and digits
    text = re.sub("(\\d|\\W)+", " ", text)

    ##Convert to list from string
    text = text.split()

    ##Stemming
    ps = PorterStemmer()

    # Lemmatisation
    lem = WordNetLemmatizer()
    text = [lem.lemmatize(word) for word in text if not word in stop_words]
    # text = " ".join(text)
    return text

def num(s):
    try:
        return float(s)
    except ValueError:
        return 0

def parse(path):
  g = gzip.open(path, 'r')
  for l in g:
    yield eval(l)

# def processText(sentence):
#     tokens = word_tokenize(sentence)
#     tokens = [w.lower() for w in tokens]
#     table = str.maketrans('', '', string.punctuation)
#     stripped = [w.translate(table) for w in tokens]
#     words = [word for word in stripped if word.isalpha()]
#     words = [w for w in words if not w in stop_words]
#
#     return ' '.join(words)


