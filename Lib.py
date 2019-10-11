import re
import gzip
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import csv
import string
import json as jsn
from scipy import io
import pickle

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


def save_data(data,data_vector,data_rating,output_file,output_label,output_data,comment=""):
    print("Started Writing data")

    pickle.dump(data, open(output_data, "wb"))

    io.mmwrite(output_file, data_vector, comment=comment)

    f = open(output_label, 'w')
    f.write("%d\n" % len(data_rating))

    print("Writing Class label")
    with open(output_label, 'a') as f:
        for item in data_rating:
            f.write("%s\n" % item)
    f.close()


