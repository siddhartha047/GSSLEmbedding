from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer
from nltk.stem import LancasterStemmer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline

import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

cachedStopWords = stopwords.words("english")


def tokenize(text):
    min_length = 3
    words = map(lambda word: word.lower(), text);
    words = [word for word in words
                  if word not in cachedStopWords]
    tokens =(list(map(lambda token: PorterStemmer().stem(token),
                  words)));
    p = re.compile('[a-zA-Z]+');
    filtered_tokens = list(filter(lambda token:
                  p.match(token) and len(token)>=min_length,
         tokens));

    return filtered_tokens

def lsi(data,LSI_config):
    # vectorizer = TfidfVectorizer(tokenizer=Tokenizer(),
    #                              stop_words='english',
    #                              use_idf=True,
    #                              smooth_idf=True)

    # vectorizer = TfidfVectorizer(tokenizer=tokenize,
    #                              #stop_words='english',
    #                              min_df=3,
    #                              max_df=0.90, max_features=LSI_config['tfidf_features'],
    #                              use_idf=True, sublinear_tf=True,
    #                              norm='l2');

    tfidf = TfidfVectorizer(min_df=3,
                            max_df=0.90, max_features=LSI_config['tfidf_features'],
                            use_idf=True, sublinear_tf=True, ngram_range=LSI_config['ngram'],
                            norm='l2');
    tfidf.fit(data);
    X = tfidf.fit_transform(data)

    print(tfidf.get_feature_names())
    print(X.shape)
    n_features=X.shape[1]


    svd_model = TruncatedSVD(n_components=min(LSI_config['max_features'],n_features-1), algorithm='randomized', n_iter=10, random_state=42)

    # svd_transformer = Pipeline([('tfidf', vectorizer),
    #                             ('svd', svd_model)])

    svd_model.fit(X)
    svd_matrix = svd_model.fit_transform(X)


    #svd_matrix = svd_transformer.fit_transform(data)

    #print(svd_matrix)
    print(svd_matrix.shape)

    # query_vector = svd_transformer.transform(query)

    return svd_matrix


if __name__ == '__main__':
    data=['This is the first document. brown adament',
        'This document is the second document. brown adament',
        'And this is the third one. adore red red brown',
        'Is this the first document adament red red?'
          ]
    lsi_config={
        'tfidf_features': 30,
        'max_features': 4,
        'ngram':(1,1)
    }
    lsi(data,lsi_config)