from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
#https://miguelmalvarez.com/2015/03/20/classifying-reuters-21578-collection-with-python-representing-the-data/

def tf_idf(docs,TF_IDF_config):
    #tokenizer=tokenize
    tfidf = TfidfVectorizer( min_df=3,
                        max_df=0.90, max_features=TF_IDF_config['max_features'],
                        use_idf=True, sublinear_tf=True, ngram_range=TF_IDF_config['ngram'],
                        norm='l2');
    tfidf.fit(docs);

    X = tfidf.fit_transform(docs)
    print(tfidf.get_feature_names())
    print(X.shape)
    print(type(X).__name__)

    return X;

if __name__ == '__main__':
    corpus = [
        'This is the first document.',
        'This document is the second document.',
        'And this is the third one.',
        'Is this the first document?'
    ]
    TF_IDF_config = {
        'max_features': 3000,
        'ngram': (1, 1)  # min max range

    }
    tf_idf(corpus,TF_IDF_config)

# def feature_values(doc, representer):
#     doc_representation = representer.transform([doc])
#     features = representer.get_feature_names()
#     return [(features[index], doc_representation[0, index])
#                  for index in doc_representation.nonzero()[1]]
#
#
# if __name__ == '__main__':
#     for doc in test_docs:
#         print(feature_values(doc, representer))
#
# vectorizer = TfidfVectorizer()
# X = vectorizer.fit_transform(corpus)
# print(vectorizer.get_feature_names())
#
# print(X.shape)