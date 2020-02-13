# from sklearn.feature_extraction.text import TfidfVectorizer
# from nltk.tokenize import RegexpTokenizer
# from nltk.stem import LancasterStemmer
#
# class Tokenizer(object):
#     def __init__(self):
#         self.tok = RegexpTokenizer(r'some_regular_expression')
#         self.stemmer = LancasterStemmer()
#     def __call__(self, doc):
#         return [self.stemmer.stem(token)
#                 for token in self.tok.tokenize(doc)]
#
#
# vectorizer = TfidfVectorizer(tokenizer=Tokenizer(),
#                      stop_words='english',
#                      use_idf=True,
#                      smooth_idf=True)
#
#
# from sklearn.decomposition import TruncatedSVD
# svd_model = TruncatedSVD(n_components=500,algorithm='randomized',n_iter=10, random_state=42)
#
# from sklearn.pipeline import Pipeline
# svd_transformer = Pipeline([('tfidf', vectorizer),
#                             ('svd', svd_model)])
#
# svd_matrix = svd_transformer.fit_transform(document_corpus)
# query_vector = svd_transformer.transform(query)
#
# from sklearn.metrics import pairwise_distances
#
# def pdistance(query_vector,svd_matrix):
#     distance_matrix = pairwise_distances(query_vector,
#                                      svd_matrix,
#                                      metric='cosine',
#                                      n_jobs=-1)
#     return distance_matrix