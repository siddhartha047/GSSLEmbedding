from gensim.models.doc2vec import Doc2Vec
from nltk.tokenize import word_tokenize
import numpy as np

model= Doc2Vec.load("d2v.model")
#to find the vector of a document which is not in training data
# test_data = word_tokenize("I love chatbots".lower())
# v1 = model.infer_vector(test_data)
# print("V1_infer", v1)

# print(model.docvecs.most_similar(0))
#
# #to find most similar doc using tags
# similar_doc = model.docvecs.most_similar('1')
# print(similar_doc)


# to find vector of doc in training data using tags or in other words, printing the vector of document at index 1 in training data
#print(model.docvecs['0'])

vectors=np.zeros((model.corpus_count,model.vector_size),np.float)
print(vectors.shape)
for i in range(model.corpus_count):
    vectors[i]=model.docvecs[i]

print(vectors.shape)

print(vectors[0])
