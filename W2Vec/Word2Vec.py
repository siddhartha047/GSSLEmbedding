import gensim
import os
from Path import *
import sys
import numpy as np
import timeit

MODEL_NAME="GOOGLE"

def load_model(name):
    if(name=='GOOGLE'):
        return gensim.models.KeyedVectors.load_word2vec_format(os.path.join(GOOGLE_NEWS), binary=True)
    elif(name=='GLOVE'):
        return []
    else:
        print("The model is not defined")
        sys.exit(0)

def apply_embedding(texts,model):
    doc = [word for word in texts if word in model.wv.vocab]
    return np.mean(model.wv[doc], axis=0)

