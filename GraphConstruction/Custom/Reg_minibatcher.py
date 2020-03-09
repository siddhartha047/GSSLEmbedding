import numpy as np
import torch
import logging
import sys

confidence=0.8

def extract_labeled(y_train_confidence):
    labeled = np.argwhere(y_train_confidence > confidence)
    return np.squeeze(labeled)



# TODO: make full torch implementation
class MiniBatcher(object):
    def __init__(self, batch_size, n_examples, y_train_all, train_confidence, shuffle=True):
        assert batch_size <= n_examples, "Error: batch_size is larger than n_examples"
        self.batch_size = batch_size
        self.n_examples = n_examples
        self.shuffle = shuffle
        self.train_confidence=train_confidence

        logging.info("batch_size={}, n_examples={}".format(batch_size, n_examples))

        #indexs=extract_labeled(train_confidence)
        #self.n_examples=len(indexs)

        print(self.n_examples)

        self.batch_size=min(batch_size,self.n_examples)
        self.idxs = np.arange(self.n_examples)
        #self.idxs=indexs
        if self.shuffle:
            np.random.shuffle(self.idxs)
        self.current_start = 0

    def get_one_batch(self):
        if self.shuffle:
            np.random.shuffle(self.idxs)
        self.current_start = 0
        while self.current_start < self.n_examples:
            batch_idxs = self.idxs[self.current_start:self.current_start+self.batch_size]
            self.current_start += self.batch_size
            yield torch.LongTensor(batch_idxs)


if __name__ == '__main__':
    train_confidence=np.append(np.ones(5),np.zeros(5))
    minibather=MiniBatcher(2,10,train_confidence)
    for idxs in minibather.get_one_batch():
        print(idxs)