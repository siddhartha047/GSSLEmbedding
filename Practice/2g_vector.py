import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from pprint import pprint  # pretty-printer

def get_data():
    documents = [
        "Human machine interface for lab abc computer applications",
        "A survey of user opinion of computer system response time",
        "The EPS user interface management system",
        "System and human system engineering testing of EPS",
        "Relation of user perceived response time to error measurement",
        "The generation of random binary unordered trees",
        "The intersection graph of paths in trees",
        "Graph minors IV Widths of trees and well quasi ordering",
        "Graph minors A survey",
    ]

    from collections import defaultdict

    # remove common words and tokenize
    stoplist = set('for a of the and to in'.split())
    texts = [
        [word for word in document.lower().split() if word not in stoplist]
        for document in documents
    ]

    # remove words that appear only once
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1

    texts = [
        [token for token in text if frequency[token] > 1]
        for text in texts
    ]

    return texts

from smart_open import open  # for transparently opening remote files
class MyCorpus(object):
    def __init__(self,dictionary,file):
        self.dictionary=dictionary
        self.file=file

    def __iter__(self):
        for line in open(self.file):
            yield self.dictionary.doc2bow(line.lower().split())

def learn():

    data=get_data()
    pprint(data)

    from gensim import corpora

    dictionary = corpora.Dictionary(data)
    dictionary.save('deerwester.dict')  # store the dictionary, for future reference
    print(dictionary)
    print(dictionary.token2id)

    new_doc = "Human computer interaction"
    new_vec = dictionary.doc2bow(new_doc.lower().split())
    print(new_vec)  # the word "interaction" does not appear in the dictionary and is ignored

    corpus = [dictionary.doc2bow(text) for text in data]
    corpora.MmCorpus.serialize('deerwester.mm', corpus)  # store to disk, for later use
    print(corpus)

    corpus_memory_friendly=MyCorpus(dictionary,"info.txt")

    for line in corpus_memory_friendly:
        print(line)

    return


def make_dictionary():
    from gensim import corpora
    from six import iteritems
    # collect statistics about all tokens

    stoplist = set('for a of the and to in'.split())

    dictionary = corpora.Dictionary(
        line.lower().split() for line in open('info.txt'))
    # remove stop words and words that appear only once
    stop_ids = [
        dictionary.token2id[stopword]
        for stopword in stoplist
        if stopword in dictionary.token2id
    ]
    once_ids = [tokenid for tokenid, docfreq in iteritems(dictionary.dfs) if docfreq == 1]
    dictionary.filter_tokens(stop_ids + once_ids)  # remove stop words and words that appear only once
    dictionary.compactify()  # remove gaps in id sequence after words that were removed
    print(dictionary)

    return
def corpus_format():
    from gensim import corpora
    corpus = [[(1, 0.5)], []]  # make one document empty, for the heck of it

    corpora.MmCorpus.serialize('corpus.mm', corpus)

    # corpora.SvmLightCorpus.serialize('corpus.svmlight', corpus)
    # corpora.BleiCorpus.serialize('corpus.lda-c', corpus)
    # corpora.LowCorpus.serialize('corpus.low', corpus)

    corpus = corpora.MmCorpus('corpus.mm')
    print(corpus)

    return

if __name__ == '__main__':
    #learn()
    #make_dictionary()
    corpus_format()