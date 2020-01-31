import gensim
import os
import shutil
import hashlib
from sys import platform

def getFileLineNums(filename):
    f = open(filename, 'r')
    count = 0
    for line in f:
        count += 1
    return count


def prepend_line(infile, outfile, line):
    with open(infile, 'r') as old:
        with open(outfile, 'w') as new:
            new.write(str(line) + "\n")
            shutil.copyfileobj(old, new)

def prepend_slow(infile, outfile, line):
    with open(infile, 'r') as fin:
        with open(outfile, 'w') as fout:
            fout.write(line + "\n")
            for line in fin:
                fout.write(line)

def load(filename,update_gensim_file,dim=300):
    num_lines = getFileLineNums(filename)
    gensim_file = update_gensim_file
    gensim_first_line = "{} {}".format(num_lines, dim)
    # Prepends the line.
    if platform == "linux" or platform == "linux2":
        prepend_line(filename, gensim_file, gensim_first_line)
    else:
        prepend_slow(filename, gensim_file, gensim_first_line)

    model = gensim.models.KeyedVectors.load_word2vec_format(gensim_file)
    return model


if __name__ == '__main__':
    dim=300
    GLOVE = "/Users/siddharthashankardas/Purdue/Dataset/Model/glove.6B/glove.6B."+str(dim)+"d.txt"
    updateGLOVE="/Users/siddharthashankardas/Purdue/Dataset/Model/glove.6B/gensim_glove.6B."+str(dim)+"d.txt"
    load(GLOVE,updateGLOVE,dim)