#-*- coding: UTF-8 -*-  
import numpy
import copy
import theano
import random
from functools import reduce

def genBatch(data):
    m =0 
    maxsentencenum = len(data[0])
    for doc in data:
        for sentence in doc:
            if len(sentence)>m:
                m = len(sentence)
        for i in range(maxsentencenum - len(doc)):
            doc.append([-1])
    tmp = [numpy.asarray([sentence + [-1]*(m - len(sentence)) for sentence in doc], dtype = numpy.int32).T for doc in data]                          #[-1]是加在最前面
    tmp = reduce(lambda doc,docs : numpy.concatenate((doc,docs),axis = 1),tmp)
    return tmp 
            
def genLenBatch(lengths,maxsentencenum):
    lengths = [numpy.asarray(length + [1.0]*(maxsentencenum-len(length)), dtype = numpy.float32)+numpy.float32(1e-4) for length in lengths]
    return reduce(lambda x,y : numpy.concatenate((x,y),axis = 0),lengths)

def genwordmask(docsbatch):
    mask = copy.deepcopy(docsbatch)
    mask = [[[1.0 ,0.0][y == -1] for y in x] for x in mask]
    mask = numpy.asarray(mask,dtype=numpy.float32)
    mask[0] = numpy.ones([mask.shape[1]],dtype=numpy.float32) 
    return mask

def gensentencemask(sentencenum):
    maxnum = sentencenum[0]
    mask = numpy.asarray([[1.0]*num + [0.0]*(maxnum - num) for num in sentencenum], dtype = numpy.float32)
    return mask.T

class Dataset(object):
    def __init__(self, filename, emb,usrdict,prddict,maxbatch = 32,maxword = 500):
        lines = [x.split('\t\t') for x in open(filename).readlines()]        
        label = numpy.asarray(
            [int(x[2])-1 for x in lines],
            dtype = numpy.int32
        )
        usr = [usrdict.getID(line[0]) for line in lines] 
        prd = [prddict.getID(line[1]) for line in lines]
        docs = [x[3][0:len(x[3])-1] for x in lines] 
        docs = [x.split('<sssss>') for x in docs] 
        docs = [[sentence.split(' ') for sentence in doc] for doc in docs]
        docs = [[[wordid for wordid in [emb.getID(word) for word in sentence] if wordid !=-1] for sentence in doc] for doc in docs]
        tmp = list(zip(docs, label,usr,prd))
        #random.shuffle(tmp)
        tmp.sort(lambda x, y: len(y[0]) - len(x[0]))  
        docs, label, usr, prd = list(zip(*tmp))

        sentencenum = [len(x) for x in docs]
        length = [[len(sentence) for sentence in doc] for doc in docs]
        self.epoch = len(docs) / maxbatch                                        
        if len(docs) % maxbatch != 0:
            self.epoch += 1
        
        self.docs = []
        self.label = []
        self.usr = []
        self.prd = []
        self.wordmask = []
        self.sentencemask = []
        self.maxsentencenum = []

        for i in range(self.epoch):
            self.maxsentencenum.append(sentencenum[i*maxbatch])
            docsbatch = genBatch(docs[i*maxbatch:(i+1)*maxbatch])
            self.docs.append(docsbatch)
            self.label.append(numpy.asarray(label[i*maxbatch:(i+1)*maxbatch], dtype = numpy.int32))
            self.usr.append(numpy.asarray(usr[i*maxbatch:(i+1)*maxbatch],dtype = numpy.int32))
            self.prd.append(numpy.asarray(prd[i*maxbatch:(i+1)*maxbatch],dtype = numpy.int32))
            self.wordmask.append(genwordmask(docsbatch))
            self.sentencemask.append(gensentencemask(sentencenum[i*maxbatch:(i+1)*maxbatch]))
        

class Wordlist(object):
    def __init__(self, filename, maxn = 100000):
        lines = [x.split() for x in open(filename).readlines()[:maxn]]
        self.size = len(lines)

        self.voc = [(item[0][0], item[1]) for item in zip(lines, range(self.size))]
        self.voc = dict(self.voc)

    def getID(self, word):
        try:
            return self.voc[word]
        except:
            return -1

class Usrlist(object):
    def __init__(self, filename, maxn = 100000):
        lines = [x.split() for x in open(filename).readlines()[:maxn]]
        self.size = len(lines)

        self.usrlist = [(item[0][0], item[1]) for item in zip(lines, range(self.size))]
        self.usrdict = dict(self.usrlist)

    def getID(self, usr):
        try:
            return self.usrdict[usr]
        except:
            return -1

class Prdlist(object):
    def __init__(self, filename, maxn = 100000):
        lines = [x.split() for x in open(filename).readlines()[:maxn]]
        self.size = len(lines)

        self.prdlist = [(item[0][0], item[1]) for item in zip(lines, range(self.size))]
        self.prddict = dict(self.prdlist)

    def getID(self, prd):
        try:
            return self.prddict[prd]
        except:
            return -1

