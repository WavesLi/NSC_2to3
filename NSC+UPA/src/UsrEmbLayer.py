#-*- coding: UTF-8 -*-  
import theano
import theano.tensor as T
import numpy
import pickle

class UsrEmbLayer(object):
    def __init__(self, rng, n_usr, dim, name, prefix=None):
        self.name = name

        if prefix == None:
            U_values = numpy.zeros((n_usr+1,dim),dtype=numpy.float32)
            U = theano.shared(value=U_values, name='U', borrow=True)
        else:
            f = file(prefix + name + '.save', 'rb')
            U = pickle.load(f)
            f.close()
        self.U = U
        
        self.output = self.U
        self.params = [self.U]

    def save(self, prefix):
        f = file(prefix + self.name + '.save', 'wb')
        for obj in self.params:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()
