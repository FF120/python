# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 22:21:37 2015

@author: zck
"""

# -*- coding: utf-8 -*-
"""
http://deeplearning.net/software/theano/tutorial/examples.html#a-real-example-logistic-regression

Created on Fri Dec 25 19:19:13 2015

@author: zck
"""

import numpy as np
from numpy import *
import theano
import theano.tensor as T
import cPickle

"""
返回numpy矩阵是(用户总数*天数*网站数)=(299320*56*10)
"""
def read_data():
    N = 299368
    x = np.linspace(0,0,N*56*10)
    x = x.reshape(N,56,10)
    name2id = {}
    names = []
    total = 0
    for line in open('d:/chrome/part-r-00000'):
        strs = line.split('\t')
        if ( len(strs)!=4 ):
            continue
        name = strs[0]
        if name not in name2id:
            name2id[name] = total
            names.append(name)
            total = total+1
        id = name2id[name]
        day = (int(strs[1][1])-1)*7 + int(strs[1][3])-1
        site = int(strs[2][1:])-1
        count = int(strs[3])
        x[id,day,site] = count
        
    return (x, name2id, names)

def cos_sim(a,b):
    s = sum(multiply(a,b),axis=1)
    t = multiply(np.linalg.norm(a,axis=1),np.linalg.norm(b,axis=1) )
    return s/t
    
def _squared_magnitude(x):
    return T.sqr(x).sum(axis=1)
def _magnitude(x):
    return T.sqrt(T.maximum(_squared_magnitude(x), np.finfo(x.dtype).tiny))
    #return T.sqrt(T.maximum(_squared_magnitude(x), 0.1))
def cosine(x, y):
    return ( (x * y).sum(axis=1) / (_magnitude(x) * _magnitude(y)) )

def write_ans(p, names):
    with open('test_sigmoid.txt','w') as of:
        for i in xrange(0, p.shape[0]):
            name = names[i]
            outstr = name+'\t'
            for d in xrange(0,7):
                for v in xrange(0,10):
                    outstr = outstr+str( (int)(p[i,d*10+v]*300) )
                    if d==6 and v==9:
                        outstr = outstr+'\n'
                    else:
                        outstr = outstr+','
            of.write(outstr)

N = 299368
feats = 420
rng = np.random
#D = (rng.rand(N, feats), rng.rand(N, 70) )
(data, name2id, names) = read_data()
data = np.asarray(data,theano.config.floatX)
data = data.reshape(N,560)
#xTr = data[:,0:350]
#yTr = data[:,350:420]
#xVal = data[:,70:420]
#yVal = data[:,420:490]
#xTest = data[:,140:490]

xTr = data[:,0:420]
yTr = data[:,420:490]
xTest = data[:,70:490]
#D = ( data[:30000,0:350], data[:30000,350:420], data[:30000,420:490] )

# Declare Theano symbolic variables
x = T.matrix("x")
y = T.matrix("y")
lr = T.scalar("lr")
w = theano.shared(np.asarray(rng.rand(feats,70),theano.config.floatX),name="w")
b = theano.shared(np.asarray(rng.rand(70),theano.config.floatX),name="b")
print("Initial model:")
print(w.get_value().shape)
print(b.get_value().shape)

# Construct Theano expression graph
p = T.nnet.softplus( T.dot(x, w) + b )
xent = cosine(y,p) # Cosine loss function
cost = xent.mean() # The cost for test
#cost = xent.mean() - 0.01 * (w ** 2).mean()# The cost to minimize
gw, gb = T.grad(cost, [w, b])             # Compute the gradient of the cost
                                          # (we shall return to this in a
                                          # following section of this tutorial)

# Compile
train = theano.function(
          inputs=[x,y,lr],
          outputs=[cost],
          updates=((w, w + lr * gw), (b, b + lr * gb)))
predict = theano.function(inputs=[x], outputs=p)

training_steps = 20
#learn_rate = 500
learn_rate = 600
# Train
#for i in xrange(1,training_steps+1):
acc = train(xTr, yTr, learn_rate)
   # print( acc )
   # if i%100==0:
        #phat = predict(xVal)
        #sim = cos_sim(phat, yVal)
        #print( sum(sim[sim==sim])/( N ) )
        #learn_rate = learn_rate*0.96
        #print( 'iterator is ' + str(i) + ' learn_rate is ' + str(learn_rate))
   # if i%1000==0 and i>0:
       # f = file('sigmoid_itr'+str(i)+'.model', 'wb')
       # cPickle.dump( (w,b), f, protocol=cPickle.HIGHEST_PROTOCOL)
       # f.close()
#    print('')

print("Final model:")
print(w.get_value())
print(b.get_value())
#print(cos_sim(D[1], predict(D[0])) )
phat = predict( xTest )
phat[phat<0] = 0
write_ans(phat, names)
