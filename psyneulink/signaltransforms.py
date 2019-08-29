import numpy as np
import intervals
import itertools
import time
# import plotlywrap
# import plotly
import copy
import importlib
import os
import pandas as pd
from importlib import reload
# import keras
# import plotly
import sklearn
from sklearn.decomposition import PCA
# import plotly.offline as po
# import plotly.graph_objs as go
import inspect
try:
    from atlatl import rivet, barcode, hera
    from atlatl.rivet import Point, PointCloud
except:
    print("couldn't find atlatl")
import math
import re
import scipy
import timeit
# import filepaths
import torch
# import pytorchfull
# import demoscripts.representationangle_demo
# import autodifffull
# from autodifffull import *



# keras platform dependencies

# import tensorflow as tf
# from keras import backend as K
# from keras.layers import Dense, Input
# from keras.layers.merge import concatenate
# from keras.models import Model
# from keras import losses, optimizers, metrics, callbacks
# from plotlywrap import *
# from demoscripts.representationangle_demo import *

import numpy as np
# import h5py
# import os
# import shutil
import copy

# import simple
# from simple import *

# pytorch dependencies

import  torch
from    torch               import  nn
from    torch               import  optim
from    torch.utils         import  data


################################################################################
    # NAMES & ABBREVIATIONS
################################################################################

#    gs, ts, ps                     =   OVERVIEW
#
#                                       the input units in each network split
#                                       naturally into two groups (or layers).
#                                       there is a
#                                       FEATURE LAYER with
#                                       (# features per dim) x (# dims) units,
#                                       each corresponding to one feature of one
#                                       dimension, and a
#                                       CONTROL LAYER with
#                                       (# dims) * (# dims) units
#                                       corresponding to the set of all possible
#                                       (input dimension, output dimension)
#                                       pairs, aka tasks.
#                                       the activation pattern on the feature
#                                       layer is the FEATURE VECTOR and the
#                                       activation pattern on the control layer
#                                       is the CONTROL VECTOR.
#                                       NB: the activation of each unit in the
#                                       control vector is always 0 or 1.
#                                       Depending on how the network is trained,
#                                       some units in the control vector may
#                                       never take a positive value (they are
#                                       always zero).  If we identify each
#                                       control unit with its corresponding task
#                                       then this can be understood to say
#                                       that the task is never trained.  If more
#                                       than one control unit is nonzero, it
#                                       means the network must perform multiple
#                                       tasks simultaneously.
#
#                                       DEFINITIONS
#
#                                       GROUND SET ~ gs
#                                       The set of all tasks for which the
#                                       network receives some training.
#                                       Equivalently, the set of control units
#                                       that take value 1 at some point in
#                                       training.
#
#                                       PERFORMANCE SET ~ ps
#                                       The set of input units that take value
#                                       1 in a single control vector.
#                                       Equivalently, a set of tasks that we
#                                       ask the network to perform
#                                       simultaneously.
#
#                                       TAST SET ~ ts
#                                       Any set of tasks.
#
#                                       ENCODING
#
#                                       each of these sets is stored as an m x 2
#                                       array, with first column representing
#                                       the input dimension and second column
#                                       representing output.  each row
#                                       corresponds to a single task
#
#   oS                                  OUTPUT SIGNAL
#                                       the desired activity pattern in the
#                                       output layer, for a given feature
#                                       vector and control vector
#   iS                                  INPUT SIGNAL
#                                       = feature vector
#   pS                                  PREDICTED  SIGNAL
#                                       the observed (rather than ideal)
#                                       activation pattern in the output layer
#   cS                                  CONTROL SIGNAL
#                                       = control vector
#   signal transform                    the (ideal, desired) function that
#                                       sends each input signal to the
#                                       corresponding output signal.  every
#                                       control vector determines a distinct
#                                       signal transform
#   parameter                           refers to the parameters that determine
#                                       a network; generally encoded as a
#                                       dictionary P
#    st                             =   signal transform
#    sta                            =   signal transform array; in certain cases
#                                       right-multiplication by this matrix
#                                       converts an input feature vector to the
#                                       corresponding (usually one-hot) output
#                                       vector
#    iS                             =   input signal
#    oS                             =   output signal
#    pS                             =   predicted signal
#    cv,cS                          =   control vector / control signal
#                                       the cv/cS is obtained, intuitively, by
#                                       identifying the set of all single tasks
#                                       with the vertices of an nd x nd grid of
#                                       points, with ROWS ~ INPUTS, COLS ~ OUTS,
#                                       then flattening in the style of python,
#                                       CONCATENATING ROWS.
#    nf                             =   number of feature nodes per dimension
#    nd                             =   number of dimensions
#    ndsq                           =   number of dimensions squared
#    lr                             =   learning rate
#    lra                            =   initial learning rate
#    lrb                            =   final learning rate
#    lrs                            =   learning rate scale(decay) term
#    lrp                            =   learning rate period
#    mb                             =   minibatch
#    mbi                            =   minibatch index (while a network is training)
#    idc                            =   input data cardinality (number of rows in the big matrix of all input-target pairs used for training)
#    wu                             =   weight update
#    nwu                            =   number of weight updates (after a network has been trained)
#    c                              =   cardinality
#    <>c                            =   cardinality of <>
#    gsc                            =   cardinality of the ground set
#    tsc                            =   cardinality of the task set
#    psc                            =   cardinality of the performance set
#    mpsc                           =   max cardinality of the performance set
#    <>s                            =   set of <>, e.g. pss = set of performance
#                                       sets
#   P                                   a "parameter" dictionary object (see
#                                       above)
#   N,net                               a "network" dictionary, containing all
#                                       the data of a "parameter" dictionary,
#                                       plus a Keras model (and maybe some
#                                       other things)
#   a                               =   general prefix for "alpha" = "initial",
#                                       e.g. aw = initial weight
#   fp                                  filepath
#   dir                                 directory
#   -l                                  l at the end of a name means 'list'
#   -a                                  a at the end of a name means 'array'
#   -t                                  t at the end of a name means 'tensor'


################################################################################
    # FILE NAMES
################################################################################

#   EXAMPLE: 20181211-205321-f3d5gsc25nwu50000pscr1-1
#
#   INTERPRETATION:
#   - 20181211-205321
#               network was created december 11, 2018 at 20hrs:53min:21sec
#   - f3        network has 3 features per dimension
#   - d5        network has 5 input (and 5 output) dimensions
#   - gsc25     this network has 25 tasks in its ground set; since this
#               network has 5 input and output dimensions, we can deduce that
#               the ground set is the set of all input-output pairs
#   - nwu50000  number of weight updates used in training = 50000
#   - pscr1-1   the performance set cardinality range for training data is {1}.
#               this means that every control vector used in training has
#               exactly one nonzero unit.  if instead we had had pscr = 0-5,
#               then the control vectors used in training would have had
#               anywhere from 0 to 5 nonzero units


################################################################################
    # COMBINATORICS
################################################################################

def length_intervalsize__norepeatperm(len,intervalsize):
    m                               =   1+(len//intervalsize)
    perm                            =   [np.random.permutation(intervalsize) for
                                         p in range(m)]
    perm                            =   np.concatenate(perm,0)
    perm                            =   perm[:len]
    return perm

def check_length_intervalsize__norepeatperm():
    numits                          =   5
    for p                           in  range(numits):
        len                         =   np.random.randint(10,1000)
        intervalsize                =   np.random.randint(10,100)
        perm                        =   length_intervalsize__norepeatperm(len,
                                        intervalsize)
        for q                       in  range(len // intervalsize):
            I                       =   intervals.itv(q,intervalsize)
            if not np.array_equal(np.sort(perm[I]), np.arange(intervalsize)):
                print("test failed")
                return len, intervalsize, perm, q

        rem                         =   perm[-(len % intervalsize):]
        if np.unique(rem).shape[0]  !=  len % intervalsize:
            print("test failed")
            return len, intervalsize, perm
    print("test passed")
    return []

# generate a sequence of length <length> whose entries are drawn from the set
# {0,..., capval-1}.  if conserve == true then each element of that set appears
# exactly p times in the first p * capval entries of the sequence ... for every
# p <= length // capval; the entries coming after capval*(length // capval) are
# sampled uniformly at random from {0, ..., capval-1}.
# if replace == true then the elements are drawn with replacement.  if both
# replace and conserve are false, then the entries of the sequence are sampled
# uniformly at random from the set
def length_capval_conserve_replace__randomsequence(
    length,capval,conserve,replace=False):
    if replace:
        return np.random.randint(0,capval,size=(length,))
    else:
        if conserve:
            return length_intervalsize__norepeatperm(length,capval)
        else:
            return np.random.permutation(length)

def rowsortperm(A):
    B                               =   A.transpose()
    B                               =   np.flipud(B)
    B                               =   tuple(B)
    perm                            =   np.lexsort(B)
    return perm

def inversepermutation(perm):
    q                               =   copy.copy(perm)
    q[perm]                         =   range(heightVlength(perm))
    return q

def check_inversepermutation():
    perm                            =   np.random.permutation(500)
    ip                              =   inversepermutation(perm)
    check0                          =   np.array_equal(ip[perm], np.array([p for p in range(500)]))
    check1                          =   np.array_equal(perm[ip], np.array([p for p in range(500)]))
    if not (check0 and check1):
        print("test failed")
        return perm,ip,check0,check1
    print("test passed")
    return []

def nwurange():
    return np.concatenate((np.arange(10**4,10**5+1,10**4),np.arange(10**5,10**6+1,10**5)))

#   NB:   UNTESTED, BUT WORKS ON THE FOLLOWING
#   a = np.array([0,1])
#   a = np.reshape(a,(2,1))
#   b = a_b__atensorbbyrow(a,a)
#   b = np.concatenate(b,axis=1)c = a_b__atensorbbyrow(a,b)
#   c = np.concatenate(c,axis=1)
def id_od_fv__tensorloc(id,od,fv,nid=5,nod=5,nfv=243):
    offset1                         =   id*nod*nfv
    offset2                         =   od*nfv
    return offset1+offset2+fv

def length_segmentsize__segmentedrandperm(length,segmentsize):
    perm                            =   np.arange(length)
    for p                           in  range(length//segmentsize):
        np.random.shuffle(perm[p*segmentsize:(p+1)*segmentsize-1])
    remainder                       =   length % segmentsize
    #   NB  If the if condition were missing, this would reshuffle the entire
    #       permutation whenever remainder == 0
    if                          remainder > 0:
        np.random.shuffle(perm[-remainder:])
    return perm

#   CHECK
def length_segmentsize__segmentedrandperm_check():
    length                          =   600
    segmentsize                     =   40
    perm                            =   length_segmentsize__segmentedrandperm(length,segmentsize)
    for p                           in  range(15):
        a                           =   np.sort(perm[intervals.itv(p,40)])
        if                          not np.array_equal(a,intervals.itv(p,40)):
            print("error")
            print(np.min(a))
            print(np.max(a))
            print(p)
            return
    print('test passed')

def perm__strpermfn(perm):
    #   param: a            m x n array
    #   param: colperm      permutation on n
    #   returns: rowperm    permutation such that rows of a[rowperm,colperm]
    #                       are in lexicographic order
    def f(w):
        s                           =   [w[p] for p in perm]
        s                           =   ''.join(s)
        return s
    return f

def ndxnfxnf__allrowseq(a):
    # example: [    [[a b],   [[e f],
    #               [c d]]  ,  [g h]]   ] |-->
    #           a b e f
    #           a b g h
    #           c d e f
    #           c d g h
    nd                              =   a.shape[0]
    nf                              =   a.shape[1]
    L                               =   []
    for p in range(nd):
        nrept                       =   nf ** (nd-p-1)
        ntile                       =   nf ** p
        x                           =   copy.copy(a[p])
        x                           =   np.repeat(x,nrept,axis=0)
        x                           =   np.tile(x,(ntile,1))
        L.append(x)
    ciSs                            =   np.concatenate(L,axis=1)
    return ciSs

def nf_x_ndxnf__allrowseq(a):
    # example: [    [a b e f],
    #               [c d g h]   ]   |-->
    #           a b e f
    #           a b g h
    #           c d e f
    #           c d g h
    a                               =   copy.copy(a)
    nf                              =   a.shape[0]
    nd                              =   a.shape[1]//nf
    b                               =   [a[:,intervals.itv(p,nf)] for p in range(nd)]
    b                               =   np.array(b)
    return ndxnfxnf__allrowseq(b)

def nd__inoutperm(nd):
#   If control units are ordered lexicographically according to  (first) input
#   dimension  and (second) output, then  precomposition with this  permutation
#   will order them lexicographically according to  first output, then  input.
    a                               =   [np.arange(p,nd**2,nd) for p in range(nd)]
    a                               =   np.concatenate(a,axis=0)
    return a

def ai_bi__asortedbyb(ai,bi):
    s                               =   np.argsort(bi)
    if type(ai)                     ==  list:
        return [ai[s[p]] for p in range(hl(bi))]
    else:
        return np.array([[ai[s[p]] for p in hl(bi)]])

################################################################################
#   LIST DICTIONARIES --
################################################################################

def ld__pcca_labl(ld):
    """
    :param ld: dictionary of equal-length lists of numbers (or possibly arrays)
    :return: pearson correlation matrix with row labels
    """
    rowl                =   []
    m                   =   np.max([hl(x) for x in ld.values()])
    for key in ld.keys():
        row             =   copy.copy(ld[key])
        row             =   list(row)
        if hl(row)      <   m:
            row.append(0)
        rowl.append(row)
    a                   =   np.array([x for x in rowl])

    pcca                =   np.corrcoef(a)
    labell              =   list(ld.keys())
    return pcca, labell

def ld__ldjustified(ld):
    ld                  =   copy.deepcopy(ld)
    m                   =   np.max([hl(x) for x in ld.values()])
    for key in ld.keys():
        row             =   ld[key]
        row             =   list(row)
        if hl(row)      <   m:
            row.append(0)
        ld[key]         =   row
    return ld


def ld__pccaheatmap(ld,title=''):
    pcca,labl         =   ld__pcca_labl(ld)
    pcca                =   np.round(100*pcca,0).astype(int)
    heatmap(pcca,x=labl,y=labl,annotate=True,title=title)

def dadd(d,keyname,keyval):
    """
    'dictionary add' function
    :param d: dictionary
    :param keyname: key
    :param keyval: value
    :return: either appends the value to the list stored in d[key] or creates
    this key, initiates its value as an empty list, and adds keyval to the list
    """
    keyval  =   copy.copy(keyval)
    if keyname in d:
        d[keyname].append(keyval)
    else:
        if hasattr(keyval,'__iter__'):
            d[keyname]      =   keyval
        else:
            d[keyname] = [keyval]

################################################################################
    # STATISTICS
################################################################################

def a_labels__withinlabelvariance(a,labels,agg = 'None'):
    u                               =   np.unique(labels)
    m                               =   u.shape[0]
    X                               =   np.zeros((m,a.shape[1]))
    for p                           in  range(m):
        I                           =   np.argwhere(labels == u[p])
        b                           =   a[I,:]
        X[p]                        =   np.var(b,axis=0)
    if agg                          ==  'sum':
        X                           =   np.sum(X,axis=0)
    if agg                          ==  'mean':
        X                           =   np.mean(X,axis=0)
    return X

def x_y__sidedsepscore_sidedseppost(x,y):
    x                               =   np.sort(copy.deepcopy(x))
    y                               =   np.sort(copy.deepcopy(y))
    if type(x)                      ==  np.ndarray:
        x                           =   np.ndarray.flatten(x)
    if type(y)                      ==  np.ndarray:
        y                           =   np.ndarray.flatten(y)
    score                           =   0
    post                            =   None
    xc                              =   hl(x)
    yc                              =   hl(y)
    for p                           in  range(len(x)):
        xscore                      =   xc-p
        yscore                      =   np.sum([1 for q in range(yc) if y[q] < x[p]])
        rawscore                    =   xscore+yscore
        tempscore                   =   rawscore/(xc+yc)
        if tempscore                >   score:
            score                   =   tempscore
            post                    =   x[p]
    return score,post

def x_y__sepscore_seppost(x,y):
    scorea,posta                    =   x_y__sidedsepscore_sidedseppost(x,y)
    scoreb,postb                    =   x_y__sidedsepscore_sidedseppost(y,x)
    if scorea                       >   scoreb:
        return scorea,posta
    else:
        return scoreb,postb

def x_y_post__sidedsepscore(x,y,post):
    x                               =   np.sort(copy.deepcopy(x))
    y                               =   np.sort(copy.deepcopy(y))
    if type(x)                      ==  np.ndarray:
        x                           =   np.ndarray.flatten(x)
    if type(y)                      ==  np.ndarray:
        y                           =   np.ndarray.flatten(y)
    xscore                          =   np.sum([1 for p in range(hl(x)) if x[p]>=post])
    yscore                          =   np.sum([1 for p in range(hl(x)) if y[p]< post])
    score                           =   (xscore+yscore)/(hl(x)+hl(y))
    return score

def x_y_post__sepscore(x,y,post):
    scorea                          =   x_y_post__sidedsepscore(x,y,post)
    scoreb                          =   x_y_post__sidedsepscore(y,x,post)
    score                           =   np.max([scorea,scoreb])
    return score

def checksepfun():
    for offset                      in  [0,0.3,0.6,0.9]:
        x                           =   np.random.rand(100)
        y                           =   np.random.rand(100)+offset
        score,post                  =   x_y__sepscore_seppost(x,y)
        tracex                      =   go.Scatter(x=np.arange(100),y=np.sort(x),name="x")
        tracey                      =   go.Scatter(x=np.arange(100),y=np.sort(y),name="y")
        tracebar                    =   go.Scatter(x=[0,100],y=[post,post])
        layout                      =   go.Layout(title=str(score))
        po.plot({"data":[tracex,tracey,tracebar],"layout":layout})

def a_alab__sepscore_seppost(a,alab):
    a                               =   copy.deepcopy(a)
    vals                            =   np.unique(alab)
    if not hl(vals)                 ==  2:
        print("error: xlab may only take 2 values")
        return
    x                               =   [a[p] for p in range(hl(a)) if a[p] == vals[0]]
    y                               =   [a[p] for p in range(hl(a)) if not a[p] == vals[0]]
    return x_y__sepscore_seppost(x,y)


################################################################################
    # FUNCTIONS
################################################################################

def io__pss(i,o):
    pss                 =   [np.array([[i,o]])]
    return pss

def arraylisteq(L,R):
    if not len(L) == len(R):
        return False
    for p                           in  range(len(L)):
        if not np.array_equal(L[p],R[p]):
            return False
    return True

def isnumber(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def isinteger(s):
    if s == int(s):
        return True
    else:
        return False

def vec__rowvec(v):
    if len(v.shape)                 ==  1:
        return np.reshape(v,(1,v.shape[0]))
    else:
        return v

def vec__colvec(v):
    if len(v.shape)                 ==  1:
        return np.reshape(v,(v.shape[0],1))
    else:
        return v

def array__rowstackvec(array):
    m                               =   np.product(array.shape)
    v                               =   np.reshape(array,(m,))
    return v

def array__colstackvec(array):
    return array__rowstackvec(np.transpose(array))

def pandapath(keyword):
    if keyword                      ==  "csv":
        return hiddentaskingdir()+'/data/nndf.csv' #"/Users/gh10/Google Drive/gregtaylorgoogledrive_nongooglefiles/GregDirectory/python_gd/gdg_hiddentasking/data/nndf.csv"
    elif keyword                    ==  "pkl":
        return hiddentaskingdir()+'/data/nndf.pkl' #"/Users/gh10/Google Drive/gregtaylorgoogledrive_nongooglefiles/GregDirectory/python_gd/gdg_hiddentasking/data/nndf.pkl"

def time32bit():
    t                               =   time.time()
    t                               =   (t  *  10**6)  %  (2**32)
    t                               =   int(t)
    return t

def idfun(x):
    return x

# def cSs_iSs_int__cv_iS(cSs,iSs,p):
#     m                               =   iSs.shape[0]
#     n                               =   cSs.shape[0]
#     cvind                           =   (p // m) % n
#     iSind                           =   p % m
#     return cSs[cvind],iSs[iSind]
#
# def ts_iSs_int__t_iS(ts,iSs,p):
#     m                               =   iSs.shape[0]
#     n                               =   len(ts)
#     cvind                           =   (p // m) % n
#     iSind                           =   p % m
#     return ts[cvind],iSs[iSind]

def heightVlength(a):
    if type(a)                      ==  list:
        m                           =   len(a)
    else:
        m                           =   a.shape[0]
    return m

def hl(a):
    return heightVlength(a)

#   *   a and b are matrices
#   *   returns the pth row of
#       [ A[0] B[0]
#         A[0] B[1]
#         ...
#         A[1] B[0]
#         ...      ]
#   *   if p is too large, we reduce it modulo the number of rows of this matrix
def a_b_int__rowofarowtensorb(a,b,p):
    m                               =   heightVlength(a)
    n                               =   heightVlength(b)
    p                               =   p % (m * n)
    aind                            =   p // n
    bind                            =   p %  n
    return a[aind], b[bind]

def check_a_b_int__rowofarowtensorb():
    numit                           =   10
    for p                           in  range(numit):
        x                           =   np.random.randint(50,high=100,size=(2,))
        m                           =   x[0]
        n                           =   x[1]
        A                           =   np.random.rand(m,m)
        B                           =   np.random.rand(n,n)
        a                           =   np.repeat(A,n,0)
        b                           =   np.tile(B,(m,1))
        for q                       in  range(m*n):
            x                       =   [a[p],b[p]]
            y                       =   a_b_int__rowofarowtensorb(a,b,p)
            if (not np.array_equal(x[0],y[0])) or (not np.array_equal(x[1],y[1])):
                print("test failed")
                return m,n,A,B,a,b,p,x,y
    print("test passed")
    return []

def a_b__atensorbbyrow(a,b):
    m                               =   heightVlength(a)
    n                               =   heightVlength(b)

    A                               =   np.repeat(a,n,0)
    if type(a)                      ==  list:
        A                           =   [p for p in a for i in range(n) ]
    elif type(a)                    ==  np.ndarray:
        A                           =   np.repeat(a,n,0)

    if type(b)                      ==  list:
        B                           =   [p for i in range(m) for p in b]
    elif type(b)                    ==  np.ndarray:
        if b.ndim                   ==  2:
            B                       =   np.tile(b,(m,1))
        elif b.ndim                 ==  1:
            B                       =   np.tile(b,(m))

    return A,B

def check_a_b__atensorbbyrow():
    a                               =   [1,2,3]
    b                               =   np.array([[5,6],[7,8]])
    A0                              =   [1,1,2,2,3,3]
    B0                              =   np.array([[5,6],[7,8],[5,6],[7,8],[5,6],[7,8]])
    A1                              =   [1,2,3,1,2,3]
    B1                              =   np.array([[5,6],[5,6],[5,6],[7,8],[7,8],[7,8]])
    A0v,B0v                         =   a_b__atensorbbyrow(a,b)
    B1v,A1v                         =   a_b__atensorbbyrow(b,a)
    check                           =   [A0 == A0v, np.array_equal(B0,B0v), A1 == A1v, np.array_equal(B1,B1v)]
    if not all(check):
        print("test failed")
        return check, a,b,A0,A0v,B0,B0v,A1,A1v,B1,B1v
    print("test passed")
    return []


def noisycircle():
    noisescale                      =   0.2
    theta                           =   np.arange(0,1,1/300)
    theta                           =   2*np.pi*theta
    noise                           =   np.random.rand(300,2)
    noise                           =   np.multiply(noisescale,noise)
    x                               =   np.sin(theta)
    y                               =   np.cos(theta)
    x                               =   x+noise[:,0]
    y                               =   y+noise[:,1]
    return x,y

def printval(x,y):
    if type(y) == np.ndarray:
        print(y + " = " + np.array_str(x))
    else:
        print(y + " = " + str(x))

def array_rowpairfun__pairwise(A,fun):
    m                               =   A.shape[0]
    B                               =   np.zeros((m,m))
    combs                           =   itertools.combinations(range(m),2)
    for comb                        in  combs:
        i0                          =   comb[0]
        i1                          =   comb[1]
        val                         =   fun(A[i0],A[i1])
        B[i0,i1]                    =   val
        B[i1,i0]                    =   val
    return B

def vector__normalvector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def vectorpair__angleinrad(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> vectorpair__angleinrad((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> vectorpair__angleinrad((1, 0, 0), (1, 0, 0))
            0.0
            >>> vectorpair__angleinrad((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = vector__normalvector(v1)
    v2_u = vector__normalvector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def cloudpair__angleinrad(A,B):
    A                               =   sklearn.preprocessing.normalize(A)
    B                               =   sklearn.preprocessing.normalize(B)
    C                               =   np.matmul(A,np.transpose(B))
    C                               =   np.clip(C, -1.0, 1.0)
    return np.arccos(C)

def cloud__angleinrad(A):
    return cloudpair__angleinrad(A,A)

def cloudpair__linearSVMclassifier(X,Y):
    nx                              =   X.shape[0]
    labx                            =   np.zeros((nx))
    ny                              =   Y.shape[0]
    laby                            =   np.ones((ny))
    lab                             =   np.concatenate((labx,laby),axis=0)
    data                            =   np.concatenate((X,Y),axis=0)

    clf                             =   sklearn.svm.SVC(kernel='linear',C=1000)
    clf.fit(data,lab)
    return  clf

def cloudpair__serparatinghyperplanenormal(X,Y):
    clf                             =   cloudpair__linearSVMclassifier(X,Y)
    v                               =   np.ndarray.flatten(clf)
    v                               =   vector__normalvector(v)
    return v

def testfun_keyorder(a = 0, b = 1):
    printval(a,"a")
    printval(b,"b")


################################################################################
#       TENSOR ------ TENSOR
################################################################################


def arcpath__gradnormdf(arcpath):

    pathl                           =   arcpath__logentrypathl(arcpath)
    nwul                            =   [logentrypath__P(path)['nwu'] for path in pathl]
    pathl                           =   ai_bi__asortedbyb(pathl,nwul)

    d                               =   dict(ih=[],ch=[],co=[],ho=[],bh=[],bo=[])
    for p                           in  range(hl(pathl)):
        N                           =   logentrypath__N(pathl[p])
        pscr                        =   N['pscr']
        ciSt,coSt,ccSt              =   P__ciSa_coSa_ccSa(N,platform='torch',pscr=pscr,basepss=None)
        if ciSt.shape[0]            >   10**4:
            I                       =   np.random.choice(ciSt.shape[0],10**5)
            ciSt                    =   ciSt[I,:]
            coSt                    =   coSt[I,:]
            ccSt                    =   ccSt[I,:]
        outputs                     =   N['model'].forward(ciSt,ccSt)
        criterion                   =   nn.MSELoss()
        loss                        =   criterion(outputs,coSt.view(-1))
        loss.backward()
        norml                       =   [float(torch.norm(x.grad).numpy()) for x in N['model'].parameters()]
        d['ih'].append(norml[0])
        d['ch'].append(norml[1])
        d['co'].append(norml[2])
        d['ho'].append(norml[3])
        d['bh'].append(norml[4])
        d['bo'].append(norml[5])
    df                      =   pd.DataFrame(d)
    return df

################################################################################
    #       ARRAY ------
################################################################################

def  a__pairwiseangle(a):
    a                           =   copy.deepcopy(a)
    a                           =   sklearn.preprocessing.normalize(a,norm='l2',axis=1)
    dotprod                     =   np.dot(a,  a.T)
    return  np.arccos(dotprod)


def a_lab0_lab1__jointmeanva(a,lab0,lab1):
        """
        Compute mean (within both conditions) vectors and store in an array

        :param a: mxn array whose rows represent points/vectors
        :param lab0: length m array of labels
        :param lab0: length m array of labels
        :return: array a  such that a[i,j] = mean over all vectors labeled i & j
        """

        vals1                   =   np.unique(lab0)
        vals2                   =   np.unique(lab1)
        dim0                    =   vals1.shape[0]
        dim1                    =   vals2.shape[0]
        dim2                    =   a.shape[1]
        M                       =   np.zeros((dim0,dim1,dim2))
        for p                   in  vals1:
            for q               in  vals2:
                I               =   [r for r in range(a.shape[0]) if (lab0[r]==p) and (lab1[r]==q)]
                M[p,q,:]        =   np.mean(a[I,:],axis=0)
        return  M

def N__hSmeanbyiod(N,lexorder='oi'):
    hSs                         =   N_pss__hSs(N,P__pss(N))
    labels                      =   null__labels()
    if lexorder                 ==  'oi':
        lab0                    =   labels[:,1]
        lab1                    =   labels[:,0]
    elif lexorder               ==  'io':
        lab0                    =   labels[:,0]
        lab1                    =   labels[:,1]
    return a_lab0_lab1__jointmeanva(hSs,lab0,lab1)

def N__hSmeanbyioddistmat(N,lexorder='oi',metric='angle'):
    a                           =   N__hSmeanbyiod(N,lexorder=lexorder)
    # a                           =   [a[p][p+1:] for p in range(a.shape[0])]
    # a                           =   np.concatenate(a)
    a                           =   a.reshape(-1,a.shape[-1])
    if metric                   ==  'euclidean':
        distmat                 =   sklearn.metrics.pairwise.euclidean_distances(a,a)
    else:
        distmat                 =   a__pairwiseangle(a)
    return distmat

# def N__plothSmeanbyioddistmat(N,lexorder='oi'):
#     distmat                     =   N__hSmeanbyioddistmat(N,lexorder=lexorder)
#     heatmap(distmat)

################################################################################
    #       PCD  ------
################################################################################

def pcd_basekeyorder__pcdop(pcd,basekeyorder=['CN','CP','WN','WP'],dropzero=False):
    """
    Apply orthogonal transformation to pcd, based on displacements of averages

    :param a:   a pcd (point cloud dictionary)
    :return:    a pcd with orthogonal projection onto R3
    """
    pcd                         =   copy.copy(pcd)
    ma                          =   [np.mean(pcd[key],axis=0) for key in basekeyorder]
    ma                          =   np.array(ma)
    basea                       =   (ma-ma[0])[1:]

    proja                       =   a__gsortha(basea)

    pcdop                       =   copy.copy(pcd)
    for key                     in  basekeyorder:
        translate               =   pcdop[key]-ma[0]
        pcdop[key]              =   np.matmul(translate,proja.T)
    return pcdop

def pcd__volume(pcd):
    keyl                        =   list(pcd.keys())
    pcdop                       =   pcd_basekeyorder__pcdop(pcd,basekeyorder=keyl)
    keys                        =   list(pcdop.keys())
    a                           =   [np.mean(pcdop[keys[p]],axis=0) for p in range(1,4)]
    a                           =   np.array(a)
    d                           =   np.linalg.det(a)
    return np.abs(d)/6

def pcd__pcdm(pcd):
    pcdm                        =   {}
    for key                     in  pcd:
        pcdm[key]               =   np.mean(pcd[key],axis=0)
    return pcdm

def pcdm__ie(pcdm):
    return pcdm["CN"]-pcdm["WN"]-pcdm["CP"]+pcdm["WP"]

def pcdm__a(pcdm):
    return np.array(list(pcdm.values()))

def pcdm__paraf(pcdm,gsp=True):
    # gsp ~ gram schmidt embed positive orthant
    a                           =   pcdm__a(pcdm)
    if gsp:
        a                       =   a__gramschmidtembed(a,center=True,posquad=True)
    base                        =   copy.deepcopy(a[[0]])
    a                           =   a-base
    aaa                         =   a[[0]]
    baa                         =   a[[1]]
    aba                         =   a[[2]]
    bbb                         =   a[[3]]
    aab                         =   bbb-aba-baa
    def f(x1,x2,x3):
        return base+x1*baa+x2*aba+x3*aab
    return f

def pcdm__paraa(pcdm,gsp=True):
    """
    :param pcdm: point cloud dictionary of mean activations
    :param gsp:  gram-schmidt projection: True or False
    :return:  an array whose rows are the relevant points of the associated
    parallelogram
    """
    f                           =   pcdm__paraf(pcdm,gsp=gsp)
    a                           =   []
    for x                       in  [0,1]:
        for y                   in  [0,1]:
            for z               in  [0,1]:
                a.append(f(x,y,z))
    return np.array(a)

def pcdm__paraplotlydata(pcdm):
    f                           =   pcdm__paraf(pcdm,gsp=True)
    data                        =   []
    for r                       in  [1,2,3]:
        for comb                in  itertools.combinations([0,1,2],r):
            for p               in  range(r):
                input           =   np.zeros((2,3),dtype=int)
                input[:,comb]   =   1
                input[0][comb[p]]   =   0
                x0              =   f(*input[0])
                x1              =   f(*input[1])
                a               =   np.concatenate((x0,x1),axis=0)
                trace           =   go.Scatter3d(x=a[:,0],y=a[:,1],z=a[:,2],showlegend=False,mode='lines',line=dict(color='black'))
                data.append(trace)
    return data

def pcdm__interheightovernormal(pcdm):
    """
    :param pcdm:
    :return: magnitude of the component of interaction term that is normal to
    the plane spanned by the "predictor" points
    """
    teta                        =   pcdm__a(pcdm)
    teta                        =   teta[[3,0,1,2]]
    return teta__height0overnorm(teta)

def pcdm__iev(pcdm):
    #   VERIFIED TO WORK CORRECTLY; SEE CHECK FUNCTION BELOW
    """
    :param pcdm:
    :return: component of interaction effect normal (vertical) to base
    """
    teta                =   pcdm__a(pcdm)
    return teta__normallegfromvert0tobase(teta)

def pcdm__ieh(pcdm):
    #   VERIFIED TO WORK CORRECTLY; SEE CHECK FUNCTION BELOW
    """
    :param pcdm:
    :return: component of interaction effect tangent (horizontal) to base
    """
    intereffvert        =   teta__normallegfromvert0tobase(pcdm__a(pcdm))
    intereff            =   pcdm__ie(pcdm)
    return intereff-intereffvert

def null__CHECK_ie_full_tangent_normal():
    N                   =   logentrypath__N('/Users/gh10/a/c/p/taskhr/data/archives/pub/ccn2019/final/noncatestrophic/l0s2co0cb0_15/20190606-052837-f3d5gsc25nwu42000pscr1-1')
    pcd                 =   N_i01o01__pcd(N,0,1,0,1)
    pcdm                =   pcd__pcdm(pcd)
    ieh                 =   pcdm__ieh(pcdm)
    iev                 =   pcdm__iev(pcdm)
    iew                 =   pcdm__ie(pcdm)
    if np.linalg.norm(iew-ieh-iev) > 0.000001:
        print("error: check algebraic decomposition")
    if np.inner(iev.reshape(-1),ieh.reshape(-1)) > 0.000001:
        print("error: check inner product")

################################################################################
    #       ------ ORTHOGONAL TRANSFORM
################################################################################


def a__gsortha(a):
    """
    Apply the Gram-Schmidt algorithm to rows of a

    :param a:   a 2d array
    :return:    orthogonalized version of a, with GS applied to each row in the
                standard order
    """
    #######     ATTEMPT 1
    # def gs_cofficient(v1, v2):
    #     return np.dot(v2, v1) / np.dot(v1, v1)
    #
    # # def multiply(cofficient, v):
    # #     return map((lambda x : x * cofficient), v)
    #
    # def proj(v1, v2):
    #     return gs_cofficient(v1, v2) * v1
    # Y = []
    # for i in range(a.shape[0]):
    #     temp_vec = a[i]
    #     for inY in Y :
    #         proj_vec = proj(inY, a[i])
    #         #print "i =", i, ", projection vector =", proj_vec
    #         temp_vec =  temp_vec-proj_vec #map(lambda x, y : x - y, temp_vec, proj_vec)
    #         #print "i =", i, ", temporary vector =", temp_vec
    #     Y.append(temp_vec)
    # Y = np.array(Y)
    # Y = sklearn.preprocessing.normalize(Y)
    # return Y
    #######     ATTEMPT 2
    # q,r                         =   np.linalg.qr(a.T)
    # return q.T
    #######     ATTEMPT 3
    conditionnumber             =   np.finfo(float).eps*np.max(a.shape)
    q,r                         =   np.linalg.qr(a.T)
    L                           =   [x for x in range(q.shape[1]) if np.abs(r[x][x]) <= conditionnumber]
    if len(L)                   >   0:
        print("warning: Gram-Schmit algorithm has cleared a row")
    q[:,L]                      =   0
    return q.T



def a__gramschmidtembed(a,center=True,posquad=True):
    """
    :param a: an array whose rows represent points
    :return:  an array whose rows represent the orthogonal projection of those
    points onto the span of the first three rows
    """
    a                           =   copy.deepcopy(a)
    if center:
        abase                   =   a[1:4]-a[[0]]
    else:
        abase                   =   a[:3]
    obase                       =   a__gsortha(abase)

    if center:
        a                       =   np.matmul(a-a[[0]],obase.T)
        if posquad:
            if a[1][0]          <   0:
                a[:,0]          =   -a[:,0]
            if a[2][1]          <   0:
                a[:,1]          =   -a[:,1]
            if a[3][2]          <   0:
                a[:,2]          =   -a[:,2]
    else:
        a                       =   np.matmul(a,obase.T)
        if posquad:
            if a[0][0]          <   0:
                a[:,0]          =   -a[:,0]
            if a[1][1]          <   0:
                a[:,1]          =   -a[:,1]
            if a[2][2]          <   0:
                a[:,2]          =   -a[:,2]
    return a

def a__orthc(a):
    """
    :param a:   an M x N matrix
    :return:    a matrix whose rows form an orthonormal basis for the orthogonal
                complement to the row space of a
    """
    aoT                         =   scipy.linalg.orth(a.T)
    X                           =   np.zeros((a.shape[1],a.shape[1]))
    X[:,:aoT.shape[1]]          =   aoT
    q,r                         =   np.linalg.qr(X)
    return q[:,aoT.shape[1]:].T
    # test 20190801:
    # a = np.array([[0,0,1]])
    # return 20190801:
    # array([[ 0.,  1., -0.],
    #        [-1.,  0.,  0.]])


def v_a__vorthpa(v,a):
    """
    orthogonally project the rows of v onto the row space of a
    :param v:   an M x N matrix, where M = # pts and N = # dimensions
    :param a:   a  K x N matrix
    :return:    the M x N matrix obtained by projecting each row of v onto the
                subspace spanned by the rows of a, orthogonally
    """
    aoT                         =   scipy.linalg.orth(a.T)
    v                           =   np.matmul(v,aoT)
    v                           =   np.matmul(v,aoT.T)
    return v

def v_a__vorthca(v,a):
    """
    orthogonally project the rows of v onto the orthogonal COMPLEMENT of the row
    space of a
    :param v:   an M x N matrix, where M = # pts and N = # dimensions
    :param a:   a  K x N matrix
    :return:    the M x N matrix obtained by projecting each row of v onto the
                orthogonal complement of the row space of a, orthogonally
    """
    return v - v_a__vorthpa(v,a)

def v_a__iscm(v,a): # magnitude of the in-space component
    """
    return the magnitude of the orthogonal projection of v onto the subspace
    spanned by the rows of a
    """
    v                           =   v.reshape((1,a.shape[1]))
    return np.linalg.norm(v_a__vorthpa(v,a))

def v_a__pscm(v,a): # magnitude of the perp-space component
    """
    return the magnitude of the orthogonal projection of v onto complement of
    the subspace spanned by the rows of a
    """
    v                           =   v.reshape((1,a.shape[1]))
    return np.linalg.norm(v_a__vorthca(v,a))

def v_a__iscm_CHECK():
    # 20190730: returns True, True
    a =         np.array([[0,0,1]])
    v =         np.array([[0,1,1]])
    print(v_a__iscm(v,a)==1.0)
    a =         np.array([[1,0,0],[0,1,0]])
    v =         np.array([[0,1,1]])
    print(v_a__iscm(v,a)==1.0)


################################################################################
    #       TETRAHEDRON --------
################################################################################

def teta__height0overnorm(teta):
    """
    :param teta: an array with four rows, each representing a vertex of a
    tetrahedron
    :return:  the height of the 0th vertex above the plane spanned by the others
    """
    teta                =   copy.deepcopy(teta)
    tetop               =   teta[[3,2,1,0]]  # place the important vertex last
    tetop               =   tetop-tetop[[0]] # place another at 0
    tetop               =   tetop[[1,2,3]]   # delete the 0
    q,r                 =   np.linalg.qr(tetop.T)  # qr decompose
    return np.abs(r[2][2])

def teta__minheightovernorm(teta):
    """
    :param teta: an array with four rows, each representing a vertex of a
    tetrahedron
    :return:  min height of any vertex above the plane spanned by its complement
    """
    teta                =   copy.deepcopy(teta)
    perml               =   [[0,1,2,3],[1,0,2,3],[2,0,1,3],[3,0,1,2]]
    return np.min([teta__height0overnorm(teta[perm]) for perm in perml])




def teta__normallegfromvert0tobase(teta):
    """
    :param teta: an array with four rows, each representing a vertex of a
    tetrahedron
    :return:  component of 1st vertex orthogonal to plane spanned by other three
    """

    teta                =   copy.deepcopy(teta)
    tetop               =   teta[[3,2,1,0]]  # place the important vertex last
    tetop               =   tetop-tetop[[0]] # place another vertex at 0
    tetop               =   tetop[[1,2,3]]   # delete the 0
    q,r                 =   np.linalg.qr(tetop.T)  # qr decompose
    return r[2][2]*q[:,2].reshape((-1)) # return just the 3rd orthogonal component




################################################################################
    #       ------ PLOTLY
################################################################################

def arcpath__timeseriesparaplots(arcpath,xploteqax=True,savepath=None,plotrange=None,ps=None):

    pathl                   =   arcpath__logentrypathl(arcpath)
    nwul                    =   [logentrypath__P(path)['nwu'] for path in pathl]
    pathl                   =   ai_bi__asortedbyb(pathl,nwul)

    if plotrange            is  None:
        plotrange           =   range(np.minimum(len(pathl),16))

    pathl                   =   [pathl[p] for p in plotrange]

    N                       =   logentrypath__N(pathl[0])

    if ps                   is  None:
        pss                 =   nd_cr__allcrps(N['nd'],[2])
        ps                  =   pss[0]


    pcdml                   =   [pcd__pcdm(N_i01o01__pcd(logentrypath__N(path),ps[0][0],ps[1][0],ps[0,1],ps[1,1])) for path in pathl]
    paraal                  =   [pcdm__paraa(pcdm) for pcdm in pcdml]
    pcl                     =   [pcdm__a(pcd__pcdm(N_i01o01__pcd(logentrypath__N(path),ps[0][0],ps[1][0],ps[0,1],ps[1,1]))) for path in pathl]

    pcgsl                   =   [a__gramschmidtembed(a) for a in pcl]
    axlimmax                =   np.max(np.concatenate(paraal))
    axlimmin                =   np.min(np.concatenate(paraal))
    axrange                 =   [axlimmin,axlimmax]

    nrow                    =   3
    ncol                    =   int(np.ceil(len(pathl)/3))
    fig = tools.make_subplots(  rows            =   nrow,
                                cols            =   ncol,
                                # subplot_titles  =   [str(p) for p in plotrange],
                                specs           =   [[{'is_3d':True}]*ncol]*nrow
                                )
    fig['layout'].update(title=os.path.basename(arcpath) +' // ' + os.path.basename(pathl[0]))

    for p                   in  range(hl(pathl)):
        row                 =   (p // ncol)+1
        col                 =   (p %  ncol)+1
        teta                =   pcgsl[p]
        fig.append_trace(dict(  type='scatter3d',
                                x = teta[:,0][:2],
                                y = teta[:,1][:2],
                                z = teta[:,2][:2],
                                text = ['A1','A2'],
                                mode = 'markers',
                                marker=dict(color=['orange','black'])),
                                row,
                                col)
        fig.append_trace(dict(  type='scatter3d',
                                x = teta[:,0][2:],
                                y = teta[:,1][2:],
                                z = teta[:,2][2:],
                                text = ['B1','B2'],
                                mode = 'markers',
                                marker=dict(cmin=0,cmax=4,color=['orange','black'],symbol='circle-open',)),
                                row,
                                col)
        pcdm                =   pcd__pcdm(N_i01o01__pcd(logentrypath__N(pathl[p]),ps[0][0],ps[1][0],ps[0,1],ps[1,1]))
        for trace           in  pcdm__paraplotlydata(pcdm):
            fig.append_trace(trace,row,col)

    def p__scenesuffix(p):
        if p                ==  0:
            return ''
        else:
            return str(p+1)
    for p                   in  range(hl(pathl)):
        suffix              =   p__scenesuffix(p)
        if not xploteqax:
            axlimmax                =   np.max(paraal[p])
            axlimmin                =   np.min(paraal[p])
            axrange                 =   [axlimmin,axlimmax]
        fig['layout']['scene'+suffix]['xaxis'].update(range=axrange)
        fig['layout']['scene'+suffix]['yaxis'].update(range=axrange)
        fig['layout']['scene'+suffix]['zaxis'].update(range=axrange)

        fig['layout']['scene'+suffix].update(aspectmode='cube')
    if savepath                 ==  None:
        po.plot(fig)
    else:
        po.plot(fig,filename=savepath)
    return fig

def pcdop__tracel(pcdop,showcloud=False):
    data                        =   []
    keys                        =   [key for key in pcdop.keys()]
    colors                      =   ['red','blue','green','grey']
    for p                       in  range(4):
        key                     =   keys[p]
        color                   =   colors[p]
        a                       =   pcdop[key]
        am                      =   np.mean(a,axis=0)
        trace                   =   go.Scatter3d(   x   =   a[:,0],
                                                    y   =   a[:,1],
                                                    z   =   a[:,2],
                                                    mode    = 'markers',
                                                    marker  = dict(color=color),
                                                    name=   key
                                                    )
        tracem                  =   go.Scatter3d(   x   =   [am[0]],
                                                    y   =   [am[1]],
                                                    z   =   [am[2]],
                                                    mode    = 'markers',
                                                    marker  =   dict(
                                                                color   =   color,
                                                                size    =   13,
                                                                line    =   dict(color  =   'black', width =18)
                                                                    ),
                                                    name    =   key+' average'
                                                )
        if showcloud:
            data.append(trace)
        data.append(tracem)
    return data

def N__layout(N):
    titlestring                 =   'pcr = '    + str(N['pscr']) + '  ' + \
                                'gsc = '    + str(N['gsc'])  + '  ' + \
                                'date = '   + str(N['traindate'])
    layout                      =   go.Layout(title= titlestring)
    return layout

def N_if_pscr__plotregressorembeddingofhiddenreps(N,f,pscr):

    pss,hSss,regressors,fiSs = N_if_pscr__pss_hSss_regressors_fiSs(N,f,pscr)
    r = np.array(regressors)
    data                        =   list()
    pcl                         =   list()
    for p                       in  range(3):
        x                       =   np.matmul(hSss[p],regressors[0])
        y                       =   np.matmul(hSss[p],regressors[1])
        z                       =   np.matmul(hSss[p],regressors[2])
        pcl.append(np.array([x,y,z]))
        trace                   =   go.Scatter3d(x=x,y=y,z=z,mode='markers')
        data.append(trace)

    plotly.offline.plot({"data":data, "layout":N__layout(N)})
    return pcl

def N__plothiddenpcafortransitionfromfirsttosecondcanonicalsignal(N):
    layout                      =   N__layout(N)
    ciSs                        =   P__ciSs(N)
    iS                          =   ciSs[0]
    iSs                         =   np.tile(vec__rowvec(iS),(101,1))
    iSs                         =   iSs.astype(float)
    iSs[:,0]                    =   np.arange(0,1.01,0.01)
    iSs[:,1]                    =   1-iSs[:,0]

    cSs                         =   []
    hSs                         =   []
    oSs                         =   []
    for p                       in  range(5):
        cSs_temp                =   P_ps_nrows__cSs(N,vec__rowvec(np.array([0,p])),nrows=101)
        oSs_temp,hSs_temp       =   N['model'].predict([iSs,cSs_temp])
        cSs.append(cSs_temp)
        hSs.append(hSs_temp)
        oSs.append(oSs_temp)

    pca                         =   []
    for p                       in  range(5):
        pcat                    =   PCA(n_components=10)
        pcat.fit(hSs[p])
        pca.append(pcat)

    pcA                         =   PCA(n_components=10)
    pcA.fit(np.vstack(hSs[:3]))
    Y = pcA.transform(np.vstack(hSs[:3]))
    Y = Y[:,:3]
    x = Y[:,0]
    y = Y[:,1]
    z = Y[:,2]
    trace                   =   go.Scatter3d(x=x,y=y,z=z)
    plotly.offline.plot({"data":[trace], "layout":layout})

def a__sorted1d(a,samplerate=None):
    b                           =   copy.copy(a)
    b                           =   np.ndarray.flatten(b)
    b                           =   np.sort(b)
    return b

def a__valplot(a,title=''):
    b                           =   a__sorted1d(a)
    L                           =   hl(b)
    if hl(b)                    >   200:
        samplerate              =   hl(b)//200
        b                       =   b[::samplerate]
    else:
        samplerate              =   1
    x_y__plotlyscatter(x=np.arange(0,L,samplerate),y=b,title=title)


def x_y__plotlytrace(   x       =   'default',
                        y       =   np.random.rand(10),
                        title   =   None):

    y                           =   array__rowstackvec(y)
    if (type(x) == str) and  x  ==  'default':
        x                       =   np.arange(hl(y))
    x                           =   array__rowstackvec(x)
    if title                    ==  None:
        trace                   =   go.Scatter(x=x,y=y,mode= 'markers')
    else:
        trace                   =   go.Scatter(x=x,y=y,mode= 'markers',name=title)
    return trace

def x_y__plotlyscatter( x       =   'default',
                        y       =   np.random.rand(10),
                        title   =   '',
                        layout  =   None,
                        savepath=   'plotly_temp_plot.html'):

    y                           =   array__rowstackvec(y)
    if (type(x)==str)  and   x  ==  'default':
        x                       =   np.arange(hl(y))
    x                           =   array__rowstackvec(x)
    trace                       =   go.Scatter(x=x,y=y,mode= 'markers')
    if layout                   ==  None:
        layout                  =   go.Layout(title= title)
    p                           =   plotly.offline.plot({"data":[trace], "layout":layout},filename=savepath)
    return p

def x_yl__plotlyscatter(x       =   'default',
                        yl      =   [np.random.rand(10)],
                        tnames  =   None,
                        title   =   '',
                        layout  =   None,
                        savepath=   'plotly_temp_plot.html'):
    ntr                         =   len(yl)
    yl                          =   copy.deepcopy(yl)
    for p                       in  range(len(yl)):
        yl[p]                   =   array__rowstackvec(yl[p])
    if x                        ==  'default':
        x                       =   np.arange(hl(yl[0]))
    x                           =   array__rowstackvec(x)
    if tnames                   ==  None:
        tracel                      =   [go.Scatter(x=x,y=yl[p],mode='markers')  for p in range(ntr)]
    else:
        tracel                      =   [go.Scatter(x=x,y=yl[p],mode='markers',name=tnames[p])  for p in range(ntr)]
    if layout                   ==  None:
        layout                  =   go.Layout(title= title)
    p                           =   plotly.offline.plot({"data":tracel, "layout":layout},filename=savepath)
    return p

def xl_yll__plotlyscatter(xl     =   'default',
                        yll     =   [[np.random.rand(10)]],
                        tnames  =   None,
                        title   =   '',
                        layout  =   None,
                        savepath=   'plotly_temp_plot.html'):
    ntr                         =   len(yll)
    yll                         =   copy.deepcopy(yll)
    nx                          =   hl(yll[0][0])
    if xl                       ==  'default':
        xl                      =   [np.tile(np.arange(nx),len(l)) for l  in yll]
    yl                          =   [array__rowstackvec(np.concatenate(l)) for l  in yll]

    if tnames                   ==  None:
        tracel                      =   [go.Scatter(x=xl[p],y=yl[p],mode='markers')  for p in range(ntr)]
    else:
        tracel                      =   [go.Scatter(x=xl[p],y=yl[p],mode='markers',name=tnames[p])  for p in range(ntr)]
    if layout                   ==  None:
        layout                  =   go.Layout(title= title)
    p                           =   plotly.offline.plot({"data":tracel, "layout":layout},filename=savepath)
    return p

def x_y__plotlylinescatter(x=None,y=None,title=''):
    if x                        is  None:
        x                       =   np.arange(hl(y))
    if y                        is  None:
        y                       =   np.arange(hl(x))
    po.plot({'data':[go.Scatter(x=x,y=y)],'layout':go.Layout(title=title)})


def array__plotsortedcurve(array):
    s                           =   array__rowstackvec(array)
    s                           =   np.sort(s)
    m                           =   heightVlength(s)//1000
    x_y__plotlyscatter(y=s[::m])


def namelist__mseparascoretrace(  namelist,
                                    nsamplepernet   =   10,
                                    title           =   None,
                                    metric          =   'volume'):
    parascorea              =   namel__parascorea(      namelist,
                                                        nsamplepernet=nsamplepernet,
                                                        metric=metric)
    msea                    =   namel__msea(namelist,score='mse_pscr5-5')

    if title                ==  None:
        trace               =   x_y__plotlytrace(x=parascorea,y=msea)
    else:
        trace               =   x_y__plotlytrace(x=parascorea,y=msea,title=title)
    return  trace

def namell__plotparavmse(namell,titlel=None,metric='euclidean',savepath=None):
    pathll                  =   [[archivedir()+'/'+name for name in L] for L in namell]
    pathll__plotparavmse(pathll,titlel=titlel,metric=metric,savepath=savepath)

def pathll__plotparavmse(pathll,titlel=None,metric='euclidean',savepath=None):

    tracec                  =   len(pathll)
    mseal                   =   [pathl__msea(L) for L in pathll]
    parascoreal             =   [pathl__parascorea(L,metric=metric) for L in pathll]
    if not titlel           ==  None:
        data                    =   [x_y__plotlytrace(x=parascoreal[p],y=mseal[p],title=titlel[p]) for p in range(tracec)]
    else:
        data                    =   [x_y__plotlytrace(x=parascoreal[p],y=mseal[p]) for p in range(tracec)]

    pearsonr,pval           =   scipy.stats.pearsonr(np.concatenate(mseal),np.concatenate(parascoreal))


    layout      =   layout= go.Layout(
        xaxis= dict(
            title= 'Parallelism Score  // '+metric + '<br> pearson r '+ str(pearsonr)+ '     p = '+str(pval)
        ),
        yaxis=dict(
            title= 'Multitasking MSE',
        ))
    if savepath             ==  None:
        plotly.offline.plot({"data":data,"layout":layout})
    else:
        plotly.offline.plot({"data":data,"layout":layout},filename=savepath)

################################################################################
    # PARAM ------ PARAM
################################################################################

def P_Q__diffkeys(P,Q):
    L                           =   np.concatenate((list(P.keys()),list(Q.keys())))
    L                           =   np.unqiue(L)
    for x                       in  L:
        iseq                    =   True
        a                       =   x in P.keys()
        b                       =   x in Q.keys()
        c                       =   not x == 'model'
        # if a and b and c:
        #     if type(P[x])       ==  np.ndarray:

def P__fvectors(P,stacked=True):
    a                           =   copy.deepcopy(P['iSg'])
    if stacked:
        L                       =   [a[:,intervals.itv(p,P['nf'])] for  p in range(P['nd'])]
        a                       =   np.concatenate(L,axis=0)
    return a

# idc = 'input data cardinality' = # rows in the matrix of (training) input
# singals passed by the user
def P__idc(P):
    return (P['nf'] ** P['nd']) * P__pssc(P)

def idc_mbc_drop__nwupe(idc,mbc,drop):
  if drop:
      nwupe                     =   idc // mbc
  else:
      nwupe                     =   int(np.ceil(idc / mbc))
  return nwupe

def idc_mbc_nnwu_drop__ne(idc,mbc,nnwu,drop):
    nwupe                       =   idc_mbc_drop__nwupe(idc,mbc,drop)
    ne                          =   int(np.ceil(nnwu / nwupe))
    return ne

def nd_nf__uniformrandomiSg(nd,nf,mindist=None):
    iSg                         =   np.zeros((nf,nf*nd))
    for p                       in  range(nd):
        a                       =   np.random.rand(nf,nf)
        if mindist              ==  None:
            iSg[:,intervals.itv(p,nf)]  =   a
        else:
            while np.min(scipy.spatial.distance.pdist(a)) < mindist:
                a               =   np.random.rand(nf,nf)
            iSg[:,intervals.itv(p,nf)]  =   a
    return iSg

def null__standardP(platform = 'torch'):
    if platform                 in  ['torch','autodiff']:

        ### ABBREVIATIONS
        #   mb                  minibatch
        #   wu                  weight update
        #   id                  input data, regarded as a list of training pairs
        #   n<X>                number of X's
        #   <X>c                cardinality of (i.e. # of elements in) X

        ### SET QUANTITIES
        #   nwu                 number of weight updates AKA number of mini-batches
        #   idc                 number of training p
        #   mbc                 minibatch cardinality
        #   smbc                sum of minibatch cardinalities
        #   dropincompletemb    if True, and if mbc does not divide idc, then drop the
        #                           last minibatch
        #   lrp                 learn rate period

        ### DERIVED QUANTITIES
        #   nwupe               number of weight updates per epoch; conditional:
        #                           if dropincompletemb:
        #                               nwupe = idc // mbc
        #                           else:
        #                               nwupe = int(np.ceil(idc / mbc))
        #   ne                  number of epochs:
        #                           ne = int(np.ceil(nwu / nwupe))

        P                           =   {}

        P['dropincompletemb']       =   False
        P['trainorderseed']         =   time32bit()
        P['conserve']               =   True
        P['replace']                =   False
        P['shuffle']                =   True
        P['rand']                   =   True

        P['pscr']                   =   [1]
        P['gsc']                    =   25
        P['nf']                     =   3
        P['nd']                     =   5
        P['nh']                     =   200
        P['gs']                     =   tsc_nd__randts(P['gsc'],P['nd'])

        P['mindist']                =   None
        P['iSg']                    =   nd_nf__uniformrandomiSg(P['nd'],P['nf'],mindist=P['mindist'])  # iSg  = input signal generator
        P['iSgm']                   =   'uniformrandom_boundeddistance'  # iSgm = input signal generator method
        P['transformtype']          =   'classification'
        P['sta']                    =   np.zeros((P['nf'],P['nf']*P['nd'])) # STa = signal transform array; in a 'classifier-type' network, you multiply a feature row-vector (on the left) with a sub-array of this (on the right) to get the desired output feature vector
        for p                       in  range(P['nd']):
            ran                     =   intervals.itv(p,P['nf'])
            P['sta'][:,ran]         =   np.linalg.inv(P['iSg'][:,ran])

        P['nwu']                    =   3*10**4
        P['lra']                    =   100
        P['lrb']                    =   20
        P['lrf']                    =   'inverse'
        P['lrs']                    =   (P['lrb']/P['lra'])**(1/P['nwu'])
        P['lrp']                    =   copy.copy(P['nwu'])
        P['idc']                    =   P__nfv(P) * P__pssc(P) # must follow nd, nf, gsc, gs
        P['mbc']                    =   100
        P['mbi']                    =   -1
        P['nwupe']                  =   idc_mbc_drop__nwupe(    P['idc'],
                                                                P['mbc'],
                                                                P['dropincompletemb'])
        P['l2p']                    =   0   # l2 penalty for weight decay
        P['l1p']                    =   0   # l1 penalty for weight decay
        P['l2pha']                  =   0   # l2 penalty for hidden activations
        P['l1pha']                  =   0   # l1 penalty for hidden activations
        P['ifo']                    =   0   # input feature offset

        P['hawmin']                 =   -0.02
        P['hawmax']                 =   0.02
        P['hawsee']                 =   time32bit()
        P['hawbia']                 =   -2
        P['hawbia_reqgrad']         =   True
        P['oawmin']                 =   -0.02
        P['oawmax']                 =   0.02
        P['oawsee']                 =   time32bit()
        P['oawbia']                 =   -2
        P['oawbia_reqgrad']         =   True

        P['msemax']                 =   0.01
        P['momentum']               =   0
        P['optimizer']              =   'sgd'       # keras method of optimization;
                                                    # stands for stochsatic gradient
                                                    # descent

        P['history_timestamps']     =   np.zeros((0))
        P['history_loss']           =   np.zeros((0))
        P['history_gradnorm']       =   np.zeros((0))

        P['platform']               =   'torch'

    else:
        nd                          =   5
        gsc                         =   25

        P                           =   {}
        P['nf']                     =   3
        P['nd']                     =   5
        P['nh']                     =   200
        P['gs']                     =   tsc_nd__randts(gsc,nd)
        P['gsc']                    =   gsc
        P['pscr']                   =   [1]
        P['rand']                   =   True
        P['conserve']               =   True
        P['replace']                =   False
        P['nwu']                    =   3*10**5
        P['nepochs']                =   P['nwu']
        P['stepsperepoch']          =   1
        P['trainorderseed']         =   time32bit()
        P['learnrate']              =   50
        P['decay']                  =   0.0001
        P['msemax']                 =   0.01
        P['hawmin']                 =   -0.02
        P['hawmax']                 =   0.02
        P['hawsee']                 =   time32bit()
        P['hawbia']                 =   -2
        P['oawmin']                 =   -0.02
        P['oawmax']                 =   0.02
        P['oawsee']                 =   time32bit()
        P['oawbia']                 =   -2
        P['mindelta']               =   -1
        P['clipnorm']               =   1
        P['momentum']               =   0
        P['baseline']               =   0.01
        P['optimizer']              =   'sgd'       # keras method of optimization;
                                                    # stands for stochsatic gradient
                                                    # descent
        P['keephistory']            =   False

    return P


#   DEPENDENCY CHART:  (LEFT TO RIGHT)
#   nd      |
#   gs      |----> pss ---> pssc ----> idc ----
#   gsc     |                         /   \    \
#   pscr    |                        /     \    \
#                                   /       \    \
#                                  /         \    \
#   nf      ----------------------/           \    \
#                                              \    \
#   nwu     ---------------------------------> ne    \
#                                             /       \
#   mbc     ---------------------------------/ ----- nwupe

def P__standardextension(P,platform='torch'):
    if platform                     ==  'torch':

    ######### INITIAL SUBSTITUTOINS
        Q                           =   null__standardP()
        for key                     in  P.keys():
            Q[key]                  =   P[key]

    ######### CACULATE GROUND SET
        ndi                         =   'nd' in P.keys()
        nfi                         =   'nf' in P.keys()
        gsi                         =   'gs' in P.keys()

        #   if P has a nonstandard number of dimensions or feature units per
        #   dimension and the ground set is not specified, we must automatically
        #   generate a customized ground set
        if (ndi or nfi) and (not gsi):
            if not ('gsc' in P.keys()):
                Q['gsc']            =   Q['nd']**2
            Q['gs']                 =   tsc_nd__randts(Q['gsc'],Q['nd'])

        #   if Q['gsc'] does not equal the number of tasks in the ground set, then
        #   something is wrong; we will
        if not (heightVlength(Q['gs']) == Q['gsc']):
            print('NB: the dictionary value for <gsc> does not equal lengh(<gs>)')
            print('    generating a random ground set')
            Q['gs']                 =   tsc_nd__randts(Q['gsc'],Q['nd'])

    ########## MAKE LEARNING RATES CONSISTENT
        if not 'lrs'                in  P.keys():
            Q['lrs']                =   None

        if 'nwu'                    in  P.keys():
            if not 'lrp'            in  P.keys():
                if P['nwu']         >   1:  # <-- if nwu < 2 then learning rate will come out either 0 or negative
                    Q['lrp']        =   copy.copy(P['nwu'])
                else:
                    Q['lrp']        =   10000

    ######### REMAINING SUBSTITUTIONS

        Q['idc']                    =   P__idc(Q)
        Q['pssc']                   =   P__pssc(Q)
        Q['nwupe']                  =   idc_mbc_drop__nwupe(    Q['idc'],
                                                                Q['mbc'],
                                                                Q['dropincompletemb'])
        if 'iSgm' in P.keys():
            if P['iSgm']            ==  'uniformrandom':
                if not 'iSg' in P.keys():
                    Q['iSg']        =   np.random.rand(Q['nf'],Q['nd']*Q['nf'])
                elif ('iSg' in P.keys()) and (not list(Q['iSg'].shape) == [Q['nf'],Q['nf']*Q['nd']]):
                    print("rewriting iSg due to size incompatiblity")
                    Q['iSg']        =   nd_nf__uniformrandomiSg(Q['nd'],Q['nf'])
            elif P['iSgm']          ==  'onehot':
                if not 'iSg' in P.keys():
                    Q['iSg']        =   np.tile(np.eye(Q['nf']),(1,Q['nd']))
                elif ('iSg' in P.keys()) and (not list(P['iSg'].shape) == [Q['nf'],Q['nf']*Q['nd']]):
                    print("rewriting iSg due to size incompatiblity")
                    Q['iSg']        =   np.tile(np.eye(Q['nf']),(1,Q['nd']))
        if Q['transformtype']       ==  'classification':
            Q['sta']                =   np.zeros(Q['iSg'].shape)
            for p                   in  range(Q['nd']):
                ran                 =   intervals.itv(p,Q['nf'])
                Q['sta'][:,ran]     =   np.linalg.inv(Q['iSg'][:,ran])

    elif platform                   ==  'keras':

        Q                           =   null__standardP()
        for key                     in  P.keys():
            Q[key]                  =   P[key]

        if not ('nepochs' in P.keys()):
            Q['nepochs']            =   Q['nwu']

        a                           =   'nd' in P.keys()
        b                           =   'nf' in P.keys()
        c                           =   'gs' in P.keys()

        #   if P has a nonstandard number of dimensions or feature units per
        #   dimension and the ground set is not specified, we must automatically
        #   generate a customized ground set
        if (a or b) and (not c):
            if not ('gsc' in P.keys()):
                Q['gsc']            =   Q['nd']**2
            Q['gs']                 =   tsc_nd__randts(Q['gsc'],Q['nd'])

        #   if Q['gsc'] does not equal the number of tasks in the ground set, then
        #   something is wrong; we will
        if not (heightVlength(Q['gs']) == Q['gsc']):
            print('NB: the dictionary value for <gsc> does not equal lengh(<gs>)')
            print('    generating a random ground set')
            Q['gs']                 =   tsc_nd__randts(Q['gsc'],Q['nd'])

        #   NB  We must ask whether <stepsperepoch> is in P.keys(), since it is
        #       always in Q.keys() by default.
        if not 'stepsperepoch' in P.keys():
            if Q['nepochs']         ==  0:
                Q['stepsperepoch']  =   0
            else:
                Q['stepsperepoch']  =   Q['nwu'] // Q['nepochs']

    return Q

# def null__standardP():
#     P                               =   nf_nd_nh_gs_pscr__taskparameters()
#     P_rand_conserve_replace_nwu_trainorderseed__appendtraingenparameters(P)
#     P_learnrate_msemax_iwmin_iwmax_iwhseed_iwoseed__appendnetworkparameters(P)
#     return P

def Pkeys__standardextension(**kwargs):
    return P__standardextension(kwargs)

# # generate a sequence of length <nwu> whose entries are drawn from the set
# # {0,..., capval-1}.  if conserve == true then each element of that set appears
# # exactly p times in the first p * capval entries of the sequence ... for every
# # p <= nwu // capval; the entries coming after capval*(nwu // capval) are
# # sampled uniformly at random from {0, ..., capval-1}.
# # if replace == true then the elements are drawn with replacement.  if both
# # replace and conserve are false, then the entries of the sequence are sampled
# # uniformly at random from the set
# def P_conserve_replace__Pappendsamplesequence(
#     P,conserve= True,replace = False):
#     seq                             =   length_capval_conserve_replace__randomsequence(
#                                         length      =   P['nwu'],
#                                         capval      =   (P['nf'] ** P['nd']) * P['pssc'],
#                                         conserve    =   conserve,
#                                         replace     =   replace)
#     P['samplesequence']             =   seq
#
# def nf_nd_gsc_pscr_conserve_replace_nwu__networkparams(nf,nd,gsc,pscr,conserve=True,replace=False,nwu=100000):
#     P                               =   {}
#     P__Pappendstandardparameters(P)
#     P['nwu']                      =   nwu
#     P_nf_nd_gsc_pscr__Pappendtaskparameters(P,nf,nd,gsc,pscr)
#     P_conserve_replace__Pappendsamplesequence(P,conserve=conserve,replace=replace)
#     return P


def P__justify(P,verbose=True,correct=False):
    if verbose:
        print('making a deep copy')
    P                           =   copy.deepcopy(P)
    if ('iSg' in P.keys()) and (not P['iSg'].shape == (P['nf'],P['nf']*P['nd'])):
        if verbose:
            print('P[\'iSg\'] has incompatible shape')
        if correct:
            if verbose:
                print(' .. replacing with uniform random generator')
            P['iSg']            =   np.random.rand(P['nf'],  P['nf']*P['nd'])


    if not 'transformtype'      in  P.keys():
        if verbose:
            print('transformtype not specified')
        if correct:
            print("setting to \'facsimile\'")
            P['transformtype']      =   'facsimile'
    if P['transformtype']=='facsimile':
        if 'sta' in P.keys():
            if verbose:
                print("keyword \'STA\' should be deleted, as the transform type is facsimile")
            if correct:
                print("deleting")
                del P['sta']
    elif P['transformtype']         ==  'classification':
        if (not 'sta' in P.keys) or (not type(P['sta'])==np.ndarray) or (not P['sta'].shape == (P['nf'],P['nf']*P['nd'])):
            if verbose:
                print("keyword \'STa\' either does not exist or has the wrong shape")
            if correct:
                print("correcting")
                STa                 =   np.zeros(P['nf'],P['nd']*P['nf'])
                for p               in  range(P['nd']):
                    ran             =   intervals.itv(p,P['nf'])
                    STa[:,ran]      =   np.linalg.inv(P['iSg'][:,ran])
                P['sta']            =   STa


    if not (P['gsc'] == P['gs'].shape[0]):
        if verbose:
            print('P[\'gsc\'] is incorrect')
        if correct:
            if verbose:
                print(' .. replacing with correct gsc')
            P['gsc']            =   P['gs'].shape[0]
    if np.max(P['pscr']) > P['nd']:
        if verbose:
            print('P[\'pscr\'] goes too high')
        if correct:
            if verbose:
                print(' .. removing incompatible elements from range')
            P['pscr']            =   [x for x in P['pscr'] if x <= P['nd']]
    if not (P['idc'] == P__idc(P)):
        if verbose:
            print('P[\'idc\'] is incorrect')
        if correct:
            if verbose:
                print(' .. replacing with correct idc')
            P['idc']            =   P__idc(P)
    if (not 'pssc' in P.keys()) or (not (P['pssc'] == P__pssc(P))):
        if verbose:
            print('P[\'pssc\'] is incorrect')
        if correct:
            if verbose:
                print(' .. replacing with correct pssc')
            P['pssc']           =   P__pssc(P)

    nwupe                       =   idc_mbc_drop__nwupe(    P['idc'],
                                                            P['mbc'],
                                                            P['dropincompletemb'])
    if not (P['nwupe'] == nwupe):
        if verbose:
            print('P[\'nwupe\'] is incorrect')
        if correct:
            if verbose:
                print(' .. replacing with correct nwupe')
            P['nwupe']            =   nwupe
    if verbose:
        print('check complete')
    if correct:
        return P

################################################################################
    # PYTORCH MODEL -----
################################################################################


def M__gradnorm(M):
    l                               =   [torch.norm(x.grad) for x in M.parameters() if x.requires_grad]
    t                               =   np.sum([t**2 for t in l]) ** (1/2)
    return t

################################################################################
    # ----- INPUT SIGNAL
################################################################################


def rangelim_combsize__rowsascombs(rangelim,combsize):
    return np.array(list(itertools.combinations(range(rangelim),combsize)))

def nf_nd__allonehotvecs(nf,nd):
    compressedform                  =   itertools.product(range(nf),repeat=nd)
    compressedform                  =   np.array(list(compressedform))
    nrows                           =   compressedform.shape[0]
    ncols                           =   nf * nd
    z                               =   np.zeros((nrows,ncols),dtype=int)
    for p                           in  range(nd):
        offset                      =   p * nf
        compressedform[:,p]         =   compressedform[:,p]+offset
    for p                           in  range(nrows):
        z[p][compressedform[p]]     =   1
    return z

def check_nf_nd__allonehotvecs():
    nf                              =   3
    nd                              =   5
    A                               =   nf_nd__allonehotvecs(nf,nd)

    nrows                           =   A.shape[0]
    ncols                           =   nd
    B                               =   np.zeros((nrows,ncols),dtype=int)
    for p                           in  range(nrows):
        B[p,:]                      =   np.nonzero(A[p])[0]
    for p                           in  range(nd):
        offset                      =   p * nf
        B[:,p]                      =   B[:,p]-offset

    C                               =   itertools.product(range(nf),repeat=nd)
    C                               =   np.array(list(C))

    if np.array_equal(B,C):
        print("test passed")
        return np.array([])
    else:
        print("test failed")
        return A,B,C

def generator_iSsshape_cSsshape__keraspredictioninput(gen,iSsshape,cSsshape):
    iSs                             =   np.zeros(iSsshape)
    cSs                             =   np.zeros(cSsshape)

    counter                         =   -1
    for x                           in  gen:
        counter                     =   counter+1
        iSs[counter]                =   x[0][0]
        cSs[counter]                =   x[0][1]
    return [iSs,cSs]


################################################################################
#   -----  LABELS
################################################################################


def null__labels(subsample=None,format=None):
    if format                       ==  None:
        if subsample            ==  None:
            P                       =   Pkeys__standardextension(nd=5,nf=3)
            ciSs                    =   P__ciSs(P,justify=True)
            pss                     =   P__pss(P)
            pSs,iSs1                =   a_b__atensorbbyrow(pss,ciSs)
            idims                   =   np.array([x[0][0] for x in pSs])
            odims                   =   np.array([x[0][1] for x in pSs])
            fvals                   =   [np.argmax(iSs1[p,intervals.itv(idims[p],P['nf'])]) for p in range(hl(iSs1))]
            labtup                  =   (idims,odims,fvals)
            labs                    =   np.transpose(np.vstack(labtup))
            return labs
        elif not subsample % 75     ==  0:
            print("error: subsample must evenly dividisible by 75")
        else:
            a                       =   np.arange(5)
            b                       =   np.arange(3)
            labssa                  =   a_b__atensorbbyrow(a,a)
            labssa                  =   np.transpose(np.array(labssa))
            labssa                  =   a_b__atensorbbyrow(labssa,b)
            x                       =   labssa[0]
            y                       =   labssa[1]
            y                       =   np.reshape(y,(hl(y),1))
            labssa                  =   np.concatenate((x,y),axis=1)
            labssa                  =   np.repeat(labssa,subsample//75,axis=0)
            return labssa
    elif format                     ==  'text':
        labs                        =   null__labels(subsample=subsample)
        labs                        =   [str(x[0])+str(x[1])+str(x[2]) for x in  labs]
        return labs


################################################################################
#   GENERATOR  -----
################################################################################


def traininggenerator__cSL0norms(traininggenerator):
    L                               =   []
    for p                           in  traininggenerator:
        c                           =   np.sum(p[0][1])
        if not (c in L):
            L.append(c)
    print("WARNING: THE GENERAOR IS NOW EXPENDED")
    return L

def traininggenerator__iSs_cSs_oSs(gen):
    iSs                             =   []
    cSs                             =   []
    oSs                             =   []
    for p                           in  gen:
        iSs.append(vec__rowvec(p[0][0]))
        cSs.append(vec__rowvec(p[0][1]))
        oSs.append(vec__rowvec(p[1]))

    iSs                             =   np.concatenate(iSs,axis=0)
    cSs                             =   np.concatenate(cSs,axis=0)
    oSs                             =   np.concatenate(oSs,axis=0)

    return iSs,cSs,oSs


################################################################################
# ----- GENERATOR
################################################################################


# pscr := performance set cardinality range
# seq: = a sequence of integers specifying the order in which training points
# are sampled
# def nf_nd_gs_pscr_rand_replace_conserve_nwu__generate_iS_oS_cv_seq(
#     nf,nd,gs,pscr,rand,replace,conserve,nwu):
#
#     iSs                                 =   nf_nd__allonehotvecs(nf,nd)
#     pss                                 =   edges_cran__allcardinalitycranpartialmatching(gs,pscr)

# seq: = a sequence of integers specifying the order in which training points
# are sampled
def nf_nd_iSs_pss_seq__generate_iS_oS_cv(nf,nd,iSs,pss,seq):
    for p                               in  seq:
        ps,iS                           =   a_b_int__rowofarowtensorb(
                                            pss,iSs,p)
        oS                              =   ps_nf_iS_fun__oS(ps,nf,iS)
        cv                              =   ts_nd__cS(ps,nd)
        yield iS,oS,cv

#############################################################################################


def P_pss__atomictestinggenerator(P,pss):
    nf                                  =   P['nf']
    nd                                  =   P['nd']
    iSs                                 =   P__ciSs(P)
    nsteps                              =   hl(pss) * hl(iSs)
    print("number of iterations in this generator: " + str(nsteps))
    seq                                 =   np.random.permutation(nsteps)
    gen                                 =   nf_nd_iSs_pss_seq__generate_iS_oS_cv(nf,nd,iSs,pss,seq)
    return gen

def P_pss__testinggenerator(P,pss):
    gen0                                =   P_pss__atomictestinggenerator(P,pss)
    for p                               in  gen0:
        yield [np.array([p[0]]),np.array([p[2]])], np.array([p[1]])

def P_pscr__testinggenerator(P,pscr):
    print('this function, P_pscr__testinggenerator, has not been updated for custom performance sets')
    pss                                 =   gs_pscr__pss(P['gs'],pscr)
    return P_pss__testinggenerator(P,pss)

def P_pscr__testinggeneratorlength(P,pscr):
    print('this function, P_pscr__testinggeneratorlength, has not been updated for custom performance sets')
    pss                                 =   gs_pscr__pss(P['gs'],pscr)
    return len(pss)*P__nfv(P)

def N_pss__iSs_cSs_oSs_vSs_hSs(N,pss):
    #   vSs stands for "viewed" signal set - what actually came out the other side
    gen                                 =   P_pss__testinggenerator(N,pss)
    iSs,cSs,oSs                         =   traininggenerator__iSs_cSs_oSs(gen)
    print(iSs.shape)
    print(cSs.shape)
    print(oSs.shape)
    vSs,hSs                             =   N['model'].predict([iSs,cSs])
    return iSs,cSs,oSs,vSs,hSs





#
# def P_pss__testingsignals(P,pss):
#     nf                                  =   P['nf']
#     nd                                  =   P['nd']
#     Ssc                                 =   (nf ** nd) * hl(pss)
#     gen                                 =   P_pss__testinggenerator(P,pss)
#
#
# # generator_iSsshape_cSsshape__keraspredictioninput()
#
#
# def nf_nd_iSs_pss__oSs(pss,nf,iSs):
#     seq                                 =   range(hl(pss) * hl(iSs))
#     gen                                 =   nf


#############################################################################################

def a_b_length_rand_conserve_replace_seed__arowtensorbgen(
                                                    a,
                                                    b,
                                                    length,
                                                    rand     = True,
                                                    conserve = True,
                                                    replace  = False,
                                                    seed     = time32bit()
                                                    ):

    modulus                         =   a.shape[0]*b.shape[0]
    indexgen                        =   length_modulus_rand_conserve_replace_seed__gen(
                                                    length,
                                                    modulus,
                                                    rand     = rand,
                                                    conserve = conserve,
                                                    replace  = replace,
                                                    seed     = seed
                                                    )
    for p                           in  indexgen:
        yield a_b_int__rowofarowtensorb(a,b,p)

def length_modulus_rand_conserve_replace_seed__gen( length,
                                                    modulus,
                                                    rand     = True,
                                                    conserve = True,
                                                    replace  = False,
                                                    seed     = time32bit()
                                                    ):
    # generate a sequence of integers from 0 to modulus-1

    if not rand:
        for q                       in  range(length):
            yield q % modulus
        return

    np.random.seed(seed)

    if replace:
        for q                       in  range(length):
            yield np.random.randint(0,modulus)
    elif conserve:
        for q                       in  range(length):
            qmod                    =   q % modulus
            if qmod                 ==  0:
                pool                =   np.random.permutation(modulus)
            yield pool[qmod]
    return

def check_length_modulus_rand_conserve_replace_seed__gen():
    length                          =   10^5
    modulus                         =   np.random.randint(50,1000)
    seed                            =   time32bit()

    genrcr                          =   length_modulus_rand_conserve_replace_seed__gen(
                                                    length,
                                                    modulus,
                                                    rand     = False,
                                                    conserve = False,
                                                    replace  = False,
                                                    seed     = seed
                                                    )
    genRCr                          =   length_modulus_rand_conserve_replace_seed__gen(
                                                    length,
                                                    modulus,
                                                    rand     = True,
                                                    conserve = True,
                                                    replace  = False,
                                                    seed     = seed
                                                    )
    genRcR                          =   length_modulus_rand_conserve_replace_seed__gen(
                                                    length,
                                                    modulus,
                                                    rand     = True,
                                                    conserve = False,
                                                    replace  = True,
                                                    seed     = seed
                                                    )
    # experimental group
    # capital/lower case encodes whether randome and conserve are true or false
    genrcra                         =   np.array([p for p in genrcr])
    genRCra                         =   np.array([p for p in genRCr])
    genRcRa                         =   np.array([p for p in genRcR])

    # control group
    rcr                             =   np.arange(length) % modulus

    np.random.seed(seed)
    numcopy                         =   1 + (length // modulus)
    RCr                             =   [np.random.permutation(modulus) for p in range(numcopy)]
    RCr                             =   np.array(RCr)
    RCr                             =   np.reshape(RCr,(numcopy*modulus,))
    RCr                             =   RCr[:length]

    np.random.seed(seed)
    RcR                             =   np.random.randint(0,modulus,(length,))

    check                           =   np.array([False,False,False])
    check[0]                        =   np.array_equal(genrcra,rcr)
    check[1]                        =   np.array_equal(genRCra,RCr)
    check[2]                        =   np.array_equal(genRcRa,RcR)

    if not np.all(check):
        print("test failed")
        return genrcra,genRCra,genRcRa,rcr,RCr,RcR,check

    print("test passed")
    return []

def f_iS_subvals__rangedgen(f,iS,subvals):
    v                           =   copy.copy(iS)
    for p                       in subvals:
        v[f]                    =   p
        yield v

def f_datagen_subvals__rangedgen(f,gen,subvals):
    for p                       in  gen:
        if p[0][0][f]           ==  1:
            supergen            =   f_iS_subvals__rangedgen(f,p[0][0],subvals)
            for q               in  supergen:
                yield [q,p[0][1]],p[1]
        else:
            yield p


################################################################################
# PARAM  ------
################################################################################

def P__Plike(P):
    """
    Copy all parameters to a new parameter dictionary with a different random
    seeds.

    :param parameter dictionary P
    :return: copy of P with different random seeds
    """
    Q                               =   copy.deepcopy(P)
    Q['trainorderseed']             =   time32bit()
    Q['hawsee']                     =   time32bit()
    Q['oawsee']                     =   time32bit()

    Q['mbi']                        =   -1
    Q['history_timestamps']         =   np.zeros((0))
    Q['history_loss']               =   np.zeros((0))
    Q['history_gradnorm']           =   np.zeros((0))

    R                               =   null__standardP(Q['platform'])
    L                               =   list(Q.keys())
    for key in L:
        if not (key in R.keys()):
            del Q[key]

    return Q




def P__platform(P):
    if (not 'platform' in P.keys()) or (P['platform']=='keras'):
        return 'keras'
    return P['platform']

def P__gsmatrix(P):
    """
    rows ~ input   dimensions
    cols ~ output  dimensions
    """
    nd                              =   P['nd']
    gs                              =   P['gs']

    Z                               =   np.zeros((nd,nd),dtype=int)
    for p                           in  range(gs.shape[0]):
        row                         =   gs[p][0]
        col                         =   gs[p][1]
        Z[row][col]                 =   nd*col+row+1
    return Z

def P__pssrowtensoriSsindexgen(P):
    # returns a generator that produces *row indices* of a the tensor product
    # of the rows of  the iSs matrix and the pss matrix (thought of as a matrix
    # with one column
    modulus                         =   P__nfv(P) * len(P__pss(P))
    return   length_modulus_rand_conserve_replace_seed__gen(    length  =   P['nwu'],
                                                                modulus =   modulus,
                                                                rand    =   P['rand'],
                                                                conserve=   P['conserve'],
                                                                replace =   P['replace'],
                                                                seed    =   P['trainorderseed']
                                                                )
def P__iS_oS_cSgen(P):
    nf                              =   P['nf']
    nd                              =   P['nd']
    iSs                             =   P__ciSs(P)
    pss                             =   P__pss(P)
    seq                             =   P__pssrowtensoriSsindexgen(P)
    return  nf_nd_iSs_pss_seq__generate_iS_oS_cv(
            nf,nd,iSs,pss,seq)

def P__trainingdatagenerator(P):
    gen                             =   P__iS_oS_cSgen(P)
    for p                           in  gen:
                                        # recall that p = (iS,oS,cv), so that
                                        # this generator returns [iS,cv], oS
        yield [np.array([p[0]]),np.array([p[2]])], np.array([p[1]])

def P_iSs_cSs_numit__rowtensortraininggenerator(P,iSs,cSs,numit='auto'):
    # untested
    if len(iSs.shape)               ==  1:
        iSs                         =   vec__rowvec(iSs)
    if len(cSs.shape)               ==  1:
        cSs                         =   vec__rowvec(cSs)
    if numit                        ==  'auto':
        numit                       =   iSs.shape[0]*cSs.shape[0]

    nf                              =   P['nf']
    for p                           in  range(numit):
        iS,cS                       =   a_b_int__rowofarowtensorb(iSs,cSs,p)
        ps                          =   cv__ts(cS)
        oS                          =   ps_nf_iS_fun__oS(ps,nf,iS)
    yield [iS,cS],oS

def P__iSs_oSs_cSs(P):

    gen                             =   P__trainingdatagenerator(P)
    nfv                             =   P__nfv(P)
    nfu                             =   P__nfu(P)
    nd                              =   P['nd']
    nwu                             =   P['nwu']

    iSs                             =   np.zeros((nwu,nfu))
    oSs                             =   np.zeros((nwu,nfu))
    cSs                             =   np.zeros((nwu,nd ** 2))

    c                               =   -1
    for p                           in  gen:
        c                           =   c+1
        iSs[c]                      =   p[0][0]
        cSs[c]                      =   p[0][1]
        oSs[c]                      =   p[1]

    return iSs,oSs,cSs

def P_pss__ciSa_coSa_ccSa(P,pss,platform='np',dtype=torch.float,ofu=None,testmode=False):
    ciSa                            =   P_pss__iSa(P,pss)
    coSa                            =   P_pss__oSs(P,pss)
    ccSa                            =   P_pss__cSs(P,pss)
    if not ofu                      is  None:
        I                           =   np.nonzero(coSa[:,ofu])[0]  # indexing straight into the pytorch tensor doesn't work for some reason
        ciSa                        =   ciSa[I,:]
        coSa                        =   coSa[I,:]
        ccSa                        =   ccSa[I,:]
    #---------------------------------------------------------------------------
    # for unit testing only
    # passed 20190717
    # if testmode:
    #     checklist                       =   []
    #     expectednnz                     =   len(pss)*P['nf']**(P['nd']-1)
    #     print("coSa.shape = ",coSa.shape)
    #     print("I.shape = ",I.shape)
    #     print("expectednnz = ",expectednnz)
    #     checklist.append(I.shape[0]==expectednnz)
    #     checklist.append(ciSa.shape[0]==expectednnz)
    #     checklist.append(coSa.shape[0]==expectednnz)
    #     checklist.append(ccSa.shape[0]==expectednnz)
    #     checklist.append(np.all(coSa[:,ofu]==1))
    #     iSt,oSt,cSt                     =   P_pss__ciSa_coSa_ccSa(P,pss,platform='torch',testmode=False)
    #     Icomp                           =   [p for p in range(iSt.shape[0]) if not p in I]
    #     checklist.append(np.all(oSt.numpy()[:,ofu][I]==1))
    #     checklist.append(np.all(oSt.numpy()[:,ofu][Icomp]==0))
    #     if np.all(checklist):
    #         print("test passed")
    #         return True,checklist
    #     else:
    #         print("test failed")
    #         return False,checklist
    #---------------------------------------------------------------------------
    if platform == 'torch':
        ciSa                        =   torch.tensor(ciSa,dtype=dtype)
        coSa                        =   torch.tensor(coSa,dtype=dtype)
        ccSa                        =   torch.tensor(ccSa,dtype=dtype)
    return ciSa,coSa,ccSa


def check_P_pss__ciSa_coSa_ccSa():
    print("this check assumes that the testing component of the function in question has been uncommented.")
    N                               =   logentryname__N("20190605-195804-f3d5gsc25nwu40000pscr1-1")
    pss                             =   nd_cr__allcrps(N['nd'],cr=[N['nd']])  # important that cr = [N['nd']] b/c otherwise the expected number of nnz's may be wrong, depending on choice of ouput feature unit and task combiation
    for ofu                         in  range(N['nf']*N['nd']):
        passed,checklist            =   P_pss__ciSa_coSa_ccSa(N,pss,ofu=ofu,testmode=True)
        if not passed:
            print("test failed")
            return checklist
    print("test passed")












#
def P__ciSa_coSa_ccSa(P,platform='np',dtype=torch.float,basepss=None,pscr=None):
    """
    :param P:
    :param platform:
    :param dtype:
    :param basepss:
    :param pscr:
    :return: canonical input signal array, canonical output signal array,
    canonical   control signal array
    """
    # NEEDED: pss |--> cSs, (iSs,cSs) |--> oSs
    if basepss                      ==  None:
        pss                         =   P__pss(P)
    elif  (basepss                  ==  'allcombination') and (not pscr == None):
        pss                         =   nd_cr__allcrps(P['nd'],pscr)
    elif  basepss                   ==  'allgssupported': # all ground-set supported
        if not pscr                 ==  None:
            pss                     =   gs_pscr__pss(P['gs'],pscr)
        else:
            pss                     =   gs_pscr__pss(P['gs'],P[pscr])
    else:
        pss                         =   copy.deepcopy(basepss)
    if not pscr                     ==  None:
        pss                         =   [x for x in pss if hl(x) in pscr]
    return P_pss__ciSa_coSa_ccSa(P,pss,platform=platform)

# stands for number of feature vectors
def P__nfv(P):
    return P['nf'] ** P['nd']

# stands for number of feature units
def P__nfu(P):
    return P['nf'] * P['nd']


def P__ciSs(P,justify=False):
    # stands for the *canonical* input singal set
    # justify = don't add a constant to all entries
    # to check for correctness, can run:
    # P = Pkeys__standardextension(nd=3,nf=2,iSgm='uniformrandom')
    # P['iSg']
    if                              not ('iSgm' in P.keys()) or (P['iSgm'] ==  None):
        ciSs                    =   nf_nd__allonehotvecs(P['nf'],P['nd'])
        if (not justify) and ('ifo' in P.keys()) and (not P['ifo']==0):
            ciSs                        =   ciSs + P['ifo']
        return ciSs

    if P['iSgm']                    in  ['uniformrandom','uniformrandom_boundeddistance','onehot']:
        ciSs                        =   nf_x_ndxnf__allrowseq(P['iSg'])
        return ciSs

def P__iSs(P):
    iSs                             =   P__ciSs(P)
    pssc                            =   P__pssc(P)
    iSs                             =   np.tile(iSs,(pssc,1))
    return iSs

def P_pss__iSa(P,pss):
    ciSa                            =   P__ciSs(P)
    ciSa                            =   np.tile(ciSa,(hl(pss),1))
    return ciSa

def gs_pscr__pss(gs,pscr):
    return edges_cran__allcardinalitycranpartialmatching(gs,pscr)

def P__pss(P):
    if not 'pss' in P.keys():
        pscr                        =   P['pscr']
        gs                          =   P['gs']
        return edges_cran__allcardinalitycranpartialmatching(gs,pscr)
    else:
        return P['pss']

def P__pssc(P):
    pss                             =   P__pss(P)
    return heightVlength(pss)

def P__idc(P):
    return P__pssc(P)*P__nfv(P)

def P_ps_iSa__oSa(P,ps,iSa):
    if (not 'transformtype' in P.keys()) or (P['transformtype'] == 'facsimile'):
        st                          =   ps_nf__facsimilefun(ps,P['nf'])
    else:
        st                          =   P_ps__iSa__oSa(P,ps)
    return st(iSa)

def P_ps__oSs(P,ps):
    iSa                             =   P__ciSs(P)
    return P_ps_iSa__oSa(P,ps,iSa)
    # nf                              =   P['nf']
    # if (not 'transformtype' in P.keys()) or (P['transformtype'] == 'facsimile'):
    #     st                          =   ps_nf__facsimilefun(ps,nf)
    # else:
    #     st                          =   P_ps__iSa__oSa(P,ps)
    # iSs                             =   P__ciSs(P)
    # return st(iSs)

def P_pss__oSs(P,pss):
    L                               =   [P_ps__oSs(P,ps) for ps in pss]
    if len(L)                       ==  0:
        return np.zeros((0,P['nd']*P['nf']))
    else:
        return np.concatenate(L,axis=0)

def P_ps_nrows__cSs(P,ps,nrows = 'auto'):
    #   cSs must stand for control signal set
    nd                              =   P['nd']
    nf                              =   P['nf']
    if nrows                        ==  'auto':
        nrows                       =   nf ** nd
    cv                              =   ts_nd__cS(ps,nd)
    cv                              =   np.reshape(cv,(1,nd ** 2))
    cSs                             =   np.repeat(cv,nrows,0)
    return cSs

def P_pss__cSs(P,pss):
    L                               =   [P_ps_nrows__cSs(P,ps) for ps in pss]
    if len(L)                       ==  0:
        return np.zeros((0,P['nd']**2))
    else:
        return np.concatenate(L,axis=0)

def check_P_ps__signals():
    nf                              =   3
    nd                              =   5
    nh                              =   200
    gsc                             =   9
    nfu                             =   nf *  nd
    nfv                             =   nf ** nd
    nwu                           =   nf ** nd
    gs                              =   np.random.randint(0,5,(1,2))
    pscr                            =   [0,1,2,3,4,5]
    conserve                        =   True
    replace                         =   False
    rand                            =   True
    seed                            =   0
    # P                               =   nf_nd_gs_pscr__taskparameters(nf=nf,nd=nd,gs=gs,pscr=pscr)
    #
    # P__appendtraingenparameters(P,rand=False,nwu=nfv)
    # P__appendnetworkparameters(P)
    # ps                              =   P['pss'][0]

    P                               =   Pkeys__standardextension(
                                        nf                              =   nf,
                                        nd                              =   nd,
                                        gsc                             =   gsc,
                                        pscr                            =   pscr)


    # the first time we defined P, we did so only in order to count elements of
    # the preformance set set; now we'll define P in earnest
    nwu                           =   (nf ** nd) * (len(P__pss(P)))

    P                               =   Pkeys__standardextension(
                                        nf                              =   nf,
                                        nd                              =   nd,
                                        nh                              =   nh,
                                        gsc                             =   gsc,
                                        pscr                            =   pscr,
                                        rand                            =   True,
                                        conserve                        =   True,
                                        replace                         =   False,
                                        trainorderseed                  =   seed,
                                        nwu                           =   nwu)


    #
    #
    # P_rand_conserve_replace_nwu_trainorderseed__appendtraingenparameters(
    #                                     P,
    #                                     rand            =   randomizetrainingsequence,
    #                                     conserve        =   conserve,
    #                                     replace         =   replace,
    #                                     nwu           =   nwu,
    #                                     trainorderseed  =   seed)

    iSs0,oSs0,cSs0                  =   P__iSs_oSs_cSs(P)

    gen                             =   P__trainingdatagenerator(P)

    np.random.seed(seed)
    perm                            =   np.random.permutation(nwu)
    ciSs                            =   P__ciSs(P)
    pss                             =   P__pss(P)
    pSs,iSs1                        =   a_b__atensorbbyrow(pss,ciSs)
    iSs1                            =   iSs1[perm]
    pSs                             =   [pSs[p] for p in perm]
    cSs1                            =   tss_nd__cSs(pSs,nd)
    oSs1                            =   [ps_nf_iS_fun__oS(pSs[p],nf,iSs1[p]) for p in range(nwu)]
    oSs1                            =   np.array(oSs1)

    check                           =   np.full((6,),False)
    check[0]                        =   np.array_equal(iSs0,iSs1)
    check[1]                        =   np.array_equal(oSs0,oSs1)
    check[2]                        =   np.array_equal(cSs0,cSs1)

    check[3]                        =   np.array_equal(iSs0[rowsortperm(iSs0)],iSs1[rowsortperm(iSs1)])
    check[4]                        =   np.array_equal(oSs0[rowsortperm(oSs0)],oSs1[rowsortperm(oSs1)])
    check[5]                        =   np.array_equal(cSs0[rowsortperm(cSs0)],cSs1[rowsortperm(cSs1)])

    if not np.all(check):
        print("test failed")
        return P,check, iSs0, iSs1, oSs0, oSs1, cSs0, cSs1, perm, pSs

    print("test passed")
    return []

def P__N(P):
    platform                        =   P__platform(P)

    if platform                     ==  'keras':
        sess = tf.Session()
        K.set_session(sess)

        model                           =   P__amodel(P)
        data_generator                  =   P__trainingdatagenerator(P)

        history                         =   prepare_model(  model,
                                                            data_generator,
                                                            learn_rate              =   P['learnrate'],
                                                            max_iter                =   P['maxit'],
                                                            min_mse                 =   P['msemax'],
                                                            dimensions              =   P['nd'],
                                                            features                =   P['nf'],
                                                            data_seed               =   0, # this value ends up being irrelevant
                                                            reuse_saved_weights     =   False,
                                                            mindelta                =   P['mindelta'],
                                                            nepochs                 =   P['nepochs'],
                                                            clipnorm                =   P['clipnorm'],
                                                            momentum                =   P['momentum'],
                                                            optimizer               =   P['optimizer'],
                                                            baseline                =   P['baseline'],
                                                            decay                   =   P['decay'],
                                                            stepsperepoch           =   P['stepsperepoch'])

        N                               =   copy.copy(P)
        N['model']                      =   model
        N['traindate']                  =   time.strftime("%Y%m%d-%H%M%S")

        if N['keephistory']:
            N['history']                =   N['model'].history.history
        else:
            N['history']                =   None

        # sanity check
        data_generator                  =   P__trainingdatagenerator(P)
        L                               =   []
        for p                           in  data_generator:
            c                           =   np.sum(p[0][1])
            if not (c in L):
                L.append(c)
        print("L0 norms of control vectors = " + str(L))
        N['control_L0norms']            =   L

        # performance report

        D                               =   N_pscr__performancemetrics(N,[1])
        N['mse_pscr1-1']                =   D['mse']
        N['accuracy_pscr1-1']           =   D['accuracy']
        print('pscr 1-1',D)

        D                               =   N_pscr__performancemetrics(N,[0,1,2,3,4,5])
        N['mse_pscr0-5']                =   D['mse']
        N['accuracy_pscr0-5']           =   D['accuracy']
        print('pscr 0-5',D)

        D                               =   N_pscr__performancemetrics(N,[5])
        N['mse_pscr5-5']                =   D['mse']
        N['accuracy_pscr5-5']           =   D['accuracy']
        print('pscr 5-5',D)

    elif platform                       ==  'torch':
        N                               =   copy.deepcopy(P)
        N['model']                      =   pytorchfull.multinet(P)
        #pytorchfull.P__torchm(N)
        pytorchfull.N__train(N)
        N['nwu']                        =   N['nwu']
        N['traindate']                  =   time.strftime("%Y%m%d-%H%M%S")

        # D                               =   N_pscr__performancemetrics(N,[1])
        # N['mse_pscr1-1']                =   D['mse']
        # print('pscr 1-1',D)
        #
        # D                               =   N_pscr__performancemetrics(N,[0,1,2,3,4,5])
        # N['mse_pscr0-5']                =   D['mse']
        # print('pscr 0-5',D)
        #
        # D                               =   N_pscr__performancemetrics(N,[5])
        # N['mse_pscr5-5']                =   D['mse']
        # print('pscr 5-5',D)
    elif platform                       ==  'autodiff':
        import autodifffull
        N                               =   copy.deepcopy(P)
        N['model']                      =   autodifffull.P__composition(P)
        autodifffull.N__trainatd(N)

    return N

def lra_lrb_lrp_lrf__mbi__lr(lra,lrb,lrp,lrf,lrs=None):
    if lrf              ==  'geometric':
        if lrs          ==  None:
            scalefactor =   (lrb/lra) ** (1/lrp)
        else:
            scalefactor =   lrs
        mbi__lr         =   lambda p : lra * (scalefactor ** p)
    elif lrf            ==  'inverse':
        #   NB: this is the decay function used by Keras
        if lrs          ==  None:
            mbi__lr     =   lambda p : lra / (1 + (p/(lrp-1)) * (lra/lrb))
        else:
            mbi__lr     =   lambda p : lra / (1 + p * lrs)
    elif lrf            ==  'linear':
        mbi__lr         =   lambda p : lra + (lrb-lra)*(p / (lrp-1))
    return mbi__lr

def P__mbi__lr(P):
    return lra_lrb_lrp_lrf__mbi__lr(P['lra'],P['lrb'],P['lrp'],P['lrf'],P['lrs'])

def P__mbiXlr(P,samplec=10):
    f                               =   P__mbi__lr(P)
    stepsize                        =   P['nwu']//samplec
    steps                           =   np.arange(0,P['nwu'],stepsize)
    stepsc                          =   hl(steps)
    A                               =   np.zeros((stepsc,2))
    for p in range(stepsc):
        A[p,0]                      =   steps[p]
        A[p,1]                      =   f(steps[p])
    return A
    
################################################################################
#   NET  ------
################################################################################

def N_odima_intereff__meanweightdot(N,odima,intereff):
    I                       =   intervals.itv(odima,N['nf'])
    W                       =   N['model'].who.detach().numpy()[:,I]
    return np.mean(np.matmul(intereff.reshape((1,-1)),W))

def N_i01_o01__lrINhlBYio(N,i0,i1,o0,o1):
    pcd                 =   N_i01o01__pcd(N,i0,i1,o0,o1)
    pcdm                =   pcd__pcdm(pcd)
    pc                  =   pcdm__a(pcdm)
    S                   =   np.array([[0,0],[0,1],[1,0],[1,1]])
    reg                 =   sklearn.linear_model.LinearRegression().fit(S, pc)
    return reg,pc,S

def N_i01_o01__lrINhlBYioscore(N,i0,i1,o0,o1):
    reg,pc,S            =   N_i01_o01__lrINhlBYio(N,i0,i1,o0,o1)
    return reg.score(S,pc)

def N__iolinreg(N,corpus=False):
    pc                  =   []
    sx                  =   []
    sy                  =   []
    S                   =   []
    for p               in  range(N['nd']):
        for q           in  range(N['nd']):
            sx.append(p)
            sy.append(q)
            S.append(np.zeros((10)))
            S[-1][[p,5+q]]  =   1
            pc.append(np.mean(N_io__hSs(N,p,q),axis=0))
    pc                  =   np.array(pc)
    S                   =   np.array(S)
    reg                 =   sklearn.linear_model.LinearRegression().fit(S, pc)
    if corpus:
        return reg, pc,S
    else:
        return reg

def N__iolinregR2(N):
    reg,pc,S            =   N__iolinreg(N,corpus=True)
    return reg.score(S,pc)

def N_io__taskirrelodvar(N,i,o):
    """
    variation (sum of square deviation from mean) in task-irrelevant ouput
    dimensions
    :param N: network dictionary
    :param i: input dimension (integer)
    :param o: output dimension (integer)
    :return:
    """
    ps                  =   np.array([[i,o]])
    ciSa,coSa,ccSa      =   P_pss__ciSa_coSa_ccSa(N,[ps])
    oSs                 =   N_ps__oSs(N,ps)
    depfi               =   np.concatenate([intervals.itv(p,N['nf']) for p in range(N['nd']) if not p==o])
    depf                =   oSs[:,depfi]
    vartot              =   np.sum(np.var(depf,axis=0))
    return vartot

def N_ps__taskirrelodvar(N,ps):
    """
    variation (sum of square deviation from mean) in task(=task #0)-irrelevant
    ouput dimensions
    :param N: network dictionary
    :param i: input dimension (integer)
    :param ps: performance set
    :return:
    """
    oSs                 =   N_pss__oSs(N,[ps])
    depfi               =   np.concatenate([intervals.itv(p,N['nf']) for p in range(N['nd']) if not p==ps[0][1]])
    depf                =   oSs[:,depfi]
    vartot              =   np.sum(np.var(depf,axis=0))
    return vartot

def N_ps__taskirrellrresid(N,ps):
    """
    residtual sum of squares in task (=task #0) irrelevant output dimensions as
    predicted by linear regression on input dimension
    dimensions
    :param N: network dictionary
    :param i: input dimension (integer)
    :param pss: performance set
    :return:
    """
    ciSa,coSa,ccSa      =   P_pss__ciSa_coSa_ccSa(N,[ps])
    indf                =   ciSa[:,intervals.itv(ps[:,0][1:],N['nf'])]  # numpy indexing is insane .. this is the way we have to index ps
    oSs                 =   N_pss__oSs(N,[ps])
    depfi               =   np.concatenate([intervals.itv(p,N['nf']) for p in range(N['nd']) if not p==ps[0][1]])
    depf                =   oSs[:,depfi]
    reg                 =   sklearn.linear_model.LinearRegression().fit(indf,depf)
    varres              =   np.sum(np.var(depf-reg.predict(indf) ,axis=0))
    return varres

def N_io__taskirrelodlinregresidualsos(N,i,o):
    """
    residual sum of squares in task-irrelevant ouput dimensions as
    predicted (by linear regression) by task input dimension
    :param N: network dictionary
    :param i: input dimension (integer)
    :param o: output dimension (integer)
    :return:
    """
    ps                  =   np.array([[i,o]])
    ciSa,coSa,ccSa      =   P_pss__ciSa_coSa_ccSa(N,[ps])
    oSs                 =   N_ps__oSs(N,ps)
    indf                =   ciSa[:,intervals.itv(i,N['nf'])]
    depfi               =   np.concatenate([intervals.itv(p,N['nf']) for p in range(N['nd']) if not p==o])
    depf                =   oSs[:,depfi]
    reg                 =   sklearn.linear_model.LinearRegression().fit(indf,depf)
    varres              =   np.sum(np.var(depf-reg.predict(indf) ,axis=0))
    return varres


def N_n_dend_metric__compofavgdispla(N,n,dend,metric):
    compofavgdispla                 =   np.zeros((n))
    combl                           =   itertools.combinations(range(N['nd']),2)
    combl                           =   [x for x in combl]
    for p                               in  range(n):
        inint                       =   np.random.randint(0,len(combl))
        ouint                       =   np.random.randint(0,len(combl))
        ia                          =   combl[inint]
        oa                          =   combl[ouint]
        pcd                         =   N_i01o01__pcd(N,ia[0],ia[1],oa[0],oa[1])
        if not metric               ==  'volume':
            x                       =   pcd_dtype_dend_metric_agtime__difa(
                                            pcd,
                                            d_type  =   'displ',
                                            d_end   =   dend,
                                            metric  =   metric,
                                            agtime  =   0)
            compofavgdispla[p]      =   x[0]
        elif metric                 ==  'volume':
            compofavgdispla[p]      =   pcd__volume(pcd)
    return compofavgdispla



def N_n__parallelscorea(N,n=10,metric='nip'):
    combl                           =   itertools.combinations(range(N['nd']),2)
    combl                           =   [x for x in combl]
    scorea                          =   []
    for p                           in  range(n):
        inint                       =   np.random.randint(0,len(combl))
        ouint                       =   np.random.randint(0,len(combl))
        ia                          =   combl[inint]
        oa                          =   combl[ouint]
        pcd                         =   N_i01o01__pcd(N,ia[0],ia[1],oa[0],oa[1])
        if metric                   ==  'volume':
            scorea.append(pcd__volume(pcd))
        else:
            score                   =   pcd_dtype_dend_metric_agtime__difa(
                                            pcd,
                                            d_type      =   'displ',
                                            metric      =   metric,
                                            agtime      =   0)
            scorea.append(score[0])
    scorea                          =   np.array(scorea)
    return scorea

def N_io__hSs(N,i,o):
    return N_pss__hSs(N,io__pss(i,o))

def P_od_fuclass__ofu(P,od,fuclass):
    if fuclass                      is  None:
        return None
    else:
        return od*P['nf']+fuclass

def N_i01o01__pcd(N,i0,i1,o0,o1,ofuclass=None):

    CN                              =   N_pss__hSs(N,io__pss(i0,o0),ofu=P_od_fuclass__ofu(N,o0,ofuclass))
    CP                              =   N_pss__hSs(N,io__pss(i0,o1),ofu=P_od_fuclass__ofu(N,o1,ofuclass))
    WN                              =   N_pss__hSs(N,io__pss(i1,o0),ofu=P_od_fuclass__ofu(N,o0,ofuclass))
    WP                              =   N_pss__hSs(N,io__pss(i1,o1),ofu=P_od_fuclass__ofu(N,o1,ofuclass))
    pcd                             =   {   "CN":   CN,
                                            "CP":   CP,
                                            "WN":   WN,
                                            "WP":   WP}
    return pcd

def N_i01o01__pcdw(N,i0,i1,o0,o1):
    pcdw                            =   {   "CN":   N_i_o_wch(N,i0,o0),
                                            "CP":   N_i_o_wch(N,i0,o1),
                                            "WN":   N_i_o_wch(N,i1,o0),
                                            "WP":   N_i_o_wch(N,i0,o0)
                                        }    
    return pcdw

def N_i_o_wch(N,i,o):
    wch                             =   N['model'].wch.data.detach().numpy()
    ts                              =   np.array([[i,o]])
    cS                              =   ts_nd__cS(ts,N['nd'])
    cS                              =   np.ndarray.reshape(cS,(1,-1))
    v                               =   np.matmul(cS,wch)
    return v

def N_pss__hSspca(N,pss,ncomponents=100):
    hSs                             =   N_pss__hSs(N,pss)
    pca                             =   sklearn.decomposition.PCA(n_components=ncomponents)
    pca.fit(hSs)
    return pca

def N__weightnorm(N,l=1):
    if l == 1:
        t                           =   0
        for param in N['model'].parameters():
            t                      +=   torch.sum(torch.abs(param))
        return  float(t.detach().numpy())


def N_io__wt(N,io):
    #   :param io: two letter string specifying input and output dimension
    if N['platform']                ==  'torch':
        D                           =   {   'fh'    :   0,
                                            'ch'    :   1,
                                            'bh'    :   4,
                                            'ho'    :   3,
                                            'bo'    :   5}
        a                           =   [param for param in N['model'].parameters()]
        return a[D[io]].detach()

def N_io__wa(N,io):
    t                               =   N_io__wt(N,io)
    return t.numpy()

def N_ps_woS(N,ps):
    # owlS = output "omnia" layer // means the output in the whole output
    # layer, which consists of two "tensors" (one for the output, one for the
    # facsimile of the hidden layer).  contrast this with oS = output signal,
    # which is only one of these tensors.
    inplatform                      =   P__platform(N)
    if not inplatform == 'keras':
        print('error: this function only applies to Keras models')
        return
    iSs                             =   P__ciSs(N)
    cSs                             =   P_ps_nrows__cSs(N,ps)
    return N['model'].predict([iSs,cSs])

def N_ps__hSs(N,ps,platform='np',ofu=None):
    """
    :param N:
    :param ps:
    :param platform:
    :param ofu: output feature unit
    :return: hSs, an array of hidden layer activations; if ofu (standing for
    output feature unit) is specified, then only activations corresponding to
    [(input + control) signals that should ideally activate that output feature
     unit] will be returned.
    """
    nd                              =   N['nd']
    ndsq                            =   nd ** 2
    inplatform                      =   P__platform(N)
    outplatform                     =   platform
    if inplatform                   ==  'keras':
        pS                          =   N_ps_woS(N,ps)#N['model'].predict([iSs,cSs])
        hSs                         =   pS[1][:,:-ndsq]
    elif inplatform                 ==  'torch':
        if not ofu                  is  None:
            if type(ofu)            is  int:
                return N_ps_ofu__hSs(N,ps,ofu,outplatform=outplatform)
            else:
                print("error: ofu must be an integer")
        iSt,oSt,cSt                 =   P_pss__ciSa_coSa_ccSa(N,[ps],platform='torch')
        hSs                         =   N['model'].hidden(iSt,cSt)
        if outplatform              ==  'np':
            hSs                     =   hSs.detach().numpy()
    return hSs


def check_N_ps__hSs():
    N                               =   logentryname__N("20180705-010618-f3d5gsc25nwu60000pscr0-5")
    ps                              =   N['gs'][[0],:]
    pS                              =   N_ps_woS(N,ps)
    hSs0                            =   pS[1][:,:-25]
    hSs1                            =   N_ps__hSs(N,ps)
    print(np.array_equal(hSs0,hSs1))

def N_ps_ofu__hSs(N,ps,ofu,outplatform='np'):
    """
    :param N:
    :param ps:
    :param ofu: output feature unit. only hidden vectors corresponding to input
    signals that should, in principle, activate this specific hidden unit will
    be returned
    :param outplatform:
    :return:
    """
    #   developer note:
    #   20190616 this function passed the test implemented by function
    #   check_N_ps_ofu__hSs below
    iSt,oSt,cSt                     =   P_pss__ciSa_coSa_ccSa(N,[ps],platform='torch')
    I                               =   np.nonzero(oSt.numpy()[:,ofu])[0]  # indexing straight into the pytorch tensor doesn't work for some reason
    iSt                             =   iSt[I,:]
    oSt                             =   oSt[I,:]
    cSt                             =   cSt[I,:]
    hSs                             =   N['model'].hidden(iSt,cSt)
    if outplatform                  ==  'np':
        hSs                         =   hSs.detach().numpy()
    # for unit testing only
    #
    # checklist                       =   []
    # expectednnz                     =   N['nf']**(N['nd']-1)
    # print("I.shape = ",I.shape)
    # checklist.append(I.shape[0]==expectednnz)
    # checklist.append(hSs.shape[0]==expectednnz)
    # checklist.append(np.all(oSt.numpy()[:,ofu]==1))
    # print(oSt.numpy()[:,ofu])
    # iSt,oSt,cSt                     =   P_pss__ciSa_coSa_ccSa(N,[ps],platform='torch')
    # Icomp                           =   [p for p in range(iSt.shape[0]) if not p in I]
    # checklist.append(np.all(oSt.numpy()[:,ofu][I]==1))
    # checklist.append(np.all(oSt.numpy()[:,ofu][Icomp]==0))
    # if np.all(checklist):
    #     print("test passed")
    #     return True,checklist
    # else:
    #     print("test failed")
    #     return False,checklist
    return hSs

def N_ps_ofu__hSs(N,ps,ofu,outplatform='np'):
    """
    :param N:
    :param ps:
    :param ofu: output feature unit. only hidden vectors corresponding to input
    signals that should, in principle, activate this specific hidden unit will
    be returned
    :param outplatform:
    :return:
    """
    #   developer note:
    #   20190616 this function passed the test implemented by function
    #   check_N_ps_ofu__hSs below
    iSt,oSt,cSt                     =   P_pss__ciSa_coSa_ccSa(N,[ps],platform='torch')
    I                               =   np.nonzero(oSt.numpy()[:,ofu])[0]  # indexing straight into the pytorch tensor doesn't work for some reason
    iSt                             =   iSt[I,:]
    oSt                             =   oSt[I,:]
    cSt                             =   cSt[I,:]
    hSs                             =   N['model'].hidden(iSt,cSt)
    if outplatform                  ==  'np':
        hSs                         =   hSs.detach().numpy()
    # for unit testing only
    #
    # checklist                       =   []
    # expectednnz                     =   N['nf']**(N['nd']-1)
    # print("I.shape = ",I.shape)
    # checklist.append(I.shape[0]==expectednnz)
    # checklist.append(hSs.shape[0]==expectednnz)
    # checklist.append(np.all(oSt.numpy()[:,ofu]==1))
    # print(oSt.numpy()[:,ofu])
    # iSt,oSt,cSt                     =   P_pss__ciSa_coSa_ccSa(N,[ps],platform='torch')
    # Icomp                           =   [p for p in range(iSt.shape[0]) if not p in I]
    # checklist.append(np.all(oSt.numpy()[:,ofu][I]==1))
    # checklist.append(np.all(oSt.numpy()[:,ofu][Icomp]==0))
    # if np.all(checklist):
    #     print("test passed")
    #     return True,checklist
    # else:
    #     print("test failed")
    #     return False,checklist
    return hSs

def check_N_ps_ofu__hSs():
    print("this check assumes that the testing component of the function in question has been uncommented.")
    N                               =   logentryname__N("20190605-195804-f3d5gsc25nwu40000pscr1-1")
    pss                             =   nd_cr__allcrps(N['nd'],cr=[N['nd']])
    ps                              =   pss[np.random.randint(len(pss))]
    for fu                          in  range(N['nf']*N['nd']):
        passed,checklist            =   N_ps_ofu__hSs(N,ps,fu)
        if not passed:
            print("test failed")
            return checklist
    print("test passed")


def N_iSs_cSs__hSs(N,iSs,cSs,outmod='np',dtype=torch.float):
    if P__platform(N)               ==  'torch':
        iSs                         =   torch.tensor(iSs,dtype=dtype)
        cSs                         =   torch.tensor(cSs,dtype=dtype)
        hSs                         =   N['model'].hidden(iSs,cSs).detach()
        if outmod                   ==  'np':
            hSs                     =   hSs.numpy()
        return hSs


def N__chSs(N,platform='np'):
    inplatform                      =   P__platform(N)
    outplatform                     =   platform
    if inplatform                   ==  'torch':
        ciSa,coSa,ccSa              =   P__ciSa_coSa_ccSa(N,platform='torch')
        hSs                         =   N['model'].forward(ciSa,coSa).detach()
        if outplatform              ==  'np':
            return hSs.numpy()
        else:
            return hSs
        return

def N_pss__hSs(N,pss,platform='np',ofu=None):
    """
    :param N:
    :param pss:
    :param platform:
    :param ofu: output feature unit
    :return: hidden unit acitvations for all performance sets; if ofu (standing
    for output feature unit) is specified, then only acivations corresponding
    to [(input+control) signals that will ideally activate that output feature
    unit] will be returned
    """
    outplatform                     =   platform
    L                               =   [N_ps__hSs(N,ps,ofu=ofu) for ps in pss]
    if outplatform                  ==  'np':
        L                           =   np.concatenate(L,axis=0)
    return L

def N_pss__hSsn(N,pss,platform='np'):
    hSs                             =   N_pss__hSs(N,pss,platform=platform)
    n                               =   np.linalg.norm(hSs,axis=1)
    return n

def N_ps__oSs(N,ps,outputplatform='numpy'):
    # ONLY WORK WITH KERAS MODELS
    # produces the *predicted* output, rather than ideal
    if N['platform']                ==  'keras':
        pS                          =   N_ps_woS(N,ps)
        return pS[0]
    elif N['platform']              ==  'torch':
        ciSt,coSt,ccSt                 =   P_pss__ciSa_coSa_ccSa(N,[ps],platform='torch')
        ooSa                        =   N['model'].forward(ciSt,ccSt).detach()
        ooSa                        =   ooSa.reshape((-1,N['nd']*N['nf']))
        if outputplatform           ==  'torch':
            return ooSa
        elif outputplatform         ==  'numpy':
            return ooSa.numpy()
        

def N_pss__oSs(N,pss):
    L                               =   [N_ps__oSs(N,ps) for ps in pss]
    return np.concatenate(L,axis=0)

def N_ps_f__aVhSs_xVregressor_bVfiSs(N,ps,f):
    hSs                             =   N_ps__hSs(N,ps)
    b                               =   P__ciSs(N)
    b                               =   b[:,f]
    x                               =   np.linalg.lstsq(hSs,b)
    x                               =   x[0]
    return hSs,x,b

def N_if_pscr__pss_hSss_regressors_fiSs(N,f,pscr): # if stands for input feature
    print('this function, N_if_pscr__pss_hSss_regressors_fiSs, has not been updated for custom performance sets')
    gs                              =   N['gs']
    nf                              =   N['nf']

    id                              =   f // nf
    pss                             =   ts_id_cr__tsswithinputid(gs,id,pscr)

    hSss                            =   []
    regressors                      =   []

    for ps                          in  pss:
        hSs,x,fiSs                  =   N_ps_f__aVhSs_xVregressor_bVfiSs(N,ps,f)
        hSss.append(hSs)
        regressors.append(x)

    iSs                             =   P__ciSs(N)
    fiSs                            =   iSs[:,f]

    return pss,hSss,regressors,fiSs

def P_if_pscr__pss(N,f,pscr): # if stands for input feature
    print('this function, P_if_pscr__pss, has not been updated for custom performance sets')
    gs                              =   N['gs']
    nf                              =   N['nf']

    id                              =   f // nf
    pss                             =   ts_id_cr__tsswithinputid(gs,id,pscr)
    return pss

def P_of_pscr__pss(N,f,pscr):
    print('this function, P_of_pscr__pss, has not been updated for custom performance sets')
    gs                              =   N['gs']
    nf                              =   N['nf']

    od                              =   f // nf
    pss                             =   ts_od_cr__tsswithinputid(gs,od,pscr)
    return pss

def N_if_pscr__hSss(N,f,pscr):
    print('this function, N_if_pscr__hSss, has not been updated for custom performance sets')
    pss                             =   P_if_pscr__pss(N,f,pscr)
    hSss                            =   []
    for ps                          in  pss:
        hSss.append(N_ps__hSs(N,ps))
    return hSss

def N_of_pscr__hSss(N,f,pscr):
    print('this function, N_of_pscr__hSss, has not been updated for custom performance sets')
    pss                             =   P_of_pscr__pss(N,f,pscr)
    hSss                            =   []
    for ps                          in  pss:
        hSss.append(N_ps__hSs(N,ps))
    return hSss

def N_if_pscr__hSs(N,f,pscr):
    hSss                            =   N_if_pscr__hSss(N,f,pscr)
    hSs                             =   np.concatenate(hSss,axis=0)
    return hSs

def N_of_pscr__hSs(N,f,pscr):
    hSss                            =   N_of_pscr__hSss(N,f,pscr)
    hSs                             =   np.concatenate(hSss,axis=0)
    return hSs

def P_if__fiSs(N,f):
    iSs                             =   P__ciSs(N)
    fiSs                            =   iSs[:,f]
    return fiSs

def N_if_pscr__regressors(N,f,pscr):
    hSss                            =   N_if_pscr__hSss(N,f,pscr)
    fiSs                            =   P_if__fiSs(N,f)
    regressors                      =   []
    for hSs                         in  hSss:
        x                               =   np.linalg.lstsq(hSs,fiSs)
        regressors.append(x[0])
    return regressors

def check_consistency_N_if_pscr__pss_hSss_regressors_fiSs():
    # this could be any network, we chose at random

    print('this function, check_consistency_N_if_pscr__pss_hSss_regressors_fiSs, has not been updated for custom performance sets')

    N                               =   logentryname__N("20180606-110240-f3d5gsc25nwu300000pscr5-5")
    f                               =   0
    pscr                            =   [1]
    pss,hSss,regressors,fiSs        =   N_if_pscr__pss_hSss_regressors_fiSs(N,f,pscr)
    check1                          =   arraylisteq(pss,        P_if_pscr__pss(N,f,pscr))
    check2                          =   arraylisteq(hSss,       N_if_pscr__hSss(N,f,pscr))
    check3                          =   arraylisteq(regressors, N_if_pscr__regressors(N,f,pscr))
    check4                          =   np.array_equal(fiSs,    P_if__fiSs(N,f))
    check                           =   [check1,check2,check3,check4]
    if not all(check):
        print("error: inconcistency detected")
        return check, regressors, N_if_pscr__regressors(N,f,pscr)
    else:
        print("test passed")
        return

def N_if_pscr__pss_hSst_regressort_fiSst_error(N,f,pscr):

    print('this function, N_if_pscr__pss_hSst_regressort_fiSst_error, has not been updated for custom performance sets')

    pss,hSss,regressors,fiSs        =   N_if_pscr__pss_hSss_regressors_fiSs(N,f,pscr)

    hSst                            =   np.concatenate(hSss)
    fiSt                            =   np.tile(vec__colvec(fiSs),(len(pss),1))
    regressort                      =   np.linalg.lstsq(hSst,fiSt)
    regressort                      =   regressort[0]
    error                           =   fiSt - np.matmul(hSst,regressort)
    return pss,hSst,fiSt,regressort,error

def N_if_pscr__hSeuclideandistance(N,f,pscr):
    hSs                             =   N_if_pscr__hSs(N,f,pscr)
    d                               =   sklearn.metrics.pairwise.euclidean_distances(hSs,hSs)
    return d

def N_of_pscr__hSeuclideandistance(N,f,pscr):
    hSs                             =   N_of_pscr__hSs(N,f,pscr)
    d                               =   sklearn.metrics.pairwise.euclidean_distances(hSs,hSs)
    return d

def N_if_pscr_k__hSeuclideandistance2kthnearestneighbor(N,f,pscr,k):
    d                               =   N_if_pscr__hSeuclideandistance(N,f,pscr)
    s                               =   array_k__kthsmallestentryofeachrow(d,k)
    return s

def check_N_if_pscr_k__hSeuclideandistance2kthnearestneighbor():
    # this could be any network, we chose at random
    N                               =   logentryname__N("20180606-110240-f3d5gsc25nwu300000pscr5-5")
    f                               =   0
    pscr                            =   [1]
    k                               =   3
    s                               =   N_if_pscr_k__hSeuclideandistance2kthnearestneighbor(N,f,pscr,k)
    hSs                             =   N_if_pscr__hSs(N,f,pscr)
    t                               =   array_k__euclideandistancetokthnearestneighbor(hSs,k)
    if np.array_equal(s,t):
        print('test passed')
        return []
    else:
        print('test failed')
        return s,t

def N_pss__hSseuclideandistances(N,pss):
    hSs                             =   N_pss__hSs(N,pss)
    d                               =   sklearn.metrics.pairwise.euclidean_distances(hSs,hSs)
    return d

def N_if_pscr__hSdistancessorted(N,f,pscr):
    d                               =   N_if_pscr__hSeuclideandistance(N,f,pscr)
    s                               =   array__rowstackvec(d)
    return np.sort(s)

def N_of_pscr__hSdistancessorted(N,f,pscr):
    d                               =   N_of_pscr__hSeuclideandistance(N,f,pscr)
    s                               =   array__rowstackvec(d)
    return np.sort(s)

def N_pscr__pSs(N,pscr):
#   return: pSs a matrix of output layer activations
    if N['platform']                ==  'torch':
        Q                           =   copy.copy(N)
        Q['model']                  =   None # just to make sure we don't mess with anything
        Q['pscr']                   =   pscr
        ciSt,coSt,ccSt              =   P__ciSa_coSa_ccSa(Q,platform='torch')
        outputs                     =   N['model'].forward(ciSt,ccSt,flatten=False).detach()
        return                          outputs

def N__performancemetrics(N):
    gen                             =   P__trainingdatagenerator(N)
    model                           =   N['model']
    loss_and_metrics                =   model.evaluate_generator(gen,steps=10000)
    D                               =   {}
    D['loss']                       =   loss_and_metrics[0]
    D['accuracy']                   =   loss_and_metrics[2]*100.0
    return D

def N_pscr__performancemetrics(N,pscr,basepss=None):
    platform                        =   P__platform(N)
    if platform                     ==  'keras':
        gen                         =   P_pscr__testinggenerator(N,pscr)
        genlen                      =   P_pscr__testinggeneratorlength(N,pscr)

        performance                 =   N['model'].evaluate_generator(gen,steps=np.minimum(10**4,genlen))
        D                           =   {}
        D['mse']                    =   performance[0]
        D['accuracy']               =   performance[2]
    elif platform                   ==  'torch':
        Q                           =   copy.copy(N)
        Q['model']                  =   None # just to make sure we don't mess with anything
        Q['pscr']                   =   pscr
        ciSt,coSt,ccSt              =   P__ciSa_coSa_ccSa(Q,platform='torch',pscr=pscr,basepss=basepss)
        if ciSt.shape[0]            >   10**4:
            I                       =   np.random.choice(ciSt.shape[0],10**4)
            ciSt                    =   ciSt[I,:]
            coSt                    =   coSt[I,:]
            ccSt                    =   ccSt[I,:]
        outputs                     =   N['model'].forward(ciSt,ccSt).detach()
        criterion                   =   nn.MSELoss()
        loss                        =   criterion(outputs,coSt.view(-1))
        loss                        =   loss.item()
        D                           =   {'mse':loss}
    elif platform                   ==  'autodiff':

        print('need to write this functionality in')

    return D


def N_pss_stepc__performance(N,pss,stepc=None):
    if stepc == 'max':
        stepc                           =   hl(pss)*P__nfv(N)
    elif stepc == None:
        stepc                           =   min(10000,hl(pss)*P__nfv(N))
    gen                                 =   P_pss__testinggenerator(N,pss)
    return N['model'].evaluate_generator(gen,steps=stepc)

#################################################################################################################################
# NEEDED:
# N,ps/pss --> ciSa // coSa // ccSa // chSs // cdismat



#metric can be either 'mse' or 'correct_nn'; the latter records the proportion
#of task-relevant output dimensions whose nearest neighbor among
#feasible output patterns is the target pattern
#note: the average is taken over task-relevant outputs only 
def N_pss_metric__performance(N,ps=None,pss=None,pscr=None,metric='mse',coStrowcriterion=None):

    #   DETERMINE THE SET OF PERFOMANCE SETS

    a                               =   3-sum([ps is None, pss is None, pscr is None])

    if a                            >   1:
        print("error, at most one of [ps, pss, pscr] can take a nontrivial value")
        return

    if  a                           ==  0:
        pscr                        =   [1]

    if not ps                       is  None:
        pss                         =   [ps]

    if not pscr                     is  None:
        pss                         =   nd_cr__allcrps(N['nd'],pscr)

    #   COMPUTE OUTPUT VECTORS

    if P__platform(N)               ==  'keras':
        oSs                         =   N_pss__oSs(N,pss)
        vSs                         =   P_pss__oSs(N,pss)
    elif P__platform(N)             ==  'torch':
        ciSt,coSt,ccSt              =   P_pss__ciSa_coSa_ccSa(N,pss,platform='torch')
        if not coStrowcriterion    is  None:
            I                       =   [p for p in range(coSt.shape[0]) if coStrowcriterion(coSt[p])]
            ciSt                    =   ciSt[I,:]
            coSt                    =   coSt[I,:]
            ccSt                    =   ccSt[I,:]

        if ciSt.shape[0]            >   10**4:
            I                       =   np.random.choice(ciSt.shape[0],10**4)
            ciSt                    =   ciSt[I,:]
            coSt                    =   coSt[I,:]
            ccSt                    =   ccSt[I,:]
        outputs                     =   N['model'].forward(ciSt,ccSt).detach()
        #note: this is a VECTOR; need to reshape to have same shape as coSt        

        if metric                   ==  'mse':
            criterion               =   nn.MSELoss()
            loss                    =   criterion(outputs,coSt.view(-1))
            loss                    =   loss.item()
            return loss
        elif metric                 ==  'correct_nn':
            running_log=[]#record the average value for each task in each ps
            for current_ps in pss:
                ciSt,coSt,ccSt              =   P_pss__ciSa_coSa_ccSa(N,[current_ps],platform='torch')
                if ciSt.shape[0]            >   10**4:
                    I                       =   np.random.choice(ciSt.shape[0],10**4)
                    ciSt                    =   ciSt[I,:]
                    coSt                    =   coSt[I,:]
                    ccSt                    =   ccSt[I,:]
                outputs    =   N['model'].forward(ciSt,ccSt).detach()

                ciSt,coSt,ccSt=[x.numpy() for x in (ciSt,coSt,ccSt)]
                

                ndims,nft= N['nd'], N['nf']
                #WARNING: i am not sure whether the resizing is done in the 
                #correct order, but it seems to give reasonable results
                outputs=outputs.view((-1, ndims*nft)).numpy()
                
                outputs_by_dim=[outputs[:,i*nft: (i+1)*nft] for i in range(ndims)]
                ideal_outputs_by_dim=[coSt[:,i*nft: (i+1)*nft] for i in range(ndims)]
                
                for task in current_ps:
                    ###################################################################################     TEMPORARILY COMMENTED OUT
                    # #first, get the feasible outputs for the relevant outdim
                    # #if there is an entry for iSg, then the codes are distributed
                    # #else, they are one-hot
                    # if 'iSg' in N.keys():
                    #     iSg=N['iSg']
                    #     unique_ideal_outputs=iSg[:,task[0]*nft:(task[0]+1)*nft]
                    # else:
                    #     unique_ideal_outputs=np.vstack((np.eye(nft),np.zeros((1,nft))))
                    ###################################################################################     TEMPORARY REPLACEMENT
                    unique_ideal_outputs=np.vstack((np.eye(nft),np.zeros((1,nft))))
                    ###################################################################################
                        
                    #get id of nearest neighbor among ideal output patters for this dimension
                    nn_id=np.argmin(scipy.spatial.distance.cdist(outputs_by_dim[task[1]], unique_ideal_outputs),axis=1)
                    #replace each output pattern with nearest neighbor among ideal patterns
                    closest_ideal=unique_ideal_outputs[nn_id]
                    
                    #compare the closest ideal to the targets for this dimension
                    ideal_out_this_dim=ideal_outputs_by_dim[task[1]]
                    pct_correct=np.mean([np.allclose(a,b) for a,b in zip(closest_ideal,ideal_out_this_dim)])
                    running_log.append(pct_correct)
            #end loop over pss
            return np.mean(running_log)

            
        else:
            print("Error: funciton N_pss_metric__performance only works with mse and correct_nn on torch platforms, for now")


def ps_nf__taskrelevantoutputnodes(ps,nf):
    oda                             =   ps[:,1]
    return intervals.itv(oda,nf)
# check:
# ps = np.array([[1,2]])
# nf = 3
# def ps_nf__taskrelevantoutputnodes(ps,nf):
#     oda                             =   ps[:,1]
#     return intervals.itv(oda,nf)
# ps_nf__taskrelevantoutputnodes(ps,nf)
# array([6, 7, 8])

def N_pss__err(N,pss,outplatform='numpy',returntargets=False,coStrowcriterion=None):
    if P__platform(N)       ==  'torch':
        ciSt,coSt,ccSt      =   P_pss__ciSa_coSa_ccSa(N,pss,platform='torch')
        if not coStrowcriterion    is  None:
            I                       =   [p for p in range(coSt.shape[0]) if coStrowcriterion(coSt[p])]
            ciSt                    =   ciSt[I,:]
            coSt                    =   coSt[I,:]
            ccSt                    =   ccSt[I,:]
        if ciSt.shape[0]    >   10**4:
            I               =   np.random.choice(ciSt.shape[0],10**4)
            ciSt            =   ciSt[I,:]
            coSt            =   coSt[I,:]
            ccSt            =   ccSt[I,:]
        outputs             =   N['model'].forward(ciSt,ccSt).detach()
        outputs.resize_(coSt.shape)
        err                 =   (coSt-outputs)
        if outplatform      ==  'torch':
            if returntargets:
                return err, coSt
            else:
                return err
        elif outplatform    ==  'numpy':
            if returntargets:
                return err.numpy(), coSt.numpy()
            else:
                return err.numpy()

    elif P__platform(N)     ==  'keras':
        oSs                     =   N_pss__oSs(N,[pss])
        vSs                     =   P_pss__oSs(N,[pss])
        err                     =   oSs-vSs
        return err

def N_ps__taskrelmse(N,ps,normalize=False,coStrowcriterion=None):
    #   it makes sense to take ps as an input, rather than pss, since
    #   the set of relevant output nodes varies from ps to ps
    if normalize:
        err,coSt            =   N_pss__err(N,[ps],returntargets=True,coStrowcriterion=coStrowcriterion)
    else:
        err                 =   N_pss__err(N,[ps],coStrowcriterion=coStrowcriterion)
    taskrelonodes           =   ps_nf__taskrelevantoutputnodes(ps,N['nf'])
    err                     =   err[:,taskrelonodes]
    if normalize:
        return np.mean(err**2) / np.mean(coSt[:,taskrelonodes]**2)
    else:
        return np.mean(err**2)

def N_ps__ratiotaskrel2totse(N,ps):
    """
    ration of squared error in the task-relevant output dimensions to squared
    error in all output dimensions
    :param N:
    :param ps:
    :return:
    """
    err                     =   N_pss__err(N,[ps])
    errsq                   =   err**2
    taskrelonodes           =   ps_nf__taskrelevantoutputnodes(ps,N['nf'])
    taskrelse               =   np.sum(errsq[:,taskrelonodes])
    totse                   =   np.sum(errsq)
    return taskrelse/totse






def N_pandaframe__appendrow(N,dataframe):
    rowindex                        =   dataframe.shape[0]
    for key                         in  N.keys():
        if not (key == 'model'):
            dataframe.at[rowindex,key]  =   N[key]

def N_addit__train(N,addit,resetlr=False):
    platform                        =   P__platform(N)

    if platform                     ==  'keras':
        # prepare a training generator
        P                               =   copy.copy(N)
        P['model']                      =   None # just to make sure we don't mess with the model
        P['nwu']                        =   addit
        gen                             =   P__trainingdatagenerator(P)

        # if necessary, adjust the learning rate
        # NB: vital that these adjustments be made before N['nwu'] is modified
        if (not P['decay'] == 0) and (not resetlr):
            if not 'learnrate_original' in P.keys():
                N['learnrate_original'] =   N['learnrate']
            N['learnrate']              =   N['learnrate_original'] * (1. / (1. + N['decay'] * N['nwu']))
            optimizer                   =   keras.optimizers.SGD(lr=N['learnrate'],clipnorm=N['clipnorm'],decay=N['decay'],momentum=N['momentum'])
            weights                     =   N['model'].get_weights()
            N['model'].compile(loss=[keras.losses.mean_squared_error, None],
                      loss_weights=[1.0, 0],
                      optimizer = optimizer,
                      metrics={'fc2':[keras.metrics.binary_accuracy]})
            N['model'].set_weights(weights)

        # train the model
        N['model'].fit_generator(gen,steps_per_epoch= 1, epochs= addit,verbose = 0)
        N['traindate_final']            =   time.strftime("%Y%m%d-%H%M%S")
        N['nwu']                        =   N['nwu']+addit

        # update the performance metrics
        # performance report
        D                               =   N_pscr__performancemetrics(N,[1])
        N['mse_pscr1-1']                =   D['mse']
        N['accuracy_pscr1-1']           =   D['accuracy']
        print('pscr 1-1',D)

        D                               =   N_pscr__performancemetrics(N,[0,1,2,3,4,5])
        N['mse_pscr0-5']                =   D['mse']
        N['accuracy_pscr0-5']           =   D['accuracy']
        print('pscr 0-5',D)

        D                               =   N_pscr__performancemetrics(N,[5])
        N['mse_pscr5-5']                =   D['mse']
        N['accuracy_pscr5-5']           =   D['accuracy']
        print('pscr 5-5',D)
    elif platform                       ==  'torch':
        print("NB: This function has limited capabilities; consider <pytorchfull.N__train> instead.")
        pytorchfull.N__train(N,addwu=addit)


def N_additrange__Nlist(N,additrange):
    print("NB: this function does not currently work as intended.")
    N                               =   copy.deepcopy(N)
    L                               =   [copy.deepcopy(N)]
    P                               =   copy.deepcopy(N)
    for p                           in  additrange:
        P['nwu']                    =   p
        gen                         =   P__trainingdatagenerator(P)
################################################################################
        # Not sure this is the right thing to do, anymore.  It will depend on
        # how keras determines decayed learn rate when you train a model on
        # multiple training generators in sequence.
        ########################################################################
        # #   NB: the following webpage
        # #   https://towardsdatascience.com/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1
        # #   claims that the formula used for updating learnrate below is
        # #   the correct one, i.e., the one that mathes the formula for learn
        # #   rate decay in the keras SGD model
        # N                           =   copy.deepcopy(L[-1])
        # N['learnrate_original']     =   L[0]['learnrate']
        # N['learnrate']              =   L[0]['learnrate']/(1+N['nwu']*N['decay']) # this value should be set before the value of N['nwu'] is updated
################################################################################
        N['model'].fit_generator(gen,steps_per_epoch= 1, epochs= p,verbose = 0)
        N['traindate_final']         =   time.strftime("%Y%m%d-%H%M%S")
        N['nwu']                  =   N['nwu']+p

        P['nwu']                  =   10**4
        gen                         =   P__trainingdatagenerator(P)
        loss_and_metrics            =   N['model'].evaluate_generator(gen, steps= 10**4)
        N['mse']                    =   loss_and_metrics[0]
        N['accuracy']               =   loss_and_metrics[2]
        L.append(N)
    return L

def N_y__plotcurve(N,y):
    x                               =   N['history_timestamps']
    if y                            ==  'loss':
        y                           =   N['history_loss']
        title                       =   'loss (mse)'
    if y                            ==  'gradnorm':
        y                           =   N['history_gradnorm']
        title                       =   'norm(lr*gradient)'
    x_y__plotlyscatter(x=x,y=y,title=title)

def fp_keptlabs__distmat_20180920(fp,keptlabs):
    if '_600pts'        in  fp:
        pc,pclab        =   parsers.read_rivet_input(fp)
    else:
        if '/'          in  fp:
            N           =   logentrypath__N(fp)
        else:
            N           =   logentryname__N(fp)
        pss             =   nd_cr__allcrps(5,[1])
        pc              =   N_pss__hSs(N,pss)
        pclab           =   null__labels().astype(str)


    pclab_trunc         =   [ ''.join([t[p] for p in keptlabs]) for t in pclab]
    perm_trunc          =   np.argsort(pclab_trunc)
    pc_perm             =   pc[perm_trunc]
    s                   =   sklearn.metrics.pairwise.euclidean_distances(pc_perm,pc_perm)
    return s

def keptlabs_samplerate__perm(keptlabs,samplerate=1):
    pclab               =   null__labels().astype(str) #    = point cloud labels
    pclab_trunc         =   [ ''.join([t[p] for p in keptlabs]) for t in pclab]
    perm_trunc          =   np.argsort(pclab_trunc)
    perm_trunc          =   perm_trunc[::samplerate]
    return perm_trunc

def N_keptlabs__hdistmat(N,samplerate=None,keptlabs=[2,0,1],normalize=None,metric=None,thresh=None):
    pss                 =   nd_cr__allcrps(N['nd'],[1])
    pc                  =   N_pss__hSs(N,pss)

    if samplerate       ==  None:
        samplerate      =   9

    if metric           ==  None:
        metric          =   'euclidean'

    perm_trunc          =   keptlabs_samplerate__perm(keptlabs,samplerate=samplerate)
    pc_perm             =   pc[perm_trunc]

    if metric           ==  'euclidean':
        if normalize    ==  True:
            n           =   np.linalg.norm(pc_perm,axis=1)
            n           =   n[:,np.newaxis]
            n           =   n**(-1)
            pc_perm     =   np.multiply(n,pc_perm)

        s               =   sklearn.metrics.pairwise.euclidean_distances(pc_perm,pc_perm)
    elif metric         ==  'covariance':
        s               =   np.cov(pc_perm)
    elif metric         ==  'correlation':
        s               =   np.corrcoef(pc_perm)
    elif metric         ==  'et':
        if thresh       ==  None:
            thresh      =   np.median(np.ndarray.flatten(pc_perm))
        pc_perm         =   (pc_perm > thresh).astype(float)
        s               =   sklearn.metrics.pairwise.euclidean_distances(pc_perm,pc_perm)
    return s

def N_keptlabs__plothdistmat(N,samplerate=9,keptlabs=[2,0,1],normalize=None,metric='euclidean',thresh=None):
    s                   =   N_keptlabs__hdistmat(N,samplerate=samplerate,keptlabs=keptlabs,normalize=normalize,metric=metric,thresh=thresh)
    heatmap(s,title='ordered lexicographically by'+str(keptlabs)+'where 0~input, 1~output, 2~feature: '+P__logentryname(N))

def N__plothistory(N,var='loss'):
    if var              ==  'loss':
        x_y__plotlyscatter(x=N['history_timestamps'],y=N['history_loss'],title= 'loss curve // '+P__logentryname(N))
    if var              ==  'gradnorm':
        x_y__plotlyscatter(x=N['history_timestamps'],y=N['history_gradnorm'],title= 'norm(gradient) curve // '+P__logentryname(N))
    if var              ==  'lr':
        mbi__lr         =   P__mbi__lr(N)
        y               =   [mbi__lr(int(x)) for x in  N['history_timestamps']]
        y               =   np.array(y)
        x_y__plotlyscatter(x=N['history_timestamps'],y=y)

def N_group_agg__withingrouphuv(N,group=None,agg=None):
    hSs                             =   N_pss__hSs(N,nd_cr__allcrps(5,[1]))
    if group                        ==  None:
        return np.var(hSs,axis=0)
    lab                             =   null__labels()
    if   group                      ==  'id':
        l                           =   lab[:,0]
    elif group                      ==  'od':
        l                           =   lab[:,1]
    elif group                      ==  'f':
        l                           =   lab[:,2]
    elif group                      ==  'iod':
        v                           =   np.transpose(np.array([[10,1,0]]))
        l                           =   np.matmul(lab,v)
        l                           =   np.ndarray.flatten(l)
    return a_labels__withinlabelvariance(hSs,l,agg=agg)

def N_thresh__hui(N,thresh):
#   returns the indices of hidden units whose mean variance across all
#   single-tasks lies above the given threshold
    v                               =   N_group_agg__withingrouphuv(N,agg='mean')
    return np.ndarray.flatten(np.argwhere(v>thresh))

def N__traindate(N):
    """
    :param      N:
    :return:    traindate as integer

    example:    if N['traindate']   =   '20190205-210910' then
                N__traindate(N)     =   20190205210910
    """
    s                               =   N['traindate'][:8]+N['traindate'][9:]
    return np.int(s)


################################################################################
#   FILE -----  FILE
################################################################################


def nndf__save(neuralnetworkdataframe):
    neuralnetworkdataframe.to_csv(pandapath())

def P__recorddirectoryfp(P,subdir=None,arcpath=None):
    # manifoldsfile                   =   os.path.dirname(__file__)
    # codedirectory                   =   os.path.dirname(manifoldsfile)
    # projectdirectory                =   os.path.dirname(codedirectory)
    # prefix                          =   projectdirectory+'/data/archives/archive/'

    # directory                       =   N['traindate']          +\
    #                                     '-f'    +str(N['nf'])   +\
    #                                     'd'     +str(N['nd'])   +\
    #                                     'gsc'   +str(N['gsc'])  +\
    #                                     'nwu' +str(N['nwu'])+\
    #                                     'pscr'  +str(min(N['pscr']))+\
    #                                     '-'     +str(max(N['pscr']))
    # directory                       =   archivedir()+directory
    if not arcpath                    ==    None:
        return arcpath+'/'+P__logentryname(P)
    if subdir == None:
        return archivedir()+'/'+P__logentryname(P)
    else:
        return  filepaths.hiddentasking_loc()+'/data/archives'+'/'+subdir+'/'+P__logentryname(P)

def hiddentaskingdir():
    return filepaths.hiddentasking_loc()

def archivedir():
    # manifoldsfile                   =   os.path.dirname(__file__)
    # codedirectory                   =   os.path.dirname(manifoldsfile)
    # projectdirectory                =   os.path.dirname(codedirectory)
    return hiddentaskingdir()+"/data/archives/archive"

def notebookdir():
    return hiddentaskingdir()+"/data/notebook"

def datadir(shortname):
    """A function"""
    if      shortname == 'hiddentasking':
                return  hiddentaskingdir()
    elif    shortname == 'archive':
                return  archivedir()
    elif    shortname == 'notebook':
                return  notebookdir()
    elif    shortname == 'rivet':
                return  datadir('hiddentasking')+'/data/rivet/'
    elif    shortname == 'msevanova':
                return hiddentaskingdir()+'/data/archives/pub/nips2019/msevanova'

def P__logentryname(P):
    name                             =   P['traindate']          +\
                                        '-f'    +str(P['nf'])   +\
                                        'd'     +str(P['nd'])   +\
                                        'gsc'   +str(P['gsc'])  +\
                                        'nwu'   +str(P['nwu'])  +\
                                        'pscr'  +str(min(P['pscr']))+\
                                        '-'     +str(max(P['pscr']))
    return name


def N__save2archive(N,subdir=None,arcpath=None):

    dir                             =   P__recorddirectoryfp(N,subdir=subdir,arcpath=arcpath)
    os.mkdir(dir)

    # make a copy to work with
    M                               =   copy.copy(N)

    # determine the platform
    platform                        =   P__platform(M)

    # save the model
    if platform                     ==  'keras':
        M['model'].save(dir+'/model.hdf5')
    elif platform                   ==  'torch':
        torch.save(M['model'].state_dict(),dir+'/model.pt')

    # remove the model and, if appropriate, the history from the copied
    # dictionary
    M['model']                      =   None
    if ('keephistory' in M.keys()) and (not M['keephistory']):
        M['history']                =   None

    # save the reduced copy
    np.savez(dir+'/dictionary.npz',[M])

    # write what one reasonably can to text files
    for keyword                     in  M.keys():
        val                         =   M[keyword]
        t                           =   type(val)
        filepath                    =   dir + "/" + keyword

        if t                        in  [np.ndarray,list,range]:
            if (t == list) and (len(val)>0) and not (type(val[0]) in [int,float,np.float64]):
                continue
            ext                     =   ".csv"
            np.savetxt(filepath+ext,val)
        elif t                      in  [str,float,int,np.float64,bool]:
            ext                     =   ".txt"
            f                       =   open(filepath+ext,"w+")
            f.write(str(val))
            f.close()
        elif t                      ==  keras.engine.training.Model:
            ext                     =   ".hdf5"
            val.save(filepath+ext)

    # display the destination directory
    print("saved to directory " + dir)

def logentrypath__P(fp):
    npzfile                         =   np.load(fp + "/dictionary.npz")
    Q                               =   npzfile['arr_0']
    Q                               =   Q[0]
    return Q

def logentrypath__N(fp,template=None):
    #   NB:     see check function below
    npzfile                         =   np.load(fp + "/dictionary.npz")
    Q                               =   npzfile['arr_0']
    Q                               =   Q[0]
    platform                        =   P__platform(Q)
    if platform                     ==  'keras':
        if template                 ==  None:
            Q['model']              =   ezread(fp+"/model.hdf5")
        else:
            Q['model']              =   copy.copy(template)
            Q['model'].load_weights(fp+"/model.hdf5")
    elif platform                   ==  'torch':
        Q['model']                  =   pytorchfull.multinet(Q)
        Q['model'].load_state_dict(torch.load(fp + "/model.pt"))
    return Q

def check_logentrypath__N(platform = 'torch'):

    #   KERAS TESTS:

    if platform                     ==  'keras':
        #   an arbitrary selection
        lep                             =   archivedir() + "/20180705-012303-f3d5gsc25nwu100000pscr0-5"
        tep                             =   archivedir() + "/20180705-012951-f3d5gsc25nwu200000pscr0-5"
        control                         =   logentrypath__N(lep)
        template                        =   logentrypath__N(tep)
        experiment                      =   logentrypath__N(lep,template=template['model'])
        #   check agreement of keras models
        pss                             =   nd_cr__allcrps(5,[1])
        hSs_control                     =   N_pss__hSs(control,pss)
        hSs_experiment                  =   N_pss__hSs(experiment,pss)
        np.array_equal(hSs_control,hSs_experiment)
        #   check agreement of everthing else
        control['model']                =   None
        experiment['model']             =   None
        np.testing.assert_equal(control,experiment)
        #   pat self on back
        print('test passed')

    #   TORCH TESTS

    elif platform               ==  'torch':
        P                           =   Pkeys__standardextension(nwu=50,mbc=100,lr=1, pscr=[1],lrs=0.01)
        N                           =   P__N(P)
        name                        =   P__logentryname(N)
        recorddir                   =   datadir('archive')+'/'+name
        N__save2archive(N)
        M                           =   logentryname__N(name)
        hSs0 = N_pss__hSs(N,nd_cr__allcrps(5,[1]))
        hSs1 = N_pss__hSs(M,nd_cr__allcrps(5,[1]))
        if np.array_equal(hSs0,hSs1):
            print('test passed')
        else:
            print('test failed')

        shutil.rmtree(recorddir)

def logentryname__N(logentryname,template=None):
    fp=archivedir()+'/'+logentryname
    return logentrypath__N(fp,template=template)

def logentryname__P(logentryname):
    fp=archivedir()+'/'+logentryname
    return logentrypath__N(fp)

def dir__dict(dir):
    filenames                       =   os.listdir(dir)

    D                               =   {}
    for name                        in  filenames:
        fp                          =   dir+"/"+name
        key                         =   os.path.splitext(name)[0]
        D[key]                      =   ezread(fp)
    return D

def ezread(fp):
    ext                             =   os.path.splitext(fp)[1]
    if ext                          ==  '.txt':
        f                           =   open(fp,"r")
        x                           =   f.read()
        f.close()
        if isnumber(x):
            if isinteger(float(x)):
                return int(float(x))
            else:
                return float(x)
        else:
            return x
    elif ext                        ==  '.csv':
        x                           =   np.loadtxt(fp)
        isint                       =   np.vectorize(isinteger)
        if np.all(isint(x)):
            return x.astype(int)
        else:
            return x
    elif ext                        ==  '.hdf5':
        return keras.models.load_model(fp)
    elif ext                        ==  '.npz':
        return np.load(fp)

def archivedir__pandaframe(fp='auto'):
    if fp                           ==  'auto':
        fp                          =   archivedir()

    recordlist                      =   os.listdir(fp)
    recordlist                      =   recordlist[1:]
    recorddir                       =   fp+recordlist[0]
    net                             =   logentrypath__N(recorddir)
    df                              =   pd.DataFrame(columns=net.keys())
    for recordname                  in  recordlist:
        recorddir                   =   fp+"/"+recordname
        net                         =   logentrypath__N(recorddir)
        df                          =   df.append(net,ignore_index=True)
    return df

def null__compiledkerasmodel():
    #   NB: see check function below

    # an arbitrary choice
    N                               =   logentryname__N("20180705-005736-f3d5gsc25nwu10000pscr0-5")
    pss                             =   N['gs'][[0],:]
    hSs                             =   N_pss__hSs(N,[pss])
    return N['model']

def check_null__compiledkerasmodel():
    N                               =   logentryname__N("20180705-005736-f3d5gsc25nwu10000pscr0-5")
    M                               =   copy.copy(N)
    M['model']                      =   null__compiledkerasmodel()

    pss                             =   nd_cr__allcrps(5,[5])
    def null__NhSs():
        N_pss__hSs(N,pss)

    def null__MhSs():
        return N_pss__hSs(M,pss)

    start                           =   time.time()
    null__NhSs()
    end                             =   time.time()
    Ntime                           =   end-start

    start                           =   time.time()
    null__MhSs()
    end                             =   time.time()
    Mtime                           =   end-start

    print("time for uncompiled = "  + str(Ntime))
    print("time for compiled = "    + str(Mtime))


def logentrynamelist_pss__srminput(logentrynamelist,pss):
    nN                              =   len(logentrynamelist) # stands for
                                                                # num. of nets
    if nN                           ==  0:
        return np.zeros((0,0,0))

    pssc                            =   heightVlength(pss)  # stands for
                                                            # performance set
                                                            # set cardinality

    N                               =   logentryname__N(logentrynamelist[0])
    nh                              =   N['nh'] # nh stands for number of
                                                # hidden units
    nsamples                        =   pssc * (N['nf'] ** N['nd'])


    srminput                        =   np.zeros((nN,nh,nsamples))

    for p                           in  range(nN):
        N                           =   logentryname__N(logentrynamelist[p])
        hSs                         =   N_pss__hSs(N,pss)
        srminput[p,:,:]             =   np.transpose(hSs)

    return srminput

def null__erasehistory_WARNING_PERMANENT():
    dirs                            =   datetimemin_datetimemax__logentrynames(33333333,33333333333334)

def nickname__filel(nickname = 'summer2018',fullpath=False):
    if nickname                     ==  'summer2018':
        logentrynames               =   datetimemin_datetimemax__logentrynames(20180827183509, 20180827215955)
        Dl                          =   [logentryname__dict(name) for name in logentrynames]
        df                          =   pd.DataFrame([logentryname__dict(name) for name in logentrynames])
        df['nwu']                 =   pd.to_numeric(df['nwu'])
        df.sort_values(by = ['datetime','nwu'],inplace=True)
        logentrynames               =   list(df['name'])
        if fullpath:
            logentrypaths           =   [datadir('archive')+'/'+name for name in logentrynames]
            return                      logentrypaths
        else:
            return                       logentrynames
    if nickname                     ==  'summer2018rivet':
        names                       =   nickname__filel(nickname = 'summer2018')
        names                       =   [name+'_600pts' for name in names]
        if fullpath:
            paths                   =   [datadir('rivet')+'/input/archives/archive/'+name for name in names]
            return                      paths
        else:
            return                      names

################################################################################
#   DATA FRAME -----
################################################################################

def pandaframe_index__net(dataframe,index):
    D                               =   {}
    for col                         in  list(dataframe.columns.values):
        D[col]                      =   dataframe.loc[index][col]
    return D


################################################################################
#   -----  SIGNAL TRANSFORM
################################################################################


# Note 1: <ps> should take the form of an mx2 matrix
# Note 2: the input signals (<iS>) that will be passed to <st> should be
# encoded as the rows of an mx(nf*nd) matrix
def ps_nf__facsimilefun(ps,nf):
    psc                             =   ps.shape[0]
    def st(iS):
        oS                          =   np.zeros(iS.shape)
        irange                      =   intervals.itv(ps[:,0],nf)
        orange                      =   intervals.itv(ps[:,1],nf)
        oS[:,orange]                =   copy.deepcopy(iS[:,irange])
        return oS
    return st

def P_ps__iSa__oSa(P,ps):
    """
    :param P: dictionary of parameters
    :param ps: a perforamnce set
    :return: iSa__oSa, a function that takes an input signal
    array and returns the corresponding (ideal) output signal array
    """
    def iSa__oSa(iSa):
        faxfun                      =   ps_nf__facsimilefun(ps,P['nf'])
        oSa                         =   faxfun(iSa)
        if ('transformtype' in P.keys())        and             (P['transformtype']  ==  'classification'):
            for p                   in  range(ps.shape[0]):
                idim                =   ps[p,0]
                odim                =   ps[p,1]
                irange              =   intervals.itv(idim,P['nf'])
                orange              =   intervals.itv(odim,P['nf'])
                oSa[:,orange]       =   np.round(np.matmul(oSa[:,orange],P['sta'][:,irange]),0)
        return oSa
    return iSa__oSa

def check_ps_nf__facsimilefun():
    nf                              =   3
    nd                              =   10

    # Test 1
    idim                            =   np.random.permutation(nd)
    odim                            =   np.random.permutation(nd)
    idimcomp                        =   idim[range(5,10)]
    odimcomp                        =   odim[range(5,10)]
    idim                            =   idim[range(5)]
    odim                            =   odim[range(5)]
    ps                              =   np.vstack((idim,odim))
    ps                              =   np.transpose(ps)

    psop                            =   np.zeros(ps.shape,dtype='int')
    psop[:,0]                       =   ps[:,1]
    psop[:,1]                       =   ps[:,0]

    facfun                          =   ps_nf__facsimilefun(ps,3)
    facfunop                        =   ps_nf__facsimilefun(psop,3)

    iS                              =   nf_nd__allonehotvecs(nf,nd)
    oS                              =   facfun(iS)
    oSop                            =   facfunop(oS)

    inzitv                          =   intervals.itv(idim,nf)
    onzitv                          =   intervals.itv(odim,nf)

    izitv                           =   intervals.itv(idimcomp,nf)
    ozitv                           =   intervals.itv(odimcomp,nf)

    if np.array_equal(iS,oSop):
        print("test passed")
        return np.array([])
    else:
        print("test failed")
        return idim,odim,ps,psop,facfun,facfunop,iS,oS,oSop

# <ps> should take the form of an mx2 matrix
# <iS> that will be passed to <st> should be an mx(nf*nd) matrix
def ps_nf_fun__st(ps,nf,fun=idfun):
    psc                             =   ps.shape[0]
    def st(iS):
        oS                          =   np.zeros(iS.shape)
        if iS.ndim                  ==  2:
            for p                   in  range(psc):
                irange              =   intervals.itv(ps[p,0],nf)
                orange              =   intervals.itv(ps[p,1],nf)
                oS[:,orange]        =   fun(iS[:,irange])
        else:
            for p                   in  range(psc):
                irange              =   intervals.itv(ps[p,0],nf)
                orange              =   intervals.itv(ps[p,1],nf)
                oS[orange]          =   fun(iS[irange])
        return oS
    return st

def check_ps_nf_fun__st():
    numits                          =   5
    nf                              =   3
    nd                              =   10
    psc                             =   5
    for p                           in  range(numits):
        ps                          =   tsc_nd__randts(psc,nd)
        st1                         =   ps_nf__facsimilefun(ps,nf)
        st2                         =   ps_nf_fun__st(ps,nf)
        for q                       in  range(10):
            iS                      =   np.random.rand(50,nf*nd)
            oS1                     =   st1(iS)
            oS2                     =   st2(iS)
            if not np.array_equal(oS1,oS2):
                print("test failed")
                return nf,nd,psc,ps,st1,st2,iS,oS1,oS2
    print("test passed")
    return []


################################################################################
#   ----- TASK SET
################################################################################


#   generate a random task set; dimensions range from 0 to nd-1
def tsc_nd__randts(tsc,nd):
    tslin                           =   np.random.choice(
                                        nd ** 2,size = (tsc,),replace=False)
    ts                              =   np.zeros((tsc,2),dtype='int')
    ts[:,0]                         =   tslin % nd
    ts[:,1]                         =   (tslin) // nd
    perm                            =   rowsortperm(ts)
    ts                              =   ts[perm]
    return ts

#   * edges should be a m x 2 matrix
#   *   output is a list of m x 2 arrays; each m x 2 array represents a set of
#       c tasks that do not overlap in input or output dimension
def edges_c__allcardinalitycpartialmatching(edges,c):
    m                               =   edges.shape[0]
    adj                             =   edges__interferencegraphadj(edges)
    partialmatchings                =   adj_c__allcardinalityccliques(adj,c)
    numpairs                        =   len(partialmatchings)
    for p                           in  range(numpairs):
        rows                        =   np.array(partialmatchings[p])
        rows                        =   rows.astype(int)
        partialmatchings[p]         =   edges[rows]
    return partialmatchings

#   *   edges should be a m x 2 matrix
#   *   output is a list of m x 2 arrays; each m x 2 array represents a set of
#       AT MOST c tasks that do not overlap in input or output dimension
def edges_cran__allcardinalitycranpartialmatching(edges,cran):
    partialmatchings                =   []
    for p                           in  cran:
        partialmatchings_new        =   edges_c__allcardinalitycpartialmatching(
                                        edges,p)
        partialmatchings            =   partialmatchings+partialmatchings_new
    return partialmatchings

def nd_cr__allcrps(nd,cr):
    if cr                           ==  [nd]:
        permi                       =   itertools.permutations([p for p in range(nd)])
        pss                         =   [np.array([np.arange(nd),x]).T for x in permi]
    else:
        ts                          =   tsc_nd__randts(nd ** 2, nd)
        pss                         =   edges_cran__allcardinalitycranpartialmatching(ts,cr)
    return pss

#   <edges> should be a m x 2 matrix
#   output is a list of lists of edges
def edges__interferencegraphadj(edges):
    m                               =   edges.shape[0]
    B                               =   np.full((m, m), False)

    for p                           in  range(m):
        B[p,p]                      =   True
    for p                           in  range(m):
        for q                       in  range(p+1,m):
            task1                   =   edges[p]
            task2                   =   edges[q]
            if np.any(task1 == task2):
                continue
            else:
                B[p,q]              =   True
                B[q,p]              =   True
    return B

#   <adj> should be a symmetric boolean matrix
def adj_c__allcardinalityccliques(adj,c):
    m                               =   adj.shape[0]

    def f(s):
        if np.all(adj[np.ix_(s,s)]):
            return True
        else:
            return False

    combos                          =   itertools.combinations(range(m), c)
    cliques                         =   []
    for c                           in  combos:
        if f(c):
            cliques.append(c)
    return cliques


# id stands for input dimension
# cr stands for cardinality range
# returns a list of task sets; each task set E that satisfies
# * E has cardinality in <cr>
# * E is a partial matching
# * at least one element of E has input dimension id
def ts_id_cr__tsswithinputid(ts,id,cr):
    # stands for potential performance set set
    ppss                            =   edges_cran__allcardinalitycranpartialmatching(ts,cr)

    pss                             =   []
    for E                           in  ppss:
        if np.any(E[:,0]==id):
            pss.append(E)

    return pss

# od stands for input dimension
# cr stands for cardinality range
# returns a list of task sets; each task set E that satisfies
# * E has cardinality in <cr>
# * E is a partial matching
# * at least one element of E has output dimension od
def ts_od_cr__tsswithinputid(ts,od,cr):
    # stands for potential performance set set
    ppss                            =   edges_cran__allcardinalitycranpartialmatching(ts,cr)

    pss                             =   []
    for E                           in  ppss:
        if np.any(E[:,1]==od):
            pss.append(E)

    return pss


################################################################################
#   SIGNAL TRANSFORMS
################################################################################


def ps_nf_iS_fun__oS(ps,nf,iS,fun=idfun):
    st                              =   ps_nf_fun__st(ps,nf,fun)
    return st(iS)

def check_ps_nf_iS__oS():
    psc                             =   5
    nd                              =   10
    nf                              =   3
    ps                              =   tsc_nd__randts(psc,nd)

    ff                              =   ps_nf__facsimilefun(ps,nf)


    iS                              =   nf_nd__allonehotvecs(nf,nd)
    oS                              =   ff(iS)
    tS                              =   ps_nf_iS_fun__oS(ps,nf,iS)
    check1                          =   np.array_equal(oS,tS)

    iS                              =   np.random.rand(iS.shape[0],iS.shape[1])
    oS                              =   ff(iS)
    tS                              =   ps_nf_iS_fun__oS(ps,nf,iS)
    check2                          =   np.array_equal(oS,tS)

    if check1 and check2:
        print("test passed")
        return []
    else:
        print("test failed")
        return psc,nd,nf,ps,ff


################################################################################
#   DATA FORMATTING
################################################################################


def cv__ts(cv):
    ndsq                            =   cv.shape[0]
    nd                              =   int(np.sqrt(ndsq))
    supp                            =   np.nonzero(cv)
    supp                            =   supp[0]
    rows                            =   supp // nd
    cols                            =   supp %  nd
    ts                              =   np.array([rows,cols])
    ts                              =   np.transpose(ts)
    return ts

#   * ts[p,0] is the input dimension
#   * ts[p,1] is the output dimension
def ts_nd__cS(ts,nd):
    """
    :param ts:  task set
    :param nd:  numver of dimensions
    :return:    imagine that the control units are arranged in an nd x nd grid,
                G, where nd = # dimensions, and identify

                node (i,j) ~ task connecting INPUT i to OUTPUT j

                a function G -> {0,1} then determines a task set.  the control
                vector associated to that set is the length nd**2 vector
                obtained by flattening this grid
                    1) along ROWS
                    2) in the STYLE OF PYTHON
                    3) exactly as done by FLATTENING:    np.ndarray.flatten()
    """
    cv                              =   np.zeros(nd ** 2)
    m                               =   ts.shape[0]
    for p                           in  range(m):
        ind                         =   nd * ts[p,0] + ts[p,1]
        cv[ind]                     =   1
    return cv

def tss_nd__cSs(tss,nd):
    return np.array([ts_nd__cS(p,nd) for p in tss])

#   use some examples to check that cv__ts and ts_nd__cS are mutually inverse
#   functions
def check_cv__ts__cv():
    numits                          =   10
    nd                              =   10
    ndsq                            =   nd ** 2
    for p                           in  range(numits):
        cv                          =   np.random.randint(0,high=2,size=(ndsq,))
        ts                          =   cv__ts(cv)
        cv2                         =   ts_nd__cS(ts,nd)
        if not np.array_equal(cv,cv2):
            print("test failed")
            return nd,cv,ts,cv2
    print("test passed")
    return []

def array_param2_maxdist__atlatlpointcloud( array,
                                            param2="codensity",
                                            maxdist=None):
    if maxdist                     ==  None:
        d                           =   sklearn.metrics.pairwise.euclidean_distances(array,array)
        d                           =   array__rowstackvec(d)
        maxdist                     =   np.median(d)/2
    elif maxdist                    ==  np.inf:
        d                           =   sklearn.metrics.pairwise.euclidean_distances(array,array)
        d                           =   array__rowstackvec(d)
        maxdist                     =   np.max(d)
    pointset                        =   [Point(*row) for row in array]
    atlatlpointcloud                =   PointCloud( pointset,
                                                    param2,
                                                    max_dist= maxdist)
    return atlatlpointcloud

def array_param2_maxdist_fp_codensityparam__rivetfile(
                                        array,
                                        param2="codensity",
                                        maxdist=None,
                                        fp="rivetpath.txt",
                                        codensityparam=None):
    file                            =   open(fp,"w")

    m                               =   array.shape[0]
    if param2                       ==  "codensity":
        if codensityparam           ==  None:
            codensityparam          =   math.ceil(np.sqrt(m))
        codensity                   =   array_k__euclideandistancetokthnearestneighbor(array,codensityparam)
        codensity                   =   np.reshape(codensity,(m,1))
        array                       =   np.concatenate((array,codensity),axis=1)

    pc                              =   array_param2_maxdist__atlatlpointcloud(
                                        array,
                                        param2="codensity",
                                        maxdist=maxdist)
    pc.save(file)
    file.close()

#   UNDER DEVELOPMENT
# def params__standardrivetextension(Q):
#     if not 'labels' in Q:
#         Q['labels']                 =   [str(p) for p in range(hl(Q['pts']))]
#     if not 'file_name' in Q:
#         Q['file_name']              =   '/Users/gh10/Desktop/rivetinputtemp.txt.'
#     if not 'fn_name' in Q:
#         Q['fn_name']                =

def array_k__euclideandistancetokthnearestneighbor(array,k):
    m                               =   array.shape[0]
    d                               =   sklearn.metrics.pairwise.euclidean_distances(array,array)
    return array_k__kthsmallestentryofeachrow(d,k)
    # codensity                       =   np.zeros((m,))
    # for p                           in  range(m):
    #     s                           =   np.sort(d[p])
    #     codensity[p]                =   s[k]
    # return codensity

def array_k__kthsmallestentryofeachrow(array,k):
    m                               =   array.shape[0]
    v                               =   np.zeros((m,))
    for p                           in  range(m):
        s                           =   np.sort(array[p])
        v[p]                        =   s[k]
    return v

    # test script:
    # a = np.random.rand(5,5)
    # d = sklearn.metrics.pairwise.euclidean_distances(a,a)
    # v = array_k__euclideandistancetokthnearestneighbor(a,2)
    # d
    # # performan a direct visual comparison


################################################################################
#   NAME/PATH --------
################################################################################

def namel__parascorea(      namelist,
                            nsamplepernet   =   10,
                            metric          =   'nip'):
    pathl           =   [archivedir()+'/'+name for name in namelist]
    return pathl__parascorea(pathl,nsamplepernet=nsamplepernet,metric=metric)

def pathl__parascorea(      pathl,
                            nsamplepernet   =   10,
                            metric          =   'nip'):
    scoreal                 =   []
    for path                in  pathl:
        N                   =   logentrypath__N(path)
        a                   =   N_n__parallelscorea(N,
                                                    n=nsamplepernet,
                                                    metric=metric)
        scoreal.append(np.mean(a))
    parascorea              =   np.array(scoreal)
    return parascorea

def namel__msea(namel,score='maxmulti'):
    pathl                   =   [archivedir()+'/'+name for name in namel]
    return pathl__msea(pathl,score=score)

def pathl__msea(pathl,score='maxmulti'):
    if score                ==  'maxmulti':
        def N__mse(N):
            return N['mse_pscr'+str(N['nd'])+'-'+str(N['nd'])]
    msea                    =   [N__mse(logentrypath__N(path)) for path in pathl]
    return np.array(msea)

def arcpath__logentrypathl(arcpath):
    """

    :param arcpath:  arc stands for archive; archive means a folder containing
                    log entry folders; this is a directory to an archive
    :return:        the list of paths to the log entries in this archive
    """
    # check
    # pathl=arcpath__logentrypathl('/Users/gh10/a/c/p/taskhr/data/archives/pub/nips2019/msevanova_noncompliantnets/20181211-212933-f3d5gsc25nwu50000pscr1-1')
    # [logentrypath__N(path) for path in pathl]
    pathl                   =   []
    names                   =   os.listdir(arcpath)
    for logentryname        in  names:
        if logentryname[:2] == '20':
            logentrypath        =   arcpath+"/"+logentryname
            pathl.append(logentrypath)
    nwul                    =   [logentrypath__P(path)['nwu'] for path in pathl]
    pathl                   =   ai_bi__asortedbyb(pathl,nwul)
    return pathl

def arcpath__PCAspread(arcpath):
    pathl                   =   arcpath__logentrypathl(arcpath)
    nwul                    =   [logentrypath__P(path)['nwu'] for path in pathl]
    pathl                   =   ai_bi__asortedbyb(pathl,nwul)

    N                       =   logentrypath__N(pathl[0])
    pss                     =   nd_cr__allcrps(N['nd'],[1])

    S                       =   np.zeros(len(pathl))
    for p                   in  range(hl(pathl)):
        N                   =   logentrypath__N(pathl[p])
        pc                  =   [np.mean(N_pss__hSs(N,[pss[q]]),axis=0) for q in range(N['nd']**2)]
        pc                  =   np.array(pc)
        pccenter            =   pc-np.mean(pc,axis=0)
        covmat              =   np.matmul(pccenter,pccenter.T)
        score               =   (np.trace(covmat)**2) / (np.trace(np.matmul(covmat,covmat)))
        S[p]                =   score
    return S

def arcpath__PCA95xplpscr22(arcpath):
    pathl                   =   arcpath__logentrypathl(arcpath)
    nwul                    =   [logentrypath__P(path)['nwu'] for path in pathl]
    pathl                   =   ai_bi__asortedbyb(pathl,nwul)

    N                       =   logentrypath__N(pathl[0])
    pss                     =   nd_cr__allcrps(N['nd'],[2])

    S                       =   np.zeros(len(pathl))
    for p                   in  range(hl(pathl)):
        N                   =   logentrypath__N(pathl[p])
        spreadl             =   []
        for q               in  range(hl(pss)):
            ps              =   pss[q]
            pcd             =   N_i01o01__pcd(N,ps[0][0],ps[1][1],ps[0][1],ps[1][1])
            pc              =   pcdm__a(pcd__pcdm(pcd))
            pca             =   sklearn.decomposition.PCA(n_components=pc.shape[0])
            pca.fit(pc)
            v               =   np.cumsum(pca.explained_variance_ratio_)
            val             =   np.min([p for p in range(pc.shape[0]) if v[p]>0.95])
            spreadl.append(val)
        S[p]                =   np.mean(spreadl)
    return S


def arcpath__PCA95xpl(arcpath):
    pathl                   =   arcpath__logentrypathl(arcpath)
    nwul                    =   [logentrypath__P(path)['nwu'] for path in pathl]
    pathl                   =   ai_bi__asortedbyb(pathl,nwul)

    N                       =   logentrypath__N(pathl[0])
    pss                     =   nd_cr__allcrps(N['nd'],[1])

    S                       =   np.zeros(len(pathl))
    for p                   in  range(hl(pathl)):
        N                   =   logentrypath__N(pathl[p])
        pc                  =   [np.mean(N_pss__hSs(N,[pss[q]]),axis=0) for q in range(N['nd']**2)]
        pc                  =   np.array(pc)
        pca                 =   sklearn.decomposition.PCA(n_components=pc.shape[0])
        pca.fit(pc)
        v                   =   np.cumsum(pca.explained_variance_ratio_)
        val                 =   np.min([p for p in range(pc.shape[0]) if v[p]>0.95])
        S[p]                =   val
    return S

def arcpath__PCAspreadpscr22(arcpath):
    pathl                   =   arcpath__logentrypathl(arcpath)
    nwul                    =   [logentrypath__P(path)['nwu'] for path in pathl]
    pathl                   =   ai_bi__asortedbyb(pathl,nwul)

    N                       =   logentrypath__N(pathl[0])
    pss                     =   nd_cr__allcrps(N['nd'],[2])

    S                       =   np.zeros(len(pathl))
    for p                   in  range(hl(pathl)):
        N                   =   logentrypath__N(pathl[p])
        spreadl             =   []
        for q               in  range(hl(pss)):
            ps              =   pss[q]
            pcd             =   N_i01o01__pcd(N,ps[0][0],ps[1][1],ps[0][1],ps[1][1])
            pc              =   pcdm__a(pcd__pcdm(pcd))
            pccenter        =   pc-np.mean(pc,axis=0)
            covmat          =   np.matmul(pccenter,pccenter.T)
            score           =   (np.trace(covmat)**2) / (np.trace(np.matmul(covmat,covmat)))
            spreadl.append(score)
        S[p]                =   np.mean(spreadl)
    return S

################################################################################
#   -------- RECORD NAMES
################################################################################

def logentriesfolderpath__alllogentrypaths(logentrysetpath):
    L                               =   [x[0] for x in os.walk(logentrysetpath)]
    L                               =   L[1:]
    return L

def logentriesfolderpath__alllogentrynames(logentrysetpath):
    L                               =   logentriesfolderpath__alllogentrypaths(logentrysetpath)
    L                               =   [os.path.basename(os.path.normpath(path)) for path in L]
    return L


def null__allrecorddirpaths():
    L                               =   [x[0] for x in os.walk(archivedir())]
    L                               =   L[1:]
    return L

def null__alllogentrynames():
    L                               =   null__allrecorddirpaths()
    L                               =   [os.path.basename(os.path.normpath(path)) for path in L]
    return L

def logentryname__dict(logentryname):
    name                        =   copy.copy(logentryname)
    dateint                     =   int(name[:8])
    timeint                     =   int(name[9:15])
    datetimeint                 =   timeint+dateint*10**6
    props                       =   name[16:]

    f                           =   re.compile('f[0-9]*')
    d                           =   re.compile('d[0-9]*')
    gsc                         =   re.compile('gsc[0-9]*')
    pscmin                      =   re.compile('[0-9]*-')
    pscmax                      =   re.compile('-[0-9]*')

    f                           =   f.findall(props)[0]
    d                           =   d.findall(props)[0]
    gsc                         =   gsc.findall(props)[0]
    pscmin                      =   pscmin.findall(props)[0]
    pscmax                      =   pscmax.findall(props)[0]

    D                           =   {}
    D['name']                   =   name
    D['f']                      =   int(f[1:])
    D['d']                      =   int(d[1:])
    D['gsc']                    =   int(gsc[3:])
    D['pscmin']                 =   int(pscmin[:-1])
    D['pscmax']                 =   int(pscmax[1:])


    D['date']                   =   dateint
    D['time']                   =   timeint
    D['datetime']               =   datetimeint
    nwu                       =   re.compile('nwu[0-9]*')
    nwu                       =   nwu.findall(name)
    if len(nwu) >   0:
        D['nwu']              =   nwu[0][5:]
    else:
        D['nwu']              =   None
    return D

def logentryname_param__paramval(logentryname,param):
    x                         =   logentryname__dict(logentryname)[param]
    return int(x)

def null__alllogentrynamedicts():
    L                               =   logentriesfolderpath__alllogentrynames(archivedir())
    R                               =   []
    m                               =   len(L)
    for p                           in  range(m):
        D                           =   logentryname__dict(L[p])

        R.append(D)
    return R

def datetimemin_datetimemax__archiveindices(datetimemin,datetimemax):
    minmax                          =   [datetimemin,datetimemax]
    for p                           in  range(2):
        bound                       =   minmax[p]
        ndigits                     =   len(str(bound))
        minmax[p]                   =   bound * 10 **(14-ndigits)

    R                               =   null__alllogentrynamedicts()
    m                               =   len(R)
    selector                        =   [p for p in range(m) if ((R[p]['datetime'] <= minmax[1]) and (minmax[0] <= R[p]['datetime']))]
    t                               =   [R[p]['datetime'] for p in selector]
    t                               =   np.array(t)
    perm                            =   np.argsort(t)
    sortedselector                  =   [selector[p] for p in perm]
    return   sortedselector

def datetimemin_datetimemax__logentrynames(datetimemin,datetimemax,logentriesfolder=None):
    if logentriesfolder             ==  None:
        logentriesfolder            =   archivedir()
    selector                        =   datetimemin_datetimemax__archiveindices(datetimemin,datetimemax)
    allnames                        =   logentriesfolderpath__alllogentrynames(logentriesfolder)
    names                           =   [allnames[p] for p in selector]
    return names

def datetimemin_datetimemax_Pkeys__logentrynames(datetimemin=0, datetimemax=99999999999999,**kwargs):
    R                               =   null__alllogentrynamedicts()
    indices                         =   datetimemin_datetimemax__archiveindices(datetimemin,datetimemax)
    R                               =   [R[p] for p in indices]
    def satisfactory(D):
        for key                     in  kwargs:
            if key                  in  D:
                if not D[key] ==  kwargs[key]:
                    return False
            else:
                return False
        return True
    return [namedict['name'] for namedict in R if satisfactory(namedict)]

def logentryname__datetimenwu(logentryname):
    D                               =   logentryname__dict(logentryname)
    return int(str(D['datetime'])+D['nwu'])

def logentryname__nwu(logentryname):
    D                               =   logentryname__dict(logentryname)
    return int(D['nwu'])

################################################################################
#   ------  BETTI 0
################################################################################


def symmat__bettidict(symmat):
    T                               =   scipy.sparse.csgraph.minimum_spanning_tree(symmat)
    vals                            =   T.data
    uvals,udns                      =   np.unique(vals,return_counts=True)
    ucum                            =   np.cumsum(udns)

    D                               =   dict(   radii           =   uvals,
                                                ncomponents     =   symmat.shape[0]-ucum)
    return D


################################################################################
#   DATED FUNCTIONS
################################################################################


# 2018-06-15

def plotdisti_20180615(w):
    f                       =   0
    pscr                    =   [1]
    N                       =   logentryname__N(w)
    s                       =   N_if_pscr__hSdistancessorted(N,f,pscr) # note input feature, not output
    x_y__plotlyscatter(y = s[::1000])


def plotdisto(w):
    f                       =   0
    pscr                    =   [1]
    N                       =   logentryname__N(w)
    s                       =   N_of_pscr__hSdistancessorted(N,f,pscr)  # note output feature, not output
    savedir                 =   notebookdir()+'/20180623'
    filename                =   'distributionofmindistances'+w+'.html'
    savepath                =   savedir+'/'+filename
    smean                   =   np.mean(s)
    smin                    =   np.min(s)
    smax                    =   np.max(s)
    svar                    =   np.var(s)
    title                   =   'sorted list of {euclideandistance(x,y): x, y in E}    E = {hidden vectors for 5 tasks with output dim 0}<br>' + \
                                'num points = ' + str(heightVlength(s)) + '    [min,mean,max] = ' + str([smin,smean,smax]) + '     var = ' + str(svar) + '<br>' + \
                                'network filename: ' + w + '   //    figure by <plotdisto><br>'
    x_y__plotlyscatter(y = s[::heightVlength(s)//300], title=title, savepath=savepath)

# 2018-06-22

def logentrynamelist__explainedlist_20180622(logentrynamelist,autoplot=True):
    Nl                          =   [logentryname__N(name) for name in logentrynamelist]
    nN                          =   len(logentrynamelist)

    # generate the performance set set
    Pdummy                      =   Pkeys__standardextension(pscr=[1])
    pss                         =   P__pss(Pdummy)

    # generate the hidden signals
    hSsl                        =   [N_pss__hSs(N,pss) for N in Nl]

    # add a third element to the list: the concatenation of the last two
    hSsl.append(np.concatenate(hSsl,axis=1))

    # run pca
    expl                        =   arraylist__singleVjointvariancexplained(hSsl,ncomponents=100)
    return expl


def arraylist__singleVjointvariancexplained(arraylist,ncomponents=100):
    L                           =   copy.copy(arraylist)
    L.append(np.concatenate(L,axis=1))
    # run pca
    pcal                        =   []
    for p                       in  range(len(L)):
        pca                     =   PCA(n_components=ncomponents)
        pca.fit(L[p])
        pcal.append(pca)

    # analyze the explained variance
    expl                        =   [np.cumsum(x.explained_variance_ratio_) for x in pcal]
    return expl

# 2018-06-23

def logentryname_pss_k__hSeucldist2kthnearestnbr(logentryname,pss=nd_cr__allcrps(5,[1]),k=1):
    # by default pss = {all singleton performance sets}
    N                           =   logentryname__N(logentryname)
    hSs                         =   N_pss__hSs(N,pss)
    s                           =   array_k__euclideandistancetokthnearestneighbor(hSs,k)
    return s

def logentryname_pss_k__plothSeucldist2kthnearestnbr(logentryname,pss=nd_cr__allcrps(5,[1]),k=1):
    s                           =   logentryname_pss_k__hSeucldist2kthnearestnbr(logentryname,pss=pss,k=1)
    s                           =   np.sort(s)
    smean                       =   np.mean(s)
    smin                        =   np.min(s)
    smax                        =   np.max(s)
    svar                        =   np.var(s)
    savedir                     =   notebookdir()+'/20180623'
    filename                    =   'distributionofnearestnbrs'+logentryname+'.html'
    savepath                    =   savedir+'/'+filename
    title                       =   'sorted list of {min_{y in E-{x}} euclideandistance(x,y): x in E}    E = {hidden vectors for the performance sets shown below}<br>' + \
                                    str(pss) + '<br>' + \
                                    'num points = ' + str(heightVlength(s)) + '    [min,mean,max] = ' + str([smin,smean,smax]) + '     var = ' + str(svar) + '<br>' + \
                                    'network filename: ' + logentryname + '   //    figure by <logentryname_pss_k__plothSeucldist2kthnearestnbr><br>'
    x_y__plotlyscatter(y = s[::heightVlength(s)//300], title=title, savepath=savepath)

################################################################################
#   NO LONGER USED
################################################################################


def generate_data(nd, nf, random_order, with_replacement, seed, nwu,
                    gs,mpsc,returnpermutation=False):
    iSs =   nf_nd__allonehotvecs(nf,nd)
    pss =   edges_cran__allcardinalitycranpartialmatching(gs,range(mpsc))
    m   =   heightVlength(pss)
    n   =   heightVlength(iSs)

    if not with_replacement:
        if random_order:
            rows                        =   np.random.permutation(m * n)
        else:
            rows                        =   np.arange(m * n)

    for p                               in  range(nwu):
        if with_replacement:
            row                         =   np.random.permutation(0,m * n)
        else:
            row                         =   rows[p % (m * n)]

        ps,iS                           =   a_b_int__rowofarowtensorb(
                                            pss,iSs,row)

        oS                              =   ps_nf_iS_fun__oS(ps,nf,iS)
        cv                              =   ts_nd__cS(ps,nd)
        if returnpermutation:
            yield  iS,oS,cv,rows
        else:
            yield  iS,oS,cv

def check_generate_data():
    numits                              =   5

    random_order                        =   False
    with_replacement                    =   False
    seed                           =   0
    for p                               in  range(numits):
        params                          =   np.random.randint(2,high=7,size=(4,))
        nd                              =   params[0]
        nf                              =   params[1]
        gsc                             =   params[2]
        mpsc                            =   params[3]
        gs                              =   tsc_nd__randts(gsc,nd)
        pss                             =   edges_cran__allcardinalitycranpartialmatching(gs,range(mpsc))
        pssc                            =   len(pss)
        niS                             =   nf ** nd
        ndata                           =   niS * pssc
        nwu                        =   ndata

        x                               =   generate_data(nd, nf, random_order,
                                            with_replacement, seed,
                                            nwu,gs,mpsc,
                                            returnpermutation=True)

        xlong                           =   generate_data(nd, nf, random_order,
                                            with_replacement, seed,
                                            10000,gs,mpsc,
                                            returnpermutation=True)

        # experimental array
        iSs                             =   np.zeros((ndata,nf*nd))
        oSs                             =   np.zeros((ndata,nf*nd))
        cSs                             =   np.zeros((ndata,nd ** 2))
        counter                         =   -1
        perm                            =   0
        for q                           in  x:
            a,b,c,d                     =   q
            counter                     =   counter+1
            iSs[counter]                =   a
            oSs[counter]                =   b
            cSs[counter]                =   c
            perm                        =   d

        modcheck                        =   True
        counter                         =   -1
        for q                           in  xlong:
            a,b,c,d                     =   q
            counter                     =   counter+1
            modidx                      =   counter % ndata
            iSscomp                     =   np.array_equal(a,iSs[modidx])
            oSscomp                     =   np.array_equal(b,oSs[modidx])
            cSscomp                     =   np.array_equal(c,cSs[modidx])
            permcomp                    =   np.array_equal(d,perm)
            subcheck                    =   [iSscomp,oSscomp,cSscomp,permcomp]
            if not all(subcheck):
                modcheck                =   False
                break

        # control arrays
        ISS                             =   nf_nd__allonehotvecs(nf,nd)
        ISS                             =   np.tile(ISS,(pssc,1))

        CVS                             =   [ts_nd__cS(E,nd) for E in pss]
        CVS                             =   np.array(CVS)
        CVS                             =   np.repeat(CVS,niS,0)

        def f(k):
            ps_f                        =   cv__ts(CVS[k])
            iS_f                        =   ISS[k]
            return ps_nf_iS_fun__oS(ps_f,nf,iS_f)

        OSS                             =   [f(k) for k in range(ndata)]
        OSS                             =   np.array(OSS)

        invperm                         =   np.arange(perm.shape[0])
        invperm[perm]                   =   np.arange(perm.shape[0])

        iSs                             =   iSs[invperm]
        oSs                             =   oSs[invperm]
        cSs                             =   cSs[invperm]

        checklist                       =   [   np.array_equal(iSs,ISS),
                                                np.array_equal(oSs,OSS),
                                                np.array_equal(cSs,CVS),
                                                modcheck]
        if not all(checklist):
            print("test failed")
            return nd,nf,gsc,mpsc,gs,pss,pssc,x,niS,ndata,iSs,oSs,cSs,perm,\
                   invperm,ISS,CVS,OSS,checklist
    print("test passed")
    return []
