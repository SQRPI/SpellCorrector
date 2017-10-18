# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 14:28:45 2017

@author: ningshangyi
"""

from collections import defaultdict
import numpy as np

letters = 'abcdefghijklmnopqrstuvwxyz'

def findRelatedWord(ew):
    '''
        Simply find related word, code from http://norvig.com/spell-correct.html
    '''
    splits     = [(ew[:i], ew[i:])        for i in range(len(ew) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    trades     = [ew[:i] + ew[j] + ew[i+1:j] + ew[i] + ew[j+1:] for i in range(len(ew)) for j in range(1, len(ew)) if i<j]
    return deletes, transposes, replaces, inserts, trades

def test():
    letterDict = defaultdict(int)
    f = open('spell-errors.txt', 'r')
    delMat, transMat, repMat, insMat, traMat =\
    defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int)
    for line in f:
        corw, errs = line.split('\n')[0].split(':')
        corw = corw.lower()
        for i in range(len(corw)):
            item = corw[i]
            letterDict[item] += 1
            if i == len(corw)-1:
                letterDict[item, '@'] += 1
            elif i == 0:
                letterDict['@', item] += 1                
            else:
                letterDict[item, corw[i+1]] += 1
                letterDict[corw[i-1], item] += 1
            
        deletes, transposes, replaces, inserts, trades = findRelatedWord(corw)
        err = errs.split(',')
        err = [x[1:].lower() for x in err]
        for item in err:
            if '*' in item:
                err.remove(item)
                ri, n = item.split('*')
                err += [ri] * int(n)
        for item in err:
            if item in deletes:
                i = deletes.index(item)
                if i != 0:
                    delMat[corw[i], corw[i-1]] += 0.01
                else:
                    delMat[corw[i], '@'] += 0.01
            if item in transposes:
                i = transposes.index(item)
                transMat[corw[i], corw[i+1]] += 8
            if item in replaces:
                i = replaces.index(item)
                r, y = divmod(i, 26)
                repMat[corw[r], letters[y]] += 1
            if item in inserts:
                i = inserts.index(item)
                r, y = divmod(i, 26)
                if r != 0:
                    insMat[corw[r-1], letters[y]] += 20
                else:
                    insMat['@', letters[y]] += 20
            if item in trades:
                i = trades.index(item)
                ij = [(corw[i], corw[j]) for i in range(len(corw)) for j in range(1, len(corw)) if i<j]
                traMat[ij[i][0], ij[i][1]] += 20
    f.close()
    return delMat, transMat, repMat, insMat, traMat, letterDict
delMat, transMat, repMat, insMat, traMat, letterDict = test()

for i in letters:
    for j in letters:
        letterDict[i, j] += 1
        delMat[i, j]   += 0.01 / letterDict[i, j]
        transMat[i, j] += 8 / letterDict[i, j]
        repMat[i, j]   += 1 / letterDict[i, j]
        insMat[i, j]   += 20 / letterDict[i, j]
        traMat[i, j]   += 20 / letterDict[i, j]
for i in letters:
    delMat[i, '@'] += 0.2 / letterDict[i, j]
    insMat['@', i] += 1 / letterDict[i, j]

s = sum(delMat.values()) + sum(transMat.values()) + sum(repMat.values()) + sum(insMat.values())

def logpMat(mat):
    for item in mat:
        mat[item] = np.log(mat[item]/s)
    return mat

delMat   = logpMat(delMat)
transMat = logpMat(transMat)
repMat   = logpMat(repMat)
insMat   = logpMat(insMat)
traMat   = logpMat(traMat)
