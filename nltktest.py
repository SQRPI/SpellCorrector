# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 15:31:26 2017

@author: ningshangyi
"""

import nltk
from collections import defaultdict
import numpy as np

#books = nltk.corpus.gutenberg.fileids()
book2 = nltk.corpus.reuters.fileids()
#book3 = nltk.corpus.inaugural.fileids()
def gramming(n):
    Dict = defaultdict(int)
#    for book in books:
#        dataText = [x.lower() for x in nltk.corpus.gutenberg.words(book)]
#        for i in range(len(dataText)-n+1):
#            t = tuple(dataText[i:i+n])
#            Dict[t] += 1
    
    for book in book2:
        dataText = [x.lower() for x in nltk.corpus.reuters.words(book)]
        for i in range(len(dataText)-n+1):
            t = tuple(dataText[i:i+n])
            Dict[t] += 1

#    for book in book3:
#        dataText = [x.lower() for x in nltk.corpus.inangural.words(book)]
#        for i in range(len(dataText)-n+1):
#            t = tuple(dataText[i:i+n])
#            Dict[t] += 1
    return Dict


def normalGram(gram):
    value = list(gram.values())
    sortedValue = sorted(set(value))
    maxValue = max(value)
    ss = sum(value) + maxValue
    valueTo = dict()
    for item in sortedValue:
        if item == maxValue:
            valueTo[item] = np.log(item/ss)
        else:
            nxtValue = sortedValue[sortedValue.index(item)+1]
            valueTo[item] = np.log(nxtValue*value.count(nxtValue)/ss)
    for item in gram:
        gram[item] = valueTo[gram[item]]
    return gram

def normalGram2(gram):
    ss = sum(gram.values())+0.01
    for item in gram:
        gram[item] = np.log(gram[item]/ss)
    return gram

UniGram = normalGram2(gramming(1))
BiGram = normalGram2(gramming(2))
TriGram = normalGram2(gramming(3))