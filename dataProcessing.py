# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 23:28:53 2017

@author: ningshangyi
"""

import nltk
import sys
# Generate Grams, may take several seconds.
from nltktest import UniGram, BiGram, TriGram

# Confusion Mat
from mat import delMat, transMat, repMat, insMat, traMat

testdata = open('testdata.txt', 'r')
vocab = open('vocab.txt', 'r')

vocabSet = set()
for line in vocab:
    i = line.split()
    for item in i:
        vocabSet.add(item)     
vocab.close()
def findRelatedWordSimple(ew):
    '''
        Simply find related word and prob of changing,
        some of the code from http://norvig.com/spell-correct.html
        added trades ('abcde' -> 'abedc') because there are many of them in testdata
    '''
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    few = ''
    tew = ew
    if ew != ew.lower():
        letters += 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        pew = ew[1:]
        few = ew[0]
        ew = pew
    length     = len(ew)
    tlen       = len(tew)
    splits     = [(ew[:i], ew[i:])        for i in range(length + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    delP       = [delMat[ew[0].lower(), '@']] + [delMat[R[0].lower(), L[-1].lower()]  for L, R in splits if (L and R)]
    trades     = [tew[:i] + tew[j] + tew[i+1:j] + tew[i] + tew[j+1:] for i in range(tlen) for j in range(1, tlen) if i<j]
    transP     = [transMat[R[0].lower(), R[1].lower()]    for L, R in splits if len(R)>1]
    repP       = [repMat[R[0].lower(), c]         for L, R in splits if R for c in letters]
    insP       = [insMat['@', c] for c in letters] +\
                 [insMat[L[-1].lower(), c]        for L, R in splits if L for c in letters]
    tradeP     = [traMat[i, j]  for i in range(tlen) for j in range(1, tlen) if i<j]
    if few:
        r = deletes + transposes + replaces + inserts
        r = [few + x for x in r] + trades
        p = delP + transP + repP + insP + tradeP
        return r, p
    return deletes + transposes + replaces + inserts + trades,\
           delP + transP + repP + insP + tradeP

def P(w1, w2='', w3=''):
    '''
        Compute language model for Uni- and Bi-Gram, easily expand to TriGrams or more
    '''
    if w3 != '':
        if TriGram[tuple([w1.lower(), w2.lower(), w3.lower()])] == 0:
            return -20
        return TriGram[tuple([w1.lower(), w2.lower(), w3.lower()])]
    if w2 == '':
        if UniGram[tuple([w1.lower()])] == 0:
            return -20
        return UniGram[tuple([w1.lower()])]
    if BiGram[tuple([w1.lower(), w2.lower()])] == 0:
        return -20
    return BiGram[tuple([w1.lower(), w2.lower()])]
    

def correctWord(ew, prew, nxtw, method='Sim2', l=2):
    '''
        correcting spelling errors
    '''
    toReturn = ew
    maxP = -99999
    if method == 'Simple' or 'Sim2':
        rWord, rp = findRelatedWordSimple(ew)
        for i in range(len(rWord)):
            item = rWord[i]
            p    = rp[i] 
            if item in vocabSet and item != ew:
    #            print('%s turned into %s' % (ew, item))
                ip = p*l*0.2 + P(prew, item) + P(item, nxtw) + P(item) * 0.2\
                         - (2-l) * P(prew) * 0.1 - (2-l) * P(nxtw) * 0.1 -\
                         (2-l) * P(ew, nxtw)*0.2 - (2-l) * P(prew, ew)*0.2 -\
                         (2-l) * P(ew)*0.1 + P(prew, item, nxtw) - P(prew, ew, nxtw)\
                         - P(prew)*0.3 - P(nxtw)*0.3
                if ip > maxP:
                    toReturn = item
                    maxP = ip
#                    print(item, P(prew, item) + P(item, nxtw))
    if toReturn != ew:
        return (toReturn, maxP)
    if method == 'Sim2':
        rWord, rp = findRelatedWordSimple(ew)
        for i in range(len(rWord)):
            item = rWord[i]
            p    = rp[i] 
            rWord2, rp2 = findRelatedWordSimple(item)
            for i in range(len(rWord2)):
                item2 = rWord2[i]
                p2    = rp2[i] 
                if item2 in vocabSet and item2 != ew:
    #                print('%s turned into %s' % (ew, item))
                    ip = p*l + p2*l + P(prew, item2) + P(item2, nxtw)
                    if ip > maxP:
                        toReturn = item2
                        maxP = ip
#                        print(item2, P(prew, item2) + P(item2, nxtw))
    return (toReturn, maxP)

def findErrWord(i):
    '''
        Find spell errors, return err as number of errors not detected
    '''
    num, err, text = i.split('\t')
    err = int(err)
    toReturn = ''
    errWL = nltk.word_tokenize(text)
    for i in range(len(errWL)):
        if not errWL[i] in vocabSet:
            err -= 1
            toReturn += ' ' + errWL[i]
            errWL[i], p = correctWord(errWL[i], errWL[i-1], errWL[i+1], 'Sim2')
    return (int(num), err, toReturn, errWL)

def realWordErr(answer):
    maxP = -99999
    words = answer.split(' ')
    for i in range(len(words)):
        if len(words[i])>3:
            if i == 0:
                c, p = correctWord(words[i], '.', words[i+1], method='Simple', l=0)
            elif i == len(words)-1:
                c, p = correctWord(words[i], words[i-1], '', method='Simple', l=0)
            else:
                c, p = correctWord(words[i], words[i-1], words[i+1], method='Simple', l=0)
            if p > maxP and c != words[i]:
                maxP = p
                changingWordIndex = i
                cWord = c
#                print(words[i-1:i+2], c, p)
    if maxP > -99999:
#        print(words[changingWordIndex], cWord)
        words[changingWordIndex] = cWord
    toReturn = ''
    for item in words:
        toReturn += item + ' '
    return toReturn

def writeAnswer(path):
    '''
        output, eval atst.
    '''
    testdata = open('testdata.txt', 'r')
    f = open(path, 'w')
    for i in testdata:
        num, err, errWords, errWL = findErrWord(i)
        answer = ''
        for item in errWL:
            if item == ',' and answer[-1] == '.':
                answer += str(item)
            else:
                answer += ' ' + str(item)
        if err > 0:
            answer = realWordErr(answer)
        f.write('%s\t%s\n' % (num, answer))
        sys.stdout.write('\rWrote %d \t' % num)
    f.close()
    ansfile=open(path,'r')
    resultfile=open('result.txt','r')
    count=0
    for i in range(1000):
        ansline=ansfile.readline().split('\t')[1]
        ansset=set(nltk.word_tokenize(ansline))
        resultline=resultfile.readline().split('\t')[1]
        resultset=set(nltk.word_tokenize(resultline))
        if ansset==resultset:
            count+=1
    print("Accuracy is : %.2f%%" % (count*1.00/10))
    
writeAnswer('answer.txt')
