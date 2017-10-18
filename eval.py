import nltk
anspath='answer.txt'
resultpath='result.txt'
ansfile=open(anspath,'r')
resultfile=open(resultpath,'r')
datafile = open('testdata.txt', 'r')
count=0
for i in range(1000):
    num, ansline=ansfile.readline().split('\t')
    resultline=resultfile.readline().split('\t')[1]
    dataline = datafile.readline().split('\t')[2]
    ansl = nltk.word_tokenize(ansline)
    resl = nltk.word_tokenize(resultline)
    datal = nltk.word_tokenize(dataline)
    ansset=set(nltk.word_tokenize(ansline))
    resultset=set(nltk.word_tokenize(resultline))
    dataset = set(nltk.word_tokenize(dataline))
    if ansset==resultset:
        count+=1
    else:
        try:
            for j in range(len(ansl)):
                if ansl[j] != resl[j]:
                    print(int(num), '\n\terror word:  \t', datal[j], '\n\tcorrect answer:\t', resl[j], '\n\tyour answer:   \t', ansl[j], '\n')
                    print('\t', dataline)
        except:
            print('===Different Length===')
            print(ansline)
            print(resultline)
            print(dataline)
            print('======')
#        print(dataline)
print("Accuracy is : %.2f%%" % (count*1.00/10))
