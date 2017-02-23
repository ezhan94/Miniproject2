import nltk
from nltk.corpus import cmudict

class NltkHandler(object):

    def __init__(self):
        self.cmuDict = cmudict.dict()

    def getDict(self, wordList):
        infodict = {}
        for word in wordList:
            infodict[word] = self.numSyll_isEmph(word)
        
        return infodict
        
        
    def numSyll_isEmph(self,word):
        
        if not self.cmuDict.has_key(word):
            return (1, 1)
        
        numSyll = [len(list(y for y in x if y[-1].isdigit())) for x in self.cmuDict[word.lower()]][0]
        if numSyll==1:
            isEmph=1
        else:
            lastSyl = [list(y for y in x if y[-1].isdigit()) for x in self.cmuDict[word.lower()]][0][-1]
            isEmph = int(lastSyl[-1])
        return (numSyll, isEmph) 
