import nltk
from nltk.corpus import cmudict
import math

class NltkHandler(object):

    def __init__(self):
        self.cmuDict = cmudict.dict()
        self.estWordLength = 4.0

    def getDict(self, wordList):
        infodict = {}
        for word in wordList:
            infodict[word] = self.numSyll_isEmph(word)
        
        return infodict
        
        
    def numSyll_isEmph(self,word):
        
        words = word.split('-')
        if len(words)>1:
            combinedInfo = []
            for wrd in words:
                combinedInfo.append(self.numSyll_isEmph(wrd))
            
            numSyll = sum([info[0] for info in combinedInfo])
            isEmph = combinedInfo[-1][1]
            
            return (numSyll, isEmph)
        
        (w, changed) = self.cleanWord(word)
        
        if not self.cmuDict.has_key(w):
            print (word, w)
            numSyll = int(math.ceil(len(w)/self.estWordLength))
            return (numSyll, 1)
        
        
        numSyll = [len(list(y for y in x if y[-1].isdigit())) for x in self.cmuDict[w]][0]
        if numSyll==1:
            isEmph=1
        else:
            lastSyl = [list(y for y in x if y[-1].isdigit()) for x in self.cmuDict[w]][0][-1]
            isEmph = int(lastSyl[-1])
            
            
        #check orig for 'ed' or 'eth' or 'ly'
        #if (that's in the original word) and changed:
            #isEmph=0
            #numSyll +=1
        
        return (numSyll, isEmph) 
    
    def cleanWord(self,word):
        
        changed = False
        
        w = word.lower()
        
        #'s, 't: remove 
        #'st: remove, test in dict, else add e, test, if not, give up
        
        #ed, eth: test in dict; if not remove, test in dict, else add e, test
        
        #ou: test in dict, if not, swap with o, test in dict
        
        #ly: test in dict, if not remove, test in dict
        
        
        return (w, changed)
