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
        
        (w, changedSylls) = self.cleanWord(word)
        #w = word.lower()
        
        if not self.cmuDict.has_key(w):
            #print (word, w)
            numSyll = int(math.ceil(len(w)/self.estWordLength))
            return (numSyll, 1)
        
        
        numSyll = [len(list(y for y in x if y[-1].isdigit())) for x in self.cmuDict[w]][0]
        if numSyll==1:
            isEmph=1
        else:
            lastSyl = [list(y for y in x if y[-1].isdigit()) for x in self.cmuDict[w]][0][-1]
            isEmph = int(lastSyl[-1])
            
            
        #check orig for 'ed' or 'eth' or 'ly' (actually, jk, the flag will tell you)
        if changedSylls:
            isEmph=0
            numSyll +=1
        
        return (numSyll, isEmph) 
    
    def cleanWord(self,word):
        
        changedSylls = False
        
        w = word.lower()
        
        #' at end: remove
        w = w.strip('\'')
        
        #'st: remove, test in dict, else add e, test, if not, give up
        if '\'st' in w:
            if self.cmuDict.has_key(w.replace('\'st','')):
                w = w.replace('\'st','')
            elif self.cmuDict.has_key(w.replace('\'st','')+'e'):
                w = w.replace('\'st','')+'e'
        
        #'s, 't: remove 
        if '\'s' in w:
            w = w.replace('\'s','')
        if '\'t' in w:
            w = w.replace('\'t','')
        
        #ed, eth: test in dict; if not remove, test in dict, else add e, test
        if 'ed' in w and not self.cmuDict.has_key(w):
            if self.cmuDict.has_key(w.replace('ed','')):
                w = w.replace('ed','')
                changedSylls = True
            elif self.cmuDict.has_key(w.replace('ed','')+'e'):
                w = w.replace('ed','')+'e'
                changedSylls = True
        if 'eth' in w and not self.cmuDict.has_key(w):
            if self.cmuDict.has_key(w.replace('eth','')):
                w = w.replace('eth','')
                changedSylls = True
            elif self.cmuDict.has_key(w.replace('eth','')+'e'):
                w = w.replace('eth','')+'e'
                changedSylls = True
        
        
        #ly: test in dict, if not remove, test in dict
        if 'ly' in w and not self.cmuDict.has_key(w):
            if self.cmuDict.has_key(w.replace('ly','')):
                w = w.replace('ly','')
                changedSylls = True
        
        #ou: test in dict, if not, swap with o, test in dict
        if 'ou' in w:
            if self.cmuDict.has_key(w.replace('ou','o')):
                w = w.replace('ou','o')
            else:
                ww = self.cleanWord(w.replace('ou','o'))
                if self.cmuDict.has_key(ww):
                    w = ww
                
        
        return (w, changedSylls)
