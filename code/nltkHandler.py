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
        
        (w, changedSylls, changedEmph) = self.cleanWord(word)
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
        if not changedSylls==0:
            isEmph=0
            numSyll += changedSylls

        if changedEmph:
            if isEmph==0:
                isEmph = 1
            else:
                isEmph = 0
        
        return (numSyll, isEmph) 
    
    def cleanWord(self,word):
        
        changedSylls = 0
        changedEmph = False 
        
        w = word.lower()
        
        if 'o\'er' in w:
            if self.cmuDict.has_key(w.replace('o\'er', 'over')):
                w = w.replace('o\'er', 'over')
                changedSylls = -1
        
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
        
        if '\'' in w:
            if self.cmuDict.has_key(w.replace('\'','e')):
                w = w.replace('\'','e')
                changedSylls = -1
        
        #ed, eth, er: test in dict; if not remove, test in dict, else add e, test
        if word[-2:]=='ed':#'ed' in w: # and not self.cmuDict.has_key(w):
            if self.cmuDict.has_key(w.replace('ed','')):
                w = w.replace('ed','')
                changedSylls = 1
                changedEmph = True
            elif self.cmuDict.has_key(w.replace('ed','')+'e'):
                w = w.replace('ed','')+'e'
                changedSylls = 1
                changedEmph = True
        if 'eth' in w and not self.cmuDict.has_key(w):
            if self.cmuDict.has_key(w.replace('eth','')):
                w = w.replace('eth','')
                changedSylls = 1
                changedEmph = True
            elif self.cmuDict.has_key(w.replace('eth','')+'e'):
                w = w.replace('eth','')+'e'
                changedSylls = 1
                changedEmph = True
        if 'er' in w and not self.cmuDict.has_key(w):
            if self.cmuDict.has_key(w.replace('er','')):
                w = w.replace('er','')
                changedSylls = 1
                changedEmph = True
            elif self.cmuDict.has_key(w.replace('er','')+'e'):
                w = w.replace('er','')+'e'
                changedSylls = 1 
                changedEmph = True
                
        #ly, less, ness: test in dict, if not remove, test in dict
        if 'ly' in w and not self.cmuDict.has_key(w):
            if self.cmuDict.has_key(w.replace('ly','')):
                w = w.replace('ly','')
                changedSylls = 1
                changedEmph = True
        if 'less' in w and not self.cmuDict.has_key(w):
            if self.cmuDict.has_key(w.replace('less','')):
                w = w.replace('less','')
                changedSylls = 1
                changedEmph = True
        if 'ness' in w and not self.cmuDict.has_key(w):
            if self.cmuDict.has_key(w.replace('ness','')):
                w = w.replace('ness','')
                changedSylls = 1
                changedEmph = True
        
        #ou: test in dict, if not, swap with o, test in dict
        if 'ou' in w and not self.cmuDict.has_key(w):
            if self.cmuDict.has_key(w.replace('ou','o')):
                w = w.replace('ou','o')
            else:
                ww = self.cleanWord(w.replace('ou','o'))
                if self.cmuDict.has_key(ww):
                    w = ww
        #remove s at end
        if w[-1]=='s' and not self.cmuDict.has_key(w):
            if self.cmuDict.has_key(w[:-1]):
                w = w[:-1]
            else:
                ww = self.cleanWord(w[:-1])
                if self.cmuDict.has_key(ww):
                    w = ww
        
        #add extra syll for ed:
        #if word[-2:]=='ed' and not self.cmuDict.has_key(word):
        #    changedSylls+=1
        #    changedEmph = True
        
        return (w, changedSylls, changedEmph)
