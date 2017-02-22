# MiniProject2
# Python 2.7
# datahandler.py
# reads/writes/handles Shakespeare sonnets

import numpy as np

class DataHandler(object):

    def __init__(self):
        self.n_sonnets = 153 # removed sonnet 126
        self.n_lines = 14
        self.data_file = 'data/shakespeare.txt'
        self.X_2quatrains = [] # final length = 1224
        self.X_volta = [] # final length = 612
        self.X_couplet = [] # final length = 306
        self.rhymes_2quatrains = []
        self.rhymes_volta = []
        self.rhymes_couplet = []

        self.read_data(save_rhymes=True)

    def read_data(self, save_rhymes=False):
        file = open(self.data_file, 'r')
        sonnetCount = 1
        lineCount = -1
        if save_rhymes:
            lastWords_quatrain = []
            lastWords_volta = []
            lastWords_couplet = []
            rhymes = []


        for line in file:
            if line == '\n':
                continue
            if str(sonnetCount) in line:
                sonnetCount += 1
                if sonnetCount == 126:
                    sonnetCount = 127
                elif sonnetCount == 99:
                    sonnetCount = 100
                lineCount = 1
                continue
            if lineCount > 0:
                sequence = line.strip().lower().split(' ')
                sequence = list(reversed(self.remove_punctuation(sequence)))

                if lineCount >= 1 and lineCount <= 8:
                    self.X_2quatrains.append(sequence)
                    if save_rhymes:
                        lastWords_quatrain.append(sequence[0])
                elif lineCount >= 9 and lineCount <= 12:
                    self.X_volta.append(sequence)
                    if save_rhymes:
                        lastWords_volta.append(sequence[0])
                else:
                    self.X_couplet.append(sequence)
                    if save_rhymes:
                        lastWords_couplet.append(sequence[0])

                lineCount += 1
                if lineCount > self.n_lines:
                    lineCount = -1
                    if save_rhymes:
                        rhymes = []
                        rhymes.append({lastWords_quatrain[0],lastWords_quatrain[2]})
                        rhymes.append({lastWords_quatrain[1], lastWords_quatrain[3]})
                        rhymes.append({lastWords_quatrain[4], lastWords_quatrain[6]})
                        rhymes.append({lastWords_quatrain[5], lastWords_quatrain[7]})
                        lastWords_quatrain = []
                        tmp_rhymes = []
                        added = False
                        for pair in rhymes:
                            for group in self.rhymes_2quatrains:
                                if(added == False):
                                    if len(pair & group) == 0:
                                        tmp_rhymes.append(group)
                                    else:
                                        tmp_rhymes.append(pair | group)
                                        added = True
                                else:
                                    tmp_rhymes.append(group)
                            if(added == False):
                                tmp_rhymes.append(pair)
                            added = False
                            self.rhymes_2quatrains = tmp_rhymes
                            tmp_rhymes = []
                        rhymes = []
                        rhymes.append({lastWords_volta[0],lastWords_volta[2]})
                        rhymes.append({lastWords_volta[1], lastWords_volta[3]})
                        lastWords_volta = []
                        tmp_rhymes = []
                        added = False
                        for pair in rhymes:
                            for group in self.rhymes_volta:
                                if(added == False):
                                    if len(pair & group) == 0:
                                        tmp_rhymes.append(group)
                                    else:
                                        tmp_rhymes.append(pair | group)
                                        added = True
                                else:
                                    tmp_rhymes.append(group)
                            if(added == False):
                                tmp_rhymes.append(pair)
                            added = False
                            self.rhymes_volta = tmp_rhymes
                            tmp_rhymes = []
                        rhymes = []
                        rhymes.append({lastWords_couplet[0], lastWords_couplet[1]})
                        lastWords_couplet = []
                        tmp_rhymes = []
                        added = False
                        for pair in rhymes:
                            for group in self.rhymes_couplet:
                                if (added == False):
                                    if len(pair & group) == 0:
                                        tmp_rhymes.append(group)
                                    else:
                                        tmp_rhymes.append(pair | group)
                                        added = True
                                else:
                                    tmp_rhymes.append(group)
                            if (added == False):
                                tmp_rhymes.append(pair)
                            added = False
                            self.rhymes_couplet = tmp_rhymes
                            tmp_rhymes = []

        
    def get_data(self):
        return (self.X_2quatrains, self.X_volta, self.X_couplet)

    def get_rhymes(self):
        #self.read_data(save_rhymes = True)
        return (self.rhymes_2quatrains,self.rhymes_volta,self.rhymes_couplet)

    def remove_punctuation(self,sequence):
        for i in range(len(sequence)):
            word = sequence[i]
            if word[-1] in ['?', ',', '.', '!', ':', ';', ')', "'"]:
                sequence[i] = sequence[i][:-1]
            if word[0] in ['(', "'"]:
                sequence[i] = sequence[i][1:]
        return sequence

    def quantify_observations(self,X):
        obsList = []
        newX = []

        for x in X:
            sequence = []
            for word in x:
                if word in obsList:
                    sequence.append(obsList.index(word))
                else:
                    obsList.append(word)
                    sequence.append(len(obsList)-1)
            newX.append(sequence)

        return (newX,obsList)

    def convert_to_sentence(self,emission,obsList):
        sentence = ''
        for w in reversed(emission):
            sentence += obsList[w]
            sentence += ' '
        return sentence[:-1]

