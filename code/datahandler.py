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
        self.rhymes = []

        self.read_data()

    def read_data(self, save_rhymes = False):
        file = open(self.data_file, 'r')
        sonnetCount = 1
        lineCount = -1
        if save_rhymes:
            lastWords = []
            rhymes = []

        for line in file:
            if line == '\n':
                continue
            if str(sonnetCount) in line:
                sonnetCount += 1
                if sonnetCount == 126:
                    sonnetCount = 127
                lineCount = 1
                continue
            if lineCount > 0:
                sequence = line.strip().lower().split(' ')
                sequence = list(reversed(self.remove_punctuation(sequence)))
                if save_rhymes:
                    lastWords.append(sequence[0])
                if lineCount >= 1 and lineCount <= 8:
                    self.X_2quatrains.append(sequence)
                elif lineCount >= 9 and lineCount <= 12:
                    self.X_volta.append(sequence)
                else:
                    self.X_couplet.append(sequence)

                lineCount += 1
                if lineCount > self.n_lines:
                    lineCount = -1
                    if save_rhymes:
                        rhymes.append({lastWords[0],lastWords[2]})
                        rhymes.append({lastWords[1], lastWords[3]})
                        rhymes.append({lastWords[4], lastWords[6]})
                        rhymes.append({lastWords[5], lastWords[7]})
                        rhymes.append({lastWords[8], lastWords[10]})
                        rhymes.append({lastWords[9], lastWords[11]})
                        rhymes.append({lastWords[12], lastWords[13]})
                        for pair in rhymes:
                            self.rhymes.append(pair)
                        rhymes = []
                        # tmp_rhymes = []
                        # for pair in rhymes:
                        #     print(pair)
                        #     if len(self.rhymes) > 0:
                        #         for sets in self.rhymes:
                        #             intersection = pair & sets
                        #             if len(intersection) == 0:
                        #                 tmp_rhymes.append(pair)
                        #             else:
                        #                 tmp_rhymes.append(intersection)
                        #                 #print(sets)
                        #     else:
                        #         tmp_rhymes.append(pair)
                        # self.rhymes = tmp_rhymes
                        # tmp_rhymes = []
                        # rhymes = []
                        # lastWords = []


        
    def get_data(self):
        return (self.X_2quatrains, self.X_volta, self.X_couplet)

    def get_rhymes(self):
        self.read_data(save_rhymes = True)
        return (self.rhymes)

    def remove_punctuation(self,sequence):
        for i in range(len(sequence)):
            word = sequence[i]
            if word[-1] in ['?', ',', '.', '!', ':', ';', ')']:
                sequence[i] = sequence[i][:-1]
            if word[0] in ['(']:
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

