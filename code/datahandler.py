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
        self.X_2quatrains = []
        self.X_volta = []
        self.X_couplet = []
        self.rhymes = []

        self.read_data()
        print len(self.X_2quatrains)
        print len(self.X_volta)
        print len(self.X_couplet)


    def read_data(self):
        file = open(self.data_file, 'r')
        sonnetCount = 1
        lineCount = -1
        lastWords = []

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
                    self.rhymes.append([lastWords[0],lastWords[2]])
                    self.rhymes.append([lastWords[1], lastWords[3]])
                    self.rhymes.append([lastWords[4], lastWords[6]])
                    self.rhymes.append([lastWords[5], lastWords[7]])
                    self.rhymes.append([lastWords[8], lastWords[10]])
                    self.rhymes.append([lastWords[9], lastWords[11]])
                    self.rhymes.append([lastWords[12], lastWords[13]])
                    lastWords = []
        

    def remove_punctuation(self,sequence):
        for i in range(len(sequence)):
            word = sequence[i]
            if word[-1] in ['?', ',', '.', '!', ':', ';', ')']:
                sequence[i] = sequence[i][:-1]
            if word[0] in ['(']:
                sequence[i] = sequence[i][1:]
        return sequence

