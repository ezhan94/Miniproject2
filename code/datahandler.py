# Kaggle Competition
# Python 2.7
# datahandler.py
# reads/writes/handles data from the voter data csv files

import csv
import numpy as np
import sklearn.preprocessing as proc
from sklearn.utils import shuffle

class DataHandler(object):

    def __init__(self,readData=True,remove_id=True,remove_blanks=False):
        self.d = 382 # dimension of data
        self.n_train = 64667 # number of training samples
        self.n_test = 16000 # numer of test samples
        self.n_test2012 = 82820 # numer of test samples

        self.train_file = 'data/train_2008.csv'
        self.test_file = 'data/test_2008.csv'
        self.test2012_file = 'data/test_2012.csv'
        self.feature_scores_file = 'data/feature_scores.csv'
        self.write_folder = 'submissions/'
        self.header = ['id','PES1']

        self.training_data = np.zeros((self.n_train,self.d+1)) # +1 because training data also has the label
        self.test_data = np.zeros((self.n_test,self.d))
        self.test2012_data = np.zeros((self.n_test2012,self.d))

        if readData:
            self.read_training_data()
            self.read_test_data()
            self.read_test2012_data()
            self.training_data = shuffle(self.training_data,random_state=0)
            
            if remove_id:
                print 'Removing ID column ...'
                self.training_data = np.delete(self.training_data,[0],1)
                self.test_data = np.delete(self.test_data,[0],1)
                self.test2012_data = np.delete(self.test2012_data,[0],1)
                self.d = 381

            if remove_blanks:
                X = self.training_data
                N = len(X[1,:])
                ids = []
                for i in range(N):
                    unique, counts = np.unique(X[:,i], return_counts=True)
                    responses = dict(zip(unique,counts))
                    if -1 in unique:
                        if -2 not in unique:
                            responses[-2] = 0
                        if -3 not in unique:
                            responses[-3] = 0
                        if responses[-1]+responses[-3]+responses[-2] >= 60000:
                            ids.append(i)
                ids = np.array(ids)
                self.training_data = np.delete(self.training_data,ids,1)
                self.test_data = np.delete(self.test_data,ids,1)
                self.d -= len(ids)

        print 'DataHandler() initialized.'

    def read_training_data(self):
        print 'Reading training data ...'
        reader = csv.reader(open(self.train_file, 'r'))
        labels = next(reader)
        i = 0
        for row in reader:
            self.training_data[i] = row
            i += 1

    def read_test_data(self):
        print 'Reading test data ...'
        reader = csv.reader(open(self.test_file, 'r'))
        labels = next(reader)
        i = 0
        for row in reader:
            self.test_data[i] = row
            i += 1

    def read_test2012_data(self):
        print 'Reading test 2012 data ...'
        reader = csv.reader(open(self.test2012_file, 'r'))
        labels = next(reader)
        i = 0
        for row in reader:
            self.test2012_data[i] = row
            i += 1

    def get_data(self):
        X_train = self.training_data[:,:-1]
        Y_train = self.training_data[:,-1]
        X_test = self.test_data
        X_test2012 = self.test2012_data
        return (X_train,Y_train,X_test,X_test2012)

    def get_data_filtered(self,remove=[],normalize=False):
        filtered_data = np.delete(self.training_data,remove,1)
        X_train = filtered_data[:,:-1]
        Y_train = filtered_data[:,-1]
        X_test = np.delete(self.test_data,remove,1)
        X_test2012 = np.delete(self.test2012_data,remove,1)
        if(normalize):
            max_abs_scaler = proc.MaxAbsScaler()
            X_train = max_abs_scaler.fit_transform(X_train)
            X_test = max_abs_scaler.transform(X_test)
            X_test2012 = max_abs_scaler.transform(X_test2012)
        return (X_train,Y_train,X_test,X_test2012)

    def get_data_topN(self,N,normalize=False):
        to_remove = self.get_bottom_features(N)
        return self.get_data_filtered(remove=to_remove,normalize=normalize)

    def get_bottom_features(self,threshold,threshold_type='number'):
        num_remove = 0

        if threshold_type == 'number':
            num_remove = self.d-threshold
        elif threshold_type == 'fraction':
            num_remove = int(self.d*(1-threshold))
        else:
            print 'ERROR: invalid threshold type'
            return

        print 'Removing %d features ...' % num_remove
        reader = csv.reader(open(self.feature_scores_file, 'r'))
        bottom = [0]*num_remove
        for i in range(num_remove):
            row = next(reader)
            bottom[i] = row[0]

        return bottom

    def filter_topN(self,train,test,N):
        remove = self.get_bottom_features(N)
        return (np.delete(train,remove,1),np.delete(test,remove,1))

    def write_feature_scores(self,index,scores,overwrite=False):
        if not overwrite:
            print 'ERROR: about to overwrite feature scores. Are you sure???'
            return

        with open(self.feature_scores_file, 'wb') as csvfile:
            writer = csv.writer(csvfile,delimiter=',')
            for i in range(len(index)):
                writer.writerow([index[i], scores[index[i]]])

    def write_predictions(self,predictions,filename):
        if len(predictions) != self.n_test: 
            print('ERROR: Invalid number of predictions ',len(predictions))
            return
        if filename[-4:] != '.csv':
            print('ERROR: file needs to end in .csv')
            return
        
        with open(self.write_folder+filename, 'wb') as csvfile:
            writer = csv.writer(csvfile,delimiter=',')
            writer.writerow(self.header)
            for i in range(self.n_test):
                writer.writerow([str(i),str(int(predictions[i]))])

    def write_predictions2012(self,predictions,filename):
        if len(predictions) != self.n_test2012: 
            print('ERROR: Invalid number of predictions ',len(predictions))
            return
        if filename[-4:] != '.csv':
            print('ERROR: file needs to end in .csv')
            return
        
        with open(self.write_folder+filename, 'wb') as csvfile:
            writer = csv.writer(csvfile,delimiter=',')
            writer.writerow(self.header)
            for i in range(self.n_test2012):
                writer.writerow([str(i),str(int(predictions[i]))])

    def get_dimension(self):
        return self.d

    def get_num_training_samples(self):
        return self.n_train

    def get_num_test_samples(self):
        return self.n_test

######################## Obsolete stuff, remove later ##################

    def compare_features(self,blank_list):
        reader = csv.reader(open(self.feature_scores_file, 'r'))
        for i in range(self.d):
            row = next(reader)
            if int(row[0]) in blank_list:
                print i+1


