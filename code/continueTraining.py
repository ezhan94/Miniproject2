from datahandler import DataHandler
from HMM import unsupervised_HMM
import pickle

VERSES = ['2quatrain', 'volta', 'couplet']
READ_FOLDER = 'modelsToLoad/'
WRITE_FOLDER = 'modelsSaved/'

dh = DataHandler()
X = {}
X[VERSES[0]],X[VERSES[1]],X[VERSES[2]] = dh.get_data()

for verse in VERSES:
    X_processed,X_conversion = dh.quantify_observations(X[verse])
    HMM = pickle.load(open(READ_FOLDER+'HMM_'+verse+'.p', 'rb'))
    HMM.unsupervised_learning(X_processed)
    pickle.dump(HMM, open(WRITE_FOLDER+'HMM_'+verse+'.p', 'wb'))
