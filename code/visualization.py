from datahandler import DataHandler
from HMM import unsupervised_HMM
import pickle

VERSES = ['2quatrain', 'volta', 'couplet']
READ_FOLDER = 'modelsToLoad/'
WRITE_FOLDER = 'modelsSaved/'

dh = DataHandler()
X = {}
X[VERSES[0]],X[VERSES[1]],X[VERSES[2]] = dh.get_data()

topN = 10
for verse in VERSES:
	if verse is not 'couplet':
		continue

	X_processed,X_conversion = dh.quantify_observations(X[verse])
	#HMM = pickle.load(open(READ_FOLDER+'300/HMM_'+verse+'.p', 'rb'))
	HMM = pickle.load(open(READ_FOLDER+'HMM_'+verse+'.p', 'rb'))

	# # transition probabilities
	# for state in range(HMM.L):
	# 	print state
	# 	print ['%.2f' % prob for prob in HMM.A[state]]


	# top words for each state
	for state in range(HMM.L):
		probs = HMM.O[state]
		ranking = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:topN]
		print('STATE: ' + str(state))
		for i in range(topN):
			print(X_conversion[ranking[i]] + '\t%.5f' % probs[ranking[i]])
