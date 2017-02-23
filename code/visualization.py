from datahandler import DataHandler
from HMM import unsupervised_HMM
import pickle
import networkx as nx
import matplotlib.pyplot as plt
import pylab

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
	#HMM = pickle.load(open(READ_FOLDER+'600/HMM_'+verse+'.p', 'rb'))
	HMM = pickle.load(open(READ_FOLDER+'HMM_'+verse+'.p', 'rb'))

	G = nx.DiGraph()

	for state in range(HMM.L):

	# 	for nxt in range(HMM.L):
	# 		if HMM.A[state][nxt] > 0:
	# 			weight = '%.2f' % HMM.A[state][nxt]
	# 			G.add_edge(state,nxt,weight=weight)
	# print G.number_of_edges()
	# edge_labels=dict([((u,v,),d['weight']) for u,v,d in G.edges(data=True)])
	# pos=nx.spring_layout(G)
	# nx.draw_networkx_edge_labels(G,pos,edge_labels=edge_labels)
	# nx.draw_networkx_labels(G,pos)
	# nx.draw(G,pos,node_size=1500)
	# pylab.show()

		print state
		truncatedProbs = filter(lambda a: a > 0.0, HMM.A[state])
		print ['%.4f' % prob for prob in truncatedProbs]
		# print ['%.2f' % prob for prob in HMM.A[state]]
		# print filter(lambda a: a > 0.0, truncatedProbs)

	for state in range(HMM.L):
		probs = HMM.O[state]
		ranking = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:topN]
		print('STATE: ' + str(state))
		for i in range(topN):
			print(X_conversion[ranking[i]] + '\t%.8f' % probs[ranking[i]])




