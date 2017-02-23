from datahandler import DataHandler
from HMM import unsupervised_HMM
import random
import pickle

dh = DataHandler()
X_2quatrains,X_volta,X_couplet = dh.get_data()
rhymes_q,rhymes_v,rhymes_c = dh.get_rhymes()
rhy = []
num_groups = len(rhymes_q)
for i in range(4):
    group = random.randrange(num_groups)
    rhyme_pair = random.sample(rhymes_q[group], 2)
    rhy.append(rhyme_pair[0])
    rhy.append(rhyme_pair[1])
num_groups = len(rhymes_v)
for i in range(2):
    group = random.randrange(num_groups)
    rhyme_pair = random.sample(rhymes_v[group], 2)
    rhy.append(rhyme_pair[0])
    rhy.append(rhyme_pair[1])
num_groups = len(rhymes_c)
for i in range(1):
    group = random.randrange(num_groups)
    rhyme_pair = random.sample(rhymes_c[group], 2)
    rhy.append(rhyme_pair[0])
    rhy.append(rhyme_pair[1])

quatrain_seeds = [rhy[0],rhy[2],rhy[1],rhy[3],rhy[4],rhy[6],rhy[5],rhy[7]]
volta_seeds = [rhy[8],rhy[10],rhy[9],rhy[11]]
couplet_seeds = [rhy[12],rhy[13]]

readFolder = 'modelsToLoad/'
writeFolder = 'modelsSaved/'

X_processed,X_conversion = dh.quantify_observations(X_2quatrains)
HMM = pickle.load( open( readFolder+'HMM_2quatrain.p', 'rb' ) )
#HMM = unsupervised_HMM(X_processed,30)
#pickle.dump( HMM, open( writeFolder+'HMM_2quatrain.p', 'wb' ) )
for word in quatrain_seeds:
    seed_num = X_conversion.index(word)
    emission = HMM.generate_emission(seed_num,8,X_conversion)
    print dh.convert_to_sentence(emission,X_conversion)

X_processed,X_conversion = dh.quantify_observations(X_volta)
HMM = pickle.load( open( readFolder+'HMM_volta.p', 'rb' ) )
#HMM = unsupervised_HMM(X_processed,30)
#pickle.dump( HMM, open( writeFolder+'HMM_volta.p', 'wb' ) )
for word in volta_seeds:
    seed_num = X_conversion.index(word)
    emission = HMM.generate_emission(seed_num,8,X_conversion)
    print dh.convert_to_sentence(emission,X_conversion)

X_processed,X_conversion = dh.quantify_observations(X_couplet)
HMM = pickle.load( open( readFolder+'HMM_couplet.p', 'rb' ) )
#HMM = unsupervised_HMM(X_processed,30)
#pickle.dump( HMM, open( writeFolder+'HMM_couplet.p', 'wb' ) )
for word in couplet_seeds:
    seed_num = X_conversion.index(word)
    emission = HMM.generate_emission(seed_num,8,X_conversion)
    print dh.convert_to_sentence(emission,X_conversion)
