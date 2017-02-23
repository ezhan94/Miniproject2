from datahandler import DataHandler
from HMM import unsupervised_HMM
import random
import pickle

VERSES = ['2quatrain', 'volta', 'couplet']
VERSE_LENGTH = {'2quatrain' : 8, 'volta': 4, 'couplet': 2}
READ_FOLDER = 'modelsToLoad/'
WRITE_FOLDER = 'modelsSaved/'

trainHMM = False

######################################################################
############################### MAIN #################################
######################################################################

dh = DataHandler()
X = {}
X[VERSES[0]],X[VERSES[1]],X[VERSES[2]] = dh.get_data()
rhymes = {}
rhymes[VERSES[0]],rhymes[VERSES[1]],rhymes[VERSES[2]] = dh.get_rhymes()

rhy = []
for verse in VERSES:
    num_groups = len(rhymes[verse])
    for i in range(VERSE_LENGTH[verse]/2):
        group = random.randrange(num_groups)
        rhyme_pair = random.sample(rhymes[verse][group], 2)
        rhy.append(rhyme_pair[0])
        rhy.append(rhyme_pair[1])

seeds = {}
seeds[VERSES[0]] = [rhy[0],rhy[2],rhy[1],rhy[3],rhy[4],rhy[6],rhy[5],rhy[7]] # quatrain seeds
seeds[VERSES[1]] = [rhy[8],rhy[10],rhy[9],rhy[11]] # volta seeds
seeds[VERSES[2]] = [rhy[12],rhy[13]] # couplet seeds

for verse in VERSES:
    X_processed,X_conversion = dh.quantify_observations(X[verse])

    if trainHMM:
        HMM = unsupervised_HMM(X_processed,30)
        pickle.dump(HMM, open(WRITE_FOLDER+'HMM_'+verse+'.p', 'wb'))
    else:
        HMM = pickle.load(open(READ_FOLDER+'HMM_'+verse+'.p', 'rb'))

    for word in seeds[verse]:
        seed_num = X_conversion.index(word)
        emission = HMM.generate_emission(seed_num,8)
        print dh.convert_to_sentence(emission,X_conversion)

