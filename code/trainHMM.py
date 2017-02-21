from datahandler import DataHandler
from HMM import unsupervised_HMM

dh = DataHandler()
X_2quatrains,X_volta,X_couplet = dh.get_data()

X_couplet_processed,X_couplet_conversion = dh.quantify_observations(X_couplet)
HMM = unsupervised_HMM(X_couplet_processed,30)

emission = HMM.generate_emission(16,8)
print dh.convert_to_sentence(emission,X_couplet_conversion)