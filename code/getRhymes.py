from datahandler import DataHandler
from HMM import unsupervised_HMM
import  csv

dh = DataHandler()
rhymes = dh.get_rhymes()

for set in rhymes:
    set = list(set)

with open("rhymes.csv","w") as f:
    wr = csv.writer(f)
    wr.writerows(rhymes)

