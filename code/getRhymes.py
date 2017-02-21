from datahandler import DataHandler
from HMM import unsupervised_HMM
import  csv

dh = DataHandler()
rhymes = dh.get_rhymes()
list_of_rhymes = []
for sets in rhymes:
    list_of_rhymes.append(list(sets))

with open("rhymes.csv","w") as f:
    wr = csv.writer(f)
    wr.writerows(list_of_rhymes)

