import json
import numpy as np
import pprint as pp
from MMM import MMM

# read example data from JSON
with open('data/example.json') as f:
    data = json.load(f)
initial_pi = (data['initial_pi'])
trained_pi = data['trained_pi']
input_x = data['input']

# read dictionary data from JSON
with open('data/ICGC-BRCA.json') as f1:
    dic_data = json.load(f1)

# TODO: finish parsing the ICGC file

# read signatures array from BRCA-signatures.npy
# this is an array of 12x96 - [i,j] is e_ij
signatures_data = np.load("data/BRCA-signatures.npy")

mmm = MMM(signatures_data,initial_pi,input_x)

# set lengths
n = mmm.getN()
m = mmm.getM()


