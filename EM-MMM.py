import json
import numpy as np
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

# TODO: finish parsing the ICGC file - i guess this is the data to run on, what should i use here?

# read signatures array from BRCA-signatures.npy
# this is an array of 12x96 - [i,j] is e_ij - fixed in this case until we change
signatures_data = np.load("data/BRCA-signatures.npy")

mmm = MMM(signatures_data, initial_pi, input_x)

mmm.fit(input_x, 0.01, 200)

# TODO: finish this function
print(mmm.likelihood(dic_data))
