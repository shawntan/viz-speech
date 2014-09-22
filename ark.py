import numpy as np
def load_ark(filename):
	result = None
	with open(filename) as f:
		f.next()
		result = [ [float(w) for w in line.strip(' []\n').split(' ')] for line in f ]
	return np.array(result,dtype=np.float32)


