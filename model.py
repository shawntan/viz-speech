import numpy as np
from theano_toolkit import utils as U
from theano import tensor as T
import theano
theano.config.floatX = 'float32'
def load_file(model_file):
	connections= []
	biases = []
	with open(model_file) as f:
		while True:
			try:
				line = f.next().strip()
				dim1,dim2 = None,None
				while True:
					if line.startswith('<affinetransform>'):
						_,dim2,dim1 = line.split(' ')
						break
					else:
						line = f.next().strip()
				f.next().strip()
				connection = np.zeros((int(dim1),int(dim2)))
				line = f.next().strip(' []\n')
				for i in range(int(dim2)):
					connection[:,i] = [ float(w) for w in line.split(' ') ]
					line = f.next().strip(' []\n')
					bias = np.array([ float(w) for w in line.split(' ') ])
				print connection.shape,bias.shape
				connections.append(connection)
				biases.append(bias)
			except StopIteration:
				break
	return connections,biases

def read_until(stream,condition):
	while True:
		line = stream.next()
		if condition(line): return line 
	

def load_train_file(model_file):
	parameters = []
	with open(model_file) as f:
		try:
			while True:
				oline = read_until(f,lambda x: x.strip().startswith("<LearnRateCoef>")).strip()
				weights = []
				while not oline.endswith("]"):
					oline = f.next().strip()
					line = oline.strip(" []\n")
					weights.append([float(x) for x in line.split()])
				parameters.append(np.array(weights).T)
				line = f.next().strip(" []\n")
				parameters.append(np.array([float(x) for x in line.split()]))
		except StopIteration:
			pass
	
	return parameters
		

def load(model_file):
	print "Loading model..."
	connections, biases = load_file(model_file)
	print "Loaded model."

	for i in range(len(connections)):
		connections[i] = U.create_shared(connections[i])
		biases[i]      = U.create_shared(biases[i])

	X = T.fmatrix('X')
	
	layers = [None]*len(connections)
	current = X
	for i in range(len(connections)-1):
		current = layers[i] = T.nnet.sigmoid(T.dot(current,connections[i]) + biases[i])
	i = len(connections)-1
	layers[i] = T.nnet.softmax(T.dot(current,connections[i]) + biases[i])

	predict = theano.function(
			inputs = [X],
			outputs = layers
		)
	
	return predict




	
	
	

