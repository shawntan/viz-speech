import math
import theano
import theano.tensor as T
import numpy         as np
import matplotlib.pyplot as plt
from theano_toolkit import utils as U
from theano_toolkit import updates
from theano_toolkit.parameters import Parameters

def adjacency_constraint(lin_activations):
	size = lin_activations.shape[1]
	constraint = T.eye(size)[:,1:] - T.eye(size+1)[1:,1:-1]
	return T.mean(T.sum(abs(T.dot(lin_activations,constraint)),axis=1)) + T.mean(abs(lin_activations[:,0]-lin_activations[:,-1]))

def build_error(X,output,P):
	return T.mean(-T.sum(X * T.log(output) + (1 - X) * T.log(1 - output), axis=1))


def build_network(input_size,hidden_size,constraint_adj=False):
	P = Parameters()
	X = T.bmatrix('X')
	
	P.W_input_hidden = U.initial_weights(input_size,hidden_size)
	P.b_hidden       = U.initial_weights(hidden_size)
	P.b_output       = U.initial_weights(input_size)
	hidden_lin = T.dot(X,P.W_input_hidden)+P.b_hidden
	hidden = T.nnet.sigmoid(hidden_lin)
	output = T.nnet.softmax(T.dot(hidden,P.W_input_hidden.T) + P.b_output)
	parameters = P.values() 
	cost = build_error(X,output,P) 
	if constraint_adj:pass
		#cost = cost + adjacency_constraint(hidden_lin)

	return X,output,cost,P
def hinton(matrix, max_weight=None, ax=None):
	"""Draw Hinton diagram for visualizing a weight matrix."""
	ax = ax if ax is not None else plt.gca()

	if not max_weight:
		max_weight = 2**np.ceil(np.log(np.abs(matrix).max())/np.log(2))

	ax.patch.set_facecolor('gray')
	ax.set_aspect('equal', 'box')
	ax.xaxis.set_major_locator(plt.NullLocator())
	ax.yaxis.set_major_locator(plt.NullLocator())

	for (x,y),w in np.ndenumerate(matrix):
		color = 'white' if w > 0 else 'black'
		size = np.sqrt(np.abs(w))
		rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
				facecolor=color, edgecolor=color)
		ax.add_patch(rect)

	ax.autoscale_view()
	ax.invert_yaxis()
	return plt.figure()

def run_experiment(constraint_adj=False):
	X,output,cost,P= build_network(8,3,constraint_adj)
	parameters = P.values()
	grads = T.grad(cost,wrt=parameters)
	train = theano.function(
			inputs=[X],
			outputs=cost,
			updates=updates.adadelta(parameters,grads)
			)
	test = theano.function(
			inputs=[X],
			outputs=output,
			)
	data = np.eye(8,dtype=np.int8)
#	data = np.vstack((data,))
	print "Training..."
	for _ in xrange(100000):
		np.random.shuffle(data)
		train(data)

	hidden_activations = theano.function(
			inputs=[X],
			outputs=T.nnet.sigmoid(T.dot(X,P.W_input_hidden)+P.b_hidden)
		)
	#print_arr(test(np.eye(8,dtype=np.int32)))
	#print_arr(1/(1 + np.exp(-parameters[0].get_value())),1)
	return hinton(hidden_activations(np.eye(8,dtype=np.int8)))

