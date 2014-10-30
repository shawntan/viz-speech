import numpy as np

import theano
import theano.tensor as T

from theano_toolkit import updates
from theano_toolkit import utils as U 
from theano_toolkit.parameters import Parameters

coeff = np.ones((1,2,2),dtype=np.float32)
coeff[0,0,1] = -1
pi = 0.5 * np.pi * np.array([[[0,1],[1,0]]],dtype=np.float32)
coeff = T.patternbroadcast(theano.shared(coeff),(True,False,False))
pi    = T.patternbroadcast(theano.shared(pi),(True,False,False))
def rotation(theta):
	theta = T.patternbroadcast(theta.reshape((theta.shape[-1],1,1)),(False,True,True))
	return T.cos(pi + coeff * theta)
def gaussian(P,rows,cols,components):
	input_size = rows * cols
	points = theano.shared(np.asarray(
		np.dstack(
			np.meshgrid(np.arange(cols), np.arange(rows))
			).reshape(input_size,2),
		dtype=np.float32)
		)
	P.g_mean   = np.random.rand(components,2) * np.array([rows,cols])
	P.g_scale  = 5 * np.random.rand(components,2) 
	P.g_thetas = 2 * np.pi * np.random.rand(components)

	shifted = T.patternbroadcast(points.reshape((input_size,1,2)),(False,True,False))\
			- T.patternbroadcast(P.g_mean.reshape((1,components,2)),(True,False,False))
	rot     = rotation(P.g_thetas)
	scale = T.patternbroadcast(P.g_scale.reshape((components,2,1)),(False,False,True))
	B = T.patternbroadcast((rot/scale).reshape((1,components,2,2)),(True,False,False,False))
	decorr = T.sum(
			B * T.patternbroadcast(shifted.reshape((input_size,components,1,2)),(False,False,True,False)),
			axis = 3
			)
	Z = T.sum(decorr ** 2,axis=2)
	return T.exp(-Z)


