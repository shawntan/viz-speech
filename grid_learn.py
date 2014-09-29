import numpy as np
import theano.tensor as T
import theano
import matplotlib.pyplot as plt
from theano_toolkit import utils as U
from theano_toolkit import updates
from collections import Counter
from matplotlib import animation
from itertools import product
import matplotlib.cm as cm
import seaborn as sns
def build_model(hidden_size,predict_only=False):
	X = T.dmatrix('X')
	Y = T.ivector('Y')
			#* (0.001 * U.initial_weights(2,hidden_size) + np.array([[0,0,1,1],[1,1,0,0]])))
	W_input_hidden   = U.create_shared(U.initial_weights(2,hidden_size))
	b_hidden         = U.create_shared(U.initial_weights(hidden_size))
	W_hidden_predict = U.create_shared(U.initial_weights(hidden_size,2))
	b_predict        = U.create_shared(U.initial_weights(2))

	params = [W_input_hidden,b_hidden,W_hidden_predict,b_predict]
	
	hidden = T.nnet.sigmoid(T.dot(X,W_input_hidden) + b_hidden)
	predict = T.nnet.softmax(T.dot(hidden,W_hidden_predict) + b_predict)
	
	cost = -T.mean(T.log(predict[T.arange(Y.shape[0]),Y])) # + 1e-4 * sum(T.sum(p**2) for p in params)
	accuracy = T.mean(T.eq(T.argmax(predict,axis=1),Y))
	grad = T.grad(cost,params)
	
	train = theano.function(
			inputs = [X,Y],
			#updates =  updates.momentum(params,grad,0.9999,0.1) if not predict_only else None,
			#updates =  updates.momentum(params,grad,0.999,0.0005),
			updates =  updates.adadelta(params,grad),
			outputs = [accuracy,W_input_hidden,b_hidden,(hidden>0.5)]
		)
	predict = theano.function(
			inputs  = [X],
			outputs = predict[:,0]
		)

	return train,predict,params

def create_normal_data(line_count,point_count):
	points = []
	labels = []
	for i in xrange(line_count):
		for j in xrange(line_count):
			points.append(0.1*np.random.randn(point_count/4,2) +\
					np.array([i,j]) - 1.5)
			if i%2 == j%2:
				labels.append(np.zeros(point_count/4,dtype=bool))
			else:
				labels.append(np.ones(point_count/4,dtype=bool))

	return np.vstack(points),np.hstack(labels)




def create_data(line_count,point_count):
	points = np.asarray(line_count * np.random.rand(point_count,2) - 2)
	ceil = np.ceil(points) + 1
	label = (ceil[:,0] % 2 == ceil[:,1] % 2)
	#label = ceil[:,0] > ceil[:,1]
	return points, label


def plot(points,label,N):
	fig = plt.figure(figsize=(6,8))
	bounds = [-2.5,2.5]
	left_right = np.array(bounds)
	intervals = np.arange(bounds[0],bounds[1]+0.1,0.1)

	ctr_pts = np.meshgrid(intervals,intervals)
	labels = [''.join(i) for i in product('01',repeat=N)]
	lbl_colors = dict(zip(labels,sns.color_palette("husl", len(labels))))


#	ax = plt.axes(xlim=(left_right[0],left_right[1]), ylim=(-2.5, 2.5))
#	ax = plt.subplot(121)
	ax1 = plt.subplot2grid((3, 2), (0, 0),rowspan=2,colspan=2)
	ax2 = plt.subplot2grid((3, 2), (2, 0),colspan=2)


	pos = np.arange(len(labels))
	width = 5.0/len(labels)
	rects = ax2.bar(pos,np.zeros(len(labels)),width)
	ax2.set_xticks(pos + (width / 2))
	ax2.set_xticklabels(labels,rotation='vertical')
#	for i,c in zip(ax2.get_xticklabels(),lbl_colors): i.set_color(c)

	
	#im = ax1.imshow(np.ones(ctr_pts[0].shape), interpolation='bilinear', origin='lower',cmap=cm.gray,extent=(left_right[0],left_right[1])*2,animated=True)
	Z = np.ones(ctr_pts[0].shape)
	im = ax1.imshow(
			Z,
			interpolation='bilinear',
			origin='lower',
			animated=True,
			cmap=cm.gray,
			extent=(left_right[0],left_right[1])*2,
			alpha=0.5,
			vmin=0,vmax=1,
		)
	ax1.set_xlim(bounds)
	ax1.set_ylim(bounds)
	class1 = points[ label]
	class2 = points[~label]
	ax1.plot(class1[:,0],class1[:,1], 'ro')
	ax1.plot(class2[:,0],class2[:,1], 'bo')
	return fig,ax1,ax2,left_right,labels,lbl_colors,rects,ctr_pts,im,Z


if __name__ == "__main__":
	line_count = 4
	instances  = 1000
	points, label = create_normal_data(line_count/2 + 1,instances)
	train,predict,params = build_model(line_count)
	fig,ax,ax_hist,left_right,labels,lbl_colors,rects,ctr_pts,im,Z = plot(points,label,line_count)
	lines = [ax.plot([], [], lw=2)[0] for _ in xrange(line_count)]
	acc_text = ax.text(0.02, 1.01, '', transform=ax.transAxes)
	acc_text.set_text('')

	def data_gen():
		acc,coeffs,bias,hidden = train(points,label)
		ctr_z = predict(np.dstack(ctr_pts).reshape(-1, 2))
		while acc < 0.999:
			print acc
			yield acc,coeffs,bias,hidden,ctr_z
			acc,coeffs,bias,hidden = train(points,label)
			ctr_z = predict(np.dstack(ctr_pts).reshape(-1, 2))
		for _ in xrange(100):
			print acc
			yield acc,coeffs,bias,hidden,ctr_z
			acc,coeffs,bias,hidden = train(points,label)
			ctr_z = predict(np.dstack(ctr_pts).reshape(-1, 2))

	it = data_gen()
	def animate(data):
		acc,coeffs,bias,hidden,ctr_z = it.next()
		acc_text.set_text("accuracy=%0.2f"%acc)
		M = - coeffs[0]/coeffs[1]
		C = - bias / coeffs[1]
		for i in xrange(len(lines)):
			lines[i].set_data(left_right, M[i]*left_right + C[i])
		#Z[:,:] = 
		#im = ax.imshow(Z, interpolation='bilinear', origin='lower',cmap=cm.gray,extent=(left_right[0],left_right[1])*2)
		im.set_data(ctr_z.reshape(ctr_pts[0].shape))
		activations = Counter(''.join(str(e) for e in row) for row in hidden)
		
		labels.sort(key=lambda x:-activations.get(x,0))
		for i in ax_hist.get_xticklabels(): i.set_color(lbl_colors[i.get_text()])
		for lbl,rct in zip(labels,rects):
			rct.set_height(activations.get(lbl,0)/float(instances))


	anim = animation.FuncAnimation(fig,animate,frames=2500, interval=20, blit=True)
	anim.save('grid_learn_init_spec.mp4', fps=60,bitrate=512)

