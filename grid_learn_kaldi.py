KALDI_HOME="/home/shawn/kaldi-trunk"
NNETBIN_HOME=KALDI_HOME+"/src/nnetbin"
NNET_UTILS=KALDI_HOME+"/egs/wsj/s5/utils/nnet"
import numpy as np
from grid_learn import *
from subprocess import call
import model
def create_ark_data(features_filename,probability_filename,grid_size,point_count):
	data,labels = create_data(grid_size,point_count)
	with open(features_filename,'w') as f:
		f.write('grid_learning_task  [')
		for row in data:
			f.write('\n')
			f.write('  ')
			f.write(' '.join("%0.6f"%e for e in row))
			f.write(' ')
		f.write(' ]')

	with open(probability_filename,'w') as f:
		f.write('grid_learning_task ')
		for row in labels:
			f.write(" [ %d 1 ] "%row)
		f.write('\n')
	return data,labels

def init_network(hidden_layer):
	with open('nnet.conf','w') as f:
		call([
			'python2',
			NNET_UTILS+"/make_nnet_proto.py",
			"2","2","1",str(hidden_layer)
		],stdout=f)

	call([
		"%s/nnet-initialize"%NNETBIN_HOME,
		"--binary=false",
		"nnet.conf",
		"nnet.out"
	])

def data_gen(data,pred,points,label,line_count):
	program_path = "%s/nnet-train-frmshuff"%NNETBIN_HOME
	train,params = build_model(line_count,False)
	while True:
		call([
			program_path,
			'--binary=false',
			'ark:%s'%data,
			'ark:%s'%pred,
			'nnet.out',
			'nnet.out.1'
		])
		parameters = model.load_train_file('nnet.out.1')
		for n,o in zip(parameters,params): o.set_value(n)
		acc,coeffs,bias,hidden = train(points,label)
		yield acc,coeffs, bias
		call(['mv','nnet.out.1','nnet.out'])

if __name__ == "__main__":
	line_count = 4
	data_filename,pred_filename = "data.ark","pred.ark"
	points, label = create_ark_data(data_filename,pred_filename,line_count/2 + 1,1000)
	init_network(line_count)
	fig,ax,left_right = plot(points,label)
	lines = [ax.plot([], [], lw=2)[0] for _ in xrange(line_count)]
	acc_text = ax.text(0.02, 1.01, '', transform=ax.transAxes)
	acc_text.set_text('')


	it = data_gen(data_filename,pred_filename,points,label,line_count)
	def animate(data):
		acc,coeffs,bias = it.next()
		print acc
		acc_text.set_text("accuracy=%0.2f"%acc)
		M = - coeffs[0]/coeffs[1]
		C = - bias / coeffs[1]
		for i in xrange(len(lines)):
			lines[i].set_data(left_right, M[i]*left_right + C[i])

	anim = animation.FuncAnimation(fig, animate, frames=1000, interval=20, blit=True)
	anim.save('grid_learn_kaldi_%d.mp4'%line_count, fps=60,bitrate=512)
	call(['rm', 'nnet.conf', 'nnet.out.1','nnet.out'])







#create_ark_data('data.ark','pred.ark',4,1000)
