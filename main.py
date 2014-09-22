import model
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as path
import matplotlib.animation as animation
from ark import load_ark

def create_animation(plot,filename):
	fig = plt.figure(figsize=(14,10))
	plot_verts = []
	bottom = 0 
	for i,sequence in enumerate(plot):
		ax = plt.subplot(len(plot),1,i+1)
		#fig, ax = plt.subplots()
		if i==0:
			time_text = ax.text(0.02, 1.1, '', transform=ax.transAxes)
			time_text.set_text('')
		# histogram our data with numpy
		n = sequence[0]
		bins = range(len(sequence[0])+1)

		# get the corners of the rectangles for the histogram
		left = np.array(bins[:-1])
		right = np.array(bins[1:])
		top = n
		nrects = len(left)
		layer_text = ax.text(0.90, 0.80, 'layer %d'%(i+1), transform=ax.transAxes)

		# here comes the tricky part -- we have to set up the vertex and path
		# codes arrays using moveto, lineto and closepoly

		# for each rect: 1 for the MOVETO, 3 for the LINETO, 1 for the
		# CLOSEPOLY; the vert for the closepoly is ignored but we still need
		# it to keep the codes aligned with the vertices
		nverts = nrects*(1+3+1)
		verts = np.zeros((nverts, 2))
		codes = np.ones(nverts, int) * path.Path.LINETO
		codes[0::5] = path.Path.MOVETO
		codes[4::5] = path.Path.CLOSEPOLY
		verts[0::5,0] = left
		verts[0::5,1] = bottom
		verts[1::5,0] = left
		verts[1::5,1] = top
		verts[2::5,0] = right
		verts[2::5,1] = top
		verts[3::5,0] = right
		verts[3::5,1] = bottom
		
		plot_verts.append(verts)
		barpath = path.Path(verts, codes)
		patch = patches.PathPatch(barpath, facecolor='green', edgecolor='yellow', alpha=0.5)
		ax.add_patch(patch)

		ax.set_xlim(left[0], right[-1])
		ax.set_ylim(0,1.1)
	def animate(t):
		time_text.set_text("t=%d"%t)
		for sequence,verts in zip(plot,plot_verts):
			n = sequence[t]
			top = bottom + n
			verts[1::5,1] = top
			verts[2::5,1] = top

	ani = animation.FuncAnimation(fig, animate, len(sequence), repeat=False)
	ani.save(filename, fps=15,bitrate=1280)



if __name__ == "__main__":
	predict = model.load(sys.argv[1])
	layers  = predict(load_ark(sys.argv[2]))

	means = [ np.sum(l > 0.5,axis=0)   for l in layers ]
	order = [ np.argsort(-m)[:100] for m in means ]
	plot = [ l[:,o] for l,o in zip(layers,order) ]

	create_animation(plot,sys.argv[3])
