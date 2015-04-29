
from __future__ import division
import sys
from sklearn.decomposition import PCA
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import random
from colorsys import hsv_to_rgb

# fo = open("testing.txt", "w")
# for i in xrange(10):
# 	f = open("data"+str(i)+".txt")
# 	train = f.readlines()[800:1000]
# 	fo.write("".join(train))
# fo.close()
# quit()

def acc(l):
	l = np.array(l)
	result = np.zeros(l.shape)
	result[0] = l[0]
	for i in xrange(1, len(result)):
		result[i] = result[i-1] + l[i]
	result = [0] + list(result)
	return result

def explore_data():
	data = np.genfromtxt(sys.argv[1], delimiter=',')
	# data = preprocessing.scale(data)
	first_example = data[0,:].reshape(28, 28)
	pca = PCA()
	pca.fit(data)
	plt.figure()
	plt.imshow(first_example, cmap=cm.Greys_r)
	for i in [0, 1, 5, 10, 30]:
		comp = pca.components_[i,:].reshape(28, 28)
		plt.figure()
		plt.imshow(comp, cmap=cm.Greys_r)
		plt.title(str(i+1)+"th component")
	plt.figure()
	expl_var_ratios = pca.explained_variance_ratio_
	acc_ratio = acc(expl_var_ratios)
	plt.plot(range(0, 28*28+1), acc_ratio, 'k')
	plt.title('Ratio of explained variance by first m principal components')
	plt.xlabel('m')
	plt.ylabel('ratio')
	plt.grid(True)
	plt.ylim([-0.1, 1.1])
	plt.show()

def two_digits():
	data = np.genfromtxt('data012.txt', delimiter=',')
	# data = preprocessing.scale(data)
	first_example = data[0,:].reshape(28, 28)
	pca = PCA(n_components=3)
	transformed = pca.fit_transform(data)
	expl_var_ratios = pca.explained_variance_ratio_
	print acc(expl_var_ratios)
	d1 = transformed[0:1000,:]
	d2 = transformed[1000:2000,:]
	d3 = transformed[2000:3000,:]
	xs1 = d1[:,0]
	ys1 = d1[:,1]
	zs1 = d1[:,2]
	xs2 = d2[:,0]
	ys2 = d2[:,1]
	zs2 = d2[:,2]
	xs3 = d3[:,0]
	ys3 = d3[:,1]
	zs3 = d3[:,2]
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.hold(True)
	ax.scatter(xs1, ys1, zs1, c=(1,0,0))
	ax.scatter(xs2, ys2, zs2, c=(0,1,0))
	ax.scatter(xs3, ys3, zs3, c=(1,1,0))
	plt.show()

def process_all():
	#color_iter = iter([(1,0,0), (1,1,0), (0,1,0)])
	hs = np.linspace(0, 0.75, 10)
	hsvs = [(h, 1, 1) for h in hs]
	rgbs = [hsv_to_rgb(h, s, v) for h,s,v in hsvs]
	color_iter = iter(rgbs)
	data = np.genfromtxt('training.txt', delimiter=',')
	pca = PCA(n_components=30)
	transformed = pca.fit_transform(data)
	expl_var_ratios = pca.explained_variance_ratio_
	plt.figure()
	plt.plot(acc(expl_var_ratios),'k')
	print acc(expl_var_ratios)
	plt.xlabel('number of principal component')
	plt.ylabel('percentage of total variance explained')
	plt.title('Percentage of total variance explained by first k principal components')
	plt.show()
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.hold(True)
	for i in xrange(10):
		d = transformed[i*1000:(i+1)*1000, :]
		xs = d[:,0]
		ys = d[:,1]
		zs = d[:,2]
		# plt.scatter(xs, ys, c=(random.random(), random.random(), random.random()))
		ax.scatter(xs, ys, zs, c=next(color_iter))
	ax.set_title("Principal component decomposition of digit data")
	ax.legend(map(str, range(10)), loc="upper left")
	ax.set_xlabel('1st principal component')
	ax.set_ylabel('2nd principal component')
	ax.set_zlabel('3rd principal component')
	plt.show()

if __name__ == '__main__':
	process_all()
	quit()
	transformed = dict()
	# digits = range(10)
	digits = [4, 5]
	colors = iter([(1,0,0), (0,1,0)])
	for digit in digits:
		f = 'data'+str(digit)+'.txt'
		data = np.genfromtxt(f, delimiter=',')
		data = data[0:1000, :]
		pca = PCA(n_components=3)
		transformed[digit] = pca.fit_transform(data)
		print ".", 
	print ""
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.hold(True)
	for digit in digits:
		xs = transformed[digit][:,0]
		ys = transformed[digit][:,1]
		zs = transformed[digit][:,2]
		# ax.scatter(xs, ys, zs, c=(random.random(),random.random(),random.random()))
		ax.scatter(xs, ys, zs, c=(next(colors)))
	ax.legend(map(str, digits))
	ax.set_xlabel('1st principal component')
	ax.set_ylabel('2nd principal component')
	ax.set_zlabel('3rd principal component')
	ax.set_title('First 3 principal component projections of digit data')
	plt.show()