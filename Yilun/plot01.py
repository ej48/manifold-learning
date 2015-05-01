from __future__ import division
import sys
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import random
from colorsys import hsv_to_rgb

pca = PCA(n_components=2)
isomap = Isomap(n_components=2)
lle = LocallyLinearEmbedding(n_components=2)
data = np.genfromtxt('data01_small.txt', delimiter=',')
pca_xform = pca.fit_transform(data)
isomap_xform = isomap.fit_transform(data)
lle_xform = lle.fit_transform(data)
label = [0]*100+[1]*100
rgbs = [(0.5,0,0), (0,0.5,0)]


plt.figure()
xs = pca_xform[:,0]
ys = pca_xform[:,1]
ax = plt.subplot(111)
for i in xrange(len(xs)):
	ax.text(xs[i], ys[i], str(label[i]), color=rgbs[label[i]], fontdict={'weight': 'bold', 'size': 9})
t = (max(xs)-min(xs))*0.1
ax.axis([min(xs)-t, max(xs)+t, min(ys)-t, max(ys)+t])
plt.xticks([]), plt.yticks([])
plt.title('PCA')

plt.figure()
xs = lle_xform[:,0]
ys = lle_xform[:,1]
ax = plt.subplot(111)
t = (max(xs)-min(xs))*0.1
for i in xrange(len(xs)):
	ax.text(xs[i], ys[i], str(label[i]), color=rgbs[label[i]], fontdict={'weight': 'bold', 'size': 9})
ax.axis([min(xs)-t, max(xs)+t, min(ys)-t, max(ys)+t])
plt.xticks([]), plt.yticks([])
plt.title('LLE')

plt.figure()
xs = isomap_xform[:,0]
ys = isomap_xform[:,1]
t = (max(xs)-min(xs))*0.1
ax = plt.subplot(111)
for i in xrange(len(xs)):
	ax.text(xs[i], ys[i], str(label[i]), color=rgbs[label[i]], fontdict={'weight': 'bold', 'size': 9})
ax.axis([min(xs)-t, max(xs)+t, min(ys)-t, max(ys)+t])
plt.xticks([]), plt.yticks([])
plt.title('Isomap')

plt.show()