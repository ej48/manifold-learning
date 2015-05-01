from __future__ import division
import sys
from sklearn.manifold import Isomap
from sklearn.decomposition import PCA
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import random
from colorsys import hsv_to_rgb


data = np.genfromtxt('data012.txt', delimiter=',')
isomap = Isomap()
data_xformed = isomap.fit_transform(data)
# pca = PCA(n_components=2)
# data_xformed = pca.fit_transform(data)
print data.shape
print data_xformed.shape
c = [(1,0,0)]*1000+[(0,1,0)]*1000+[(1,1,0)]*1000
plt.figure()
plt.scatter(data_xformed[:,0], data_xformed[:,1], c=c)
plt.show()
quit()

train_data = np.genfromtxt('training.txt', delimiter=',')
isomap = Isomap(n_components=4)
train_xformed = isomap.fit_transform(train_data)
test_data = np.genfromtxt('testing.txt', delimiter=',')
test_xformed = isomap.transform(test_data)
np.savetxt("isomap_training_reduced4.txt", train_xformed, delimiter=',')
np.savetxt("isomap_testing_reduced4.txt", test_xformed, delimiter=',')