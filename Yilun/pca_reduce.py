
'''
train PCA basis based on training.txt and output dimension-reduced coefficients for both training.txt and testing.txt
'''

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

final_dim = 30
data = np.genfromtxt("100examples.txt", delimiter=',')
pca = PCA(n_components=final_dim)
isomap = Isomap(n_components=final_dim)
lle = LocallyLinearEmbedding(n_components=final_dim)
data_xformed = lle.fit_transform(data)
np.savetxt("lle_data_30_dims.txt", data_xformed, delimiter=',')
