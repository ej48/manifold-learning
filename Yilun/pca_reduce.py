
'''
train PCA basis based on training.txt and output dimension-reduced coefficients for both training.txt and testing.txt
'''

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

final_dim = 4
train = np.genfromtxt("training.txt", delimiter=',')
test = np.genfromtxt("testing.txt", delimiter=',')
pca = PCA(n_components=final_dim)
reduced_train = pca.fit_transform(train)
reduced_test = pca.transform(test)
np.savetxt("training_reduced4.txt", reduced_train, delimiter=',')
np.savetxt("testing_reduced4.txt", reduced_test, delimiter=',')
