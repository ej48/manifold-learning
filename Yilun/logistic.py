
from __future__ import division
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
import numpy as np
import matplotlib.pyplot as plt

train = np.genfromtxt("lle_data_2_dims.txt", delimiter=',')
train_labels = []
for i in xrange(10):
	for _ in xrange(100):
		train_labels.append(i)

logistic = LogisticRegression()
nb = GaussianNB()
svm = LinearSVC()
svm.fit(train, train_labels)
labels = list(svm.predict(train))
correct = [1 if train_labels[i]==labels[i] else 0 for i in xrange(len(train_labels))]
print sum(correct)/len(train_labels)
