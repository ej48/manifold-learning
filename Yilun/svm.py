
from __future__ import division
from sklearn.svm import LinearSVC
import numpy as np
import matplotlib.pyplot as plt

train = np.genfromtxt("training_reduced4.txt", delimiter=',')
test = np.genfromtxt("testing_reduced4.txt", delimiter=',')
train_labels = []
for i in xrange(10):
	for _ in xrange(800):
		train_labels.append(i)
test_labels = []
for i in xrange(10):
	for _ in xrange(200):
		test_labels.append(i)

svm = LinearSVC()
svm.fit(train, train_labels)
predicted_labels = list(svm.predict(test))
correct = [1 if test_labels[i]==predicted_labels[i] else 0 for i in xrange(len(test_labels))]
print sum(correct)/len(test_labels)
