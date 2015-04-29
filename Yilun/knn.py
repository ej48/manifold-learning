
from __future__ import division
from sklearn.neighbors import KNeighborsClassifier
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

corrects = []
for n in xrange(1,301):
	knn = KNeighborsClassifier(n_neighbors=n)
	knn.fit(train, train_labels)
	predicted_labels = list(knn.predict(test))
	correct = [1 if test_labels[i]==predicted_labels[i] else 0 for i in xrange(len(test_labels))]
	corrects.append(sum(correct)/2000)
	if n%50==0:
		print ".",
print ""
print max((val,idx) for idx,val in enumerate(corrects))
plt.figure()
plt.plot(range(1, 301), corrects, 'k')
plt.xlabel('number of nearest neighbors')
plt.ylabel('correct rate')
plt.title('Percentage of correct prediction vs. number of nearest neighbors used')
plt.show()