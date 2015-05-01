print(__doc__)
from time import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble, lda,
                     random_projection, linear_model, metrics)
# dim reduction parameters
n_neighbors = 30
dimensions = 2              ### change this number

# load data
X = np.genfromtxt('100examples.txt', delimiter=',')
y = []
for i in xrange(10):
    for _ in xrange(100):
        y.append(i)

#----------------------------------------------------------------------
# Scale and visualize the embedding vectors
def plot_embedding(X, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.Set1(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(X.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-3:
                # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X[i]]]
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)

#----------------------------------------------------------------------
# Projection on to the first d principal components
print("Computing PCA projection")
clf = decomposition.TruncatedSVD(n_components=dimensions)
X_pca = clf.fit_transform(X)
plot_embedding(X_pca,
               "Principal Components projection of the digits")
               
# Classification               
log_clf = linear_model.LogisticRegression()
log_clf.fit(X_pca, y)
y_pca_pred = log_clf.predict(X_pca)

#----------------------------------------------------------------------
# Isomap projection of the digits dataset
print("Computing Isomap embedding")
clf = manifold.Isomap(n_neighbors, n_components=dimensions)
X_iso = clf.fit_transform(X)
plot_embedding(X_iso,
               "Isomap projection of the digits")

# Classification               
log_clf = linear_model.LogisticRegression()
log_clf.fit(X_iso, y)
y_iso_pred = log_clf.predict(X_iso)

#----------------------------------------------------------------------
# Locally linear embedding of the digits dataset
print("Computing LLE embedding")
clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=dimensions,
                                      method='standard')
X_lle = clf.fit_transform(X)
plot_embedding(X_lle,
               "Locally Linear Embedding of the digits")
               
# Classification               
log_clf = linear_model.LogisticRegression()
log_clf.fit(X_lle, y)
y_lle_pred = log_clf.predict(X_lle)

#----------------------------------------------------------------------
# Display results

# baseline, fit on high dimension (unreduced data)
log_clf = linear_model.LogisticRegression()
log_clf.fit(X, y)
y_pred = log_clf.predict(X)
print "Accuracy score for baseline: {0}".format(metrics.accuracy_score(y_pred, y))

print "Accuracy score for PCA: {0}".format(metrics.accuracy_score(y_pca_pred, y))
print "Accuracy score for Isomap: {0}".format(metrics.accuracy_score(y_iso_pred, y))
print "Accuracy score for LLE: {0}".format(metrics.accuracy_score(y_lle_pred, y))

plt.show()