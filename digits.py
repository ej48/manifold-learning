print(__doc__)
from time import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble, lda,
                     random_projection, linear_model, metrics)

digits = datasets.load_digits(n_class=6)
n_samples, n_features = digits.data.shape
n_neighbors = 30
dimensions = 2
training_size = int(0.8*n_samples)

# training set
X = digits.data[:training_size] 
y = digits.target[:training_size]

# test set
X_test = digits.data[training_size:]
y_test = digits.target[training_size:]


#----------------------------------------------------------------------
# Scale and visualize the embedding vectors
def plot_embedding(X, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(digits.target[i]),
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
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r),
                X[i])
            ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)


#----------------------------------------------------------------------
# Plot images of the digits
n_img_per_row = 20
img = np.zeros((10 * n_img_per_row, 10 * n_img_per_row))
for i in range(n_img_per_row):
    ix = 10 * i + 1
    for j in range(n_img_per_row):
        iy = 10 * j + 1
        img[ix:ix + 8, iy:iy + 8] = X[i * n_img_per_row + j].reshape((8, 8))

plt.imshow(img, cmap=plt.cm.binary)
plt.xticks([])
plt.yticks([])
plt.title('A selection from the 64-dimensional digits dataset')


#----------------------------------------------------------------------
# Projection on to the first dimensions principal components

print("Computing PCA projection")
t0 = time()
clf = decomposition.TruncatedSVD(n_components=dimensions)
X_pca = clf.fit_transform(X)
t1 = time()
plot_embedding(X_pca,
               "Principal Components projection of the digits (time %.2fs)" %
               (t1 - t0))
               
# Classification               
log_clf = linear_model.LogisticRegression()
log_clf.fit(X_pca, y)
X_pca_test = clf.transform(X_test)
y_pca_pred = log_clf.predict(X_pca_test)

#----------------------------------------------------------------------
# Isomap projection of the digits dataset
print("Computing Isomap embedding")
t0 = time()
clf = manifold.Isomap(n_neighbors, n_components=dimensions)
X_iso = clf.fit_transform(X)
t1 = time()
print("Done.")
plot_embedding(X_iso,
               "Isomap projection of the digits (time %.2fs)" %
               (t1 - t0))

# Classification               
log_clf = linear_model.LogisticRegression()
log_clf.fit(X_iso, y)
X_iso_test = clf.transform(X_test)
y_iso_pred = log_clf.predict(X_iso_test)

#----------------------------------------------------------------------
# Locally linear embedding of the digits dataset
print("Computing LLE embedding")
t0 = time()
clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=dimensions,
                                      method='standard')
X_lle = clf.fit_transform(X)
t1 = time()
print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
plot_embedding(X_lle,
               "Locally Linear Embedding of the digits (time %.2fs)" %
               (t1 - t0))
               
# Classification               
log_clf = linear_model.LogisticRegression()
log_clf.fit(X_lle, y)
X_lle_test = clf.transform(X_test)
y_lle_pred = log_clf.predict(X_lle_test)

#----------------------------------------------------------------------
# MDS  embedding of the digits dataset
print("Computing MDS embedding")
t0 = time()
clf = manifold.MDS(n_components=dimensions, n_init=1, max_iter=100)
X_mds = clf.fit_transform(X)
t1 = time()
print("Done. Stress: %f" % clf.stress_)
plot_embedding(X_mds,
               "MDS embedding of the digits (time %.2fs)" %
               (t1 - t0))

# Classification               
log_clf = linear_model.LogisticRegression()
log_clf.fit(X_mds, y)
#X_mds_test = clf.transform(X_test)
#y_mds_pred = log_clf.predict(X_mds_test)
y_mds_pred = y_test

#----------------------------------------------------------------------
# Display results

# baseline, fit on high dimension
log_clf = linear_model.LogisticRegression()
log_clf.fit(X, y)
y_pred = log_clf.predict(X_test)
print "Accuracy score for baseline: {0}".format(metrics.accuracy_score(y_pred, y_test))

print "Accuracy score for PCA: {0}".format(metrics.accuracy_score(y_pca_pred, y_test))
print "Accuracy score for Isomap: {0}".format(metrics.accuracy_score(y_iso_pred, y_test))
print "Accuracy score for LLE: {0}".format(metrics.accuracy_score(y_lle_pred, y_test))
print "Accuracy score for MDS: {0}".format(metrics.accuracy_score(y_mds_pred, y_test))

plt.show()