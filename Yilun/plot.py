
from __future__ import division

import matplotlib.pyplot as plt
import numpy as np
import sys
from colorsys import hsv_to_rgb

data = np.genfromtxt(sys.argv[1], delimiter=',')
hs = np.linspace(0, 0.75, 10)
hsvs = [(h, 1, 1) for h in hs]
rgbs = [hsv_to_rgb(h, s, v) for h,s,v in hsvs]
xs = data[:,0]
ys = data[:,1]
label = sum([[i]*100 for i in xrange(10)], [])
plt.figure()
ax = plt.subplot(111)
for i in xrange(len(xs)):
	# print xs[i], ys[i], label[i]
	ax.text(xs[i], ys[i], str(label[i]), color=rgbs[label[i]], fontdict={'weight': 'bold', 'size': 9})
ax.axis([min(xs), max(xs), min(ys), max(ys)])
plt.xticks([]), plt.yticks([])
plt.title(sys.argv[2])
plt.show()