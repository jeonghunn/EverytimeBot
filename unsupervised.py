from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn import datasets
#Iris Dataset
#iris = datasets.load_iris()
x_ar = np.array([[[9, 9, 9],
                  [0, 0, 9],
                  [0, 0, 9]],

                 [[9, 0, 0],
                  [9, 0, 0],
                  [9, 9, 9]],

                 [[9, 0, 0],
                  [9, 0, 0],
                  [9, 9, 9]],

                 [[9, 9, 9],
                  [0, 3, 0],
                  [0,0, 9]]])

test_ar = np.array([[[9, 0,0],
                  [9, 0, 1],
                  [8, 7, 3]]
])
PX = x_ar

#reshape
nsamples, nx, ny = PX.shape
tsample, nx, ny = test_ar.shape
X = PX.reshape((nsamples,nx*ny))
test_ar = test_ar.reshape((tsample,nx*ny))
#KMeans
km = KMeans(n_clusters=3)
km.fit(X)
km.predict(X)
labels = km.labels_
#Plotting
fig = plt.figure(1, figsize=(7,7))
ax = Axes3D(fig, rect=[0, 0, 0.95, 1], elev=48, azim=134)
ax.scatter(X[:, 3], X[:, 0], X[:, 2],
           c=labels.astype(np.float), edgecolor="k", s=50)
ax.set_xlabel("Petal width")
ax.set_ylabel("Sepal length")
ax.set_zlabel("Petal length")
plt.title("K Means", fontsize=14)
plt.show()

predict_result =km.predict(test_ar)

print(predict_result)
a = 3