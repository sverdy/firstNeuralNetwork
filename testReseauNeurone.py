# test le reseau de neurone
from reseauNeurone import ReseauNeurone


# test de l'implementation d'un exercice de segmentation à deux dimensions
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_circles

X, y = make_circles(n_samples=100, noise=0.1, factor=0.3, random_state=0)
# X = X.T
y = y.reshape((1, y.shape[0])).T


print("Dimensions de X : ", X.shape)
print("Dimensions de y : ", y.shape)

plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()

model = ReseauNeurone([2, 10, 10, 1])
model.train(X.T, y.T, 0.1, 30000)


