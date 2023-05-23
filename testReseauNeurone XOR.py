# test le reseau de neurone
from reseauNeurone import ReseauNeurone


# test de l'implementation d'un exercice de segmentation à deux dimensions
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_circles

X, y = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([0, 1, 1, 0])
y = y.reshape((y.shape[0], 1))


print("Dimensions de X : ", X.shape)
print("Dimensions de y : ", y.shape)

plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()

model = ReseauNeurone([2, 2, 1])
model.train(X.T, y.T, learning_rate=0.1, epochs=30000)

