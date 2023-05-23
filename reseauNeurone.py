# class ReseauNeurone : class de base pour la création d'un réseau de neurones

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

class ReseauNeurone:

    parametres = {}
    dimensions = []

    def __init__(self, dimensions):
        self.dimensions = dimensions
        C = len(dimensions)
        for l in range(1, C):
            self.parametres['W' + str(l)] = np.random.randn(dimensions[l], dimensions[l - 1])
            self.parametres['b' + str(l)] = np.zeros((dimensions[l], 1))

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))
    
    def forward_propagation(self, X):
        C = len(self.dimensions)
        activations = { "A0": X }

        for l in range(1, C):
            Z = self.parametres["W" + str(l)].dot(activations["A" + str(l - 1)]) + self.parametres["b" + str(l)]
            activations["A" + str(l)] = self.sigmoid(Z)

        return activations
    
    def predict(self, X):
        activations = self.forward_propagation(X)
        Af = activations["A" + str(len(self.dimensions) - 1)]
        return Af >= 0.5
    
    def back_propagation( self, y, activations ):
        C = len(self.dimensions)
        m = y.shape[1]

        dZ = activations["A" + str(C - 1)] - y
        gradients = {}

        for l in reversed(range(1, C)):
            gradients["dW" + str(l)] = 1 / m * np.dot(dZ, activations["A" + str(l - 1)].T)
            gradients["db" + str(l)] = 1 / m * np.sum(dZ, axis=1, keepdims=True)    
            if l > 1:
                dZ = np.dot(self.parametres["W" + str(l)].T, dZ) * (activations["A" + str(l - 1)] * (1 - activations["A" + str(l - 1)]))

        return gradients

    def update_parameters(self, gradients, learning_rate):
        C = len(self.dimensions)
        for l in range(1, C):
            self.parametres["W" + str(l)] -= learning_rate * gradients["dW" + str(l)]
            self.parametres["b" + str(l)] -= learning_rate * gradients["db" + str(l)]

    def compute_cost(self, A, y):
        m = y.shape[1]
        return -1 / m * np.sum(y * np.log(A) + (1 - y) * np.log(1 - A))
    
    def compute_accuracy_score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y.flatten(), y_pred.flatten())

    def train(self, X, y, learning_rate, epochs):

        train_loss = []
        train_acc = []

        for i in range(epochs):
            activations = self.forward_propagation(X)
            gradients = self.back_propagation(y, activations)
            self.update_parameters(gradients, learning_rate)

            if i % 100 == 0:
                train_loss.append(self.compute_cost(activations["A" + str(len(self.dimensions) - 1)], y))
                train_acc.append(self.compute_accuracy_score(X, y))

        # visualisation de la courbe d'apprentissage
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(18, 4))
        ax[0].plot(train_loss, label="Train loss")
        ax[0].set_xlabel("Epochs")
        ax[0].legend()
        ax[1].plot(train_acc, label="Train acc")
        ax[1].set_xlabel("Epochs")
        ax[1].legend()
        # Affichage de la frontière de décision
        ax[2].set_title("Frontière de décision")
        ax[2].set_xlabel("x1")
        ax[2].set_ylabel("x2")
        ax[2].set_xlim([-2, 2])
        ax[2].set_ylim([-2, 2])
        x1 = np.linspace(-2, 2, 100)
        x2 = np.linspace(-2, 2, 100)
        xx1, xx2 = np.meshgrid(x1, x2)
        grid = np.c_[xx1.ravel(), xx2.ravel()].T
        Z = self.predict(grid)
        Z = Z.reshape(xx1.shape)
        ax[2].contourf(xx1, xx2, Z, cmap=plt.cm.Spectral)
        ax[2].scatter(X[0, :], X[1, :], c=y.flatten(), cmap=plt.cm.Spectral)


        plt.show()
