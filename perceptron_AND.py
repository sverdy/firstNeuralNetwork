# test de l'implementation d'un exercice de segmentation à deux dimensions
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score

X, y = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([0, 0, 0, 1])
y = y.reshape((y.shape[0], 1))

print("Dimensions de X : ", X.shape)
print("Dimensions de y : ", y.shape)

plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()

# Fonction d'activation
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Fonction d'initialistion des hyperparamètres
def init_params(X):
    # Initialisation des paramètres
    w = np.random.randn(X.shape[1], 1)
    b = np.random.randn(1)
    return (w, b)

# Fonction de propagation avant
def forward_prop(X, w, b):
    # Calcul de la prédiction
    Z = np.dot(X, w) + b
    A = sigmoid(Z)
    return A

# Fonction de prédiction
def predict(X, w, b):
    # Calcul de la prédiction
    A = forward_prop(X, w, b)
    # Transformation des valeurs de A en 0 ou 1
    return np.round(A)

# Fonction de calcul du coût
# autrement dit la fonction de perte (log loss ou cross entropy  )
def compute_cost(A, y):
    # Calcul du coût
    m = y.shape[0]
    cost = -1 / m * np.sum(y * np.log(A) + (1 - y) * np.log(1 - A))
    return cost

# Fonction de calcul du gradient
def backward_prop(X, y, A):
    # Calcul du gradient
    m = y.shape[0]
    dw = 1 / m * np.dot(X.T, (A - y)) # jacobien
    db = 1 / m * np.sum(A - y)
    return (dw, db)

# Fonction de mise à jour des paramètres
def update_params(w, b, dw, db, learning_rate):
    # Mise à jour des paramètres
    w = w - learning_rate * dw
    b = b - learning_rate * db
    return (w, b)

# Fonction d'apprentissage
def learn(X, y, learning_rate, epochs):
    # Initialisation des paramètres
    w, b = init_params(X)
    costs = []

    # Boucle d'apprentissage
    for i in range(epochs):
        # Propagation avant
        A = forward_prop(X, w, b)
        # Calcul du coût
        costs.append(compute_cost(A, y))
        # Calcul du gradient
        dw, db = backward_prop(X, y, A)
        # Mise à jour des paramètres
        w, b = update_params(w, b, dw, db, learning_rate)
        # Affichage du coût
        if i % 100 == 0:
            print("Coût après itération %i: %f" % (i, costs[i]))

    y_pred = predict(X, w, b)
    print("Accuracy: ", accuracy_score(y, y_pred))

    return (w, b, costs)

W, B, costs = learn(X, y, 0.1, 1000)


# affichage de l'evolution du cout
plt.plot(costs)
plt.show()

# Affichage de la frontière de décision
x1 = np.linspace(-1, 4, 100)
x2 = -(B + W[0] * x1) / W[1]
plt.plot(x1, x2, c='red')
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()

