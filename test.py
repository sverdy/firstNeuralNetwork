## test si l'environnement est bien configuré
import numpy as np
import matplotlib.pyplot as plt

# Création d'un tableau de 1000 valeurs aléatoires
data = np.random.randn(1000)

# Affichage de l'histogramme
plt.hist(data)
plt.show()
