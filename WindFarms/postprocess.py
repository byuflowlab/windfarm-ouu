import numpy as np
import matplotlib.pyplot as plt

locations = np.genfromtxt('layout_amalia.txt', delimiter=' ')
tXa = locations[:,0]
tYa = locations[:,1]

locations = np.genfromtxt('layout_grid.txt', delimiter=' ')
tXg = locations[:,0]
tYg = locations[:,1]

locations = np.genfromtxt('layout_optimized.txt', delimiter=' ')
tXo = locations[:,0]
tYo = locations[:,1]

locations = np.genfromtxt('layout_random.txt', delimiter=' ')
tXr = locations[:,0]
tYr = locations[:,1]

colors = ['b', 'g', 'r', 'c', 'y']
fig, ax = plt.subplots()
ax.scatter(tXg, tYg, label='grid', color=colors[0])
#ax.scatter(tXa, tYa, label='amalia', color=colors[1])
#ax.scatter(tXo, tYo, label='optimized', color=colors[2])
#ax.scatter(tXr, tYr, label='random', color=colors[3])

plt.show()




