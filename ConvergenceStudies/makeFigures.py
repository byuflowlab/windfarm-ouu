import numpy as np
import matplotlib.pyplot as plt

if __name__=="__main__":
    filename = "powerVSspeed30.0.txt"
    file = open(filename)
    data = np.loadtxt(file)
    x = np.zeros(len(data))
    y = np.zeros(len(data))
    variance = np.zeros(len(data))
    error = np.zeros(len(data))

    converged = 1212031792.82

    for i in range(len(data)):
        x[i] = data[i][0]
        y[i] = data[i][1]


    plt.plot(x, y)
    plt.ylim([0, 550000])
    plt.xlabel('Number of Directions')
    plt.ylabel('AEP')
    plt.title('AEP vs Number of Wind directions: 10x10 grid')
    plt.show()

