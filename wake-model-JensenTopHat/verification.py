from jensen_topHat import jensen_topHat, powerCalc
import numpy as np
import math
import pylab as plt

# Define turbine characteristics
axialInd = np.array([1.0/3.0, 1.0/3.0])
rotorDiameter = np.array([126.4, 126.4])

# Define turbine measurements
Vinf = 8.0
k = 0.1
Cp = 0.7737 * 4.0 * 1.0/3.0 * math.pow((1 - 1.0/3.0), 2)
Cp = np.array([Cp, Cp])

# Define site measurements
windDirection = 270
airDensity = 1.1716

# Define turbine locations
dist = 3.0
turbineX = np.array([0.0, dist*rotorDiameter[0]])
turbineY = np.array([0.0, 0.0])

res = 200
Ueff = np.zeros(res)
y = np.linspace(-1.5*rotorDiameter[0], 1.5*rotorDiameter[0], res)
theta = np.zeros_like(y)


for i in np.arange(0, res):
    turbineY[1] = y[i]
    _, temp = jensen_topHat(Vinf, rotorDiameter, axialInd, turbineX, turbineY, k, windDirection, Cp, airDensity)
    print temp
    Ueff[i] = temp[1]
    theta[i] = y[i]/dist
    print y[i], theta[i]

plt.figure()
plt.plot(theta, Ueff/Vinf)
plt.show()

# Display retunrs