from jensen_topHat import jensen_topHat, powerCalc
import numpy as np
import math

# Define turbine locations
turbineX = np.array([1164.7, 947.2,  1682.4, 1464.9, 1982.6, 2200.1])
turbineY = np.array([1024.7, 1335.3, 1387.2, 1697.8, 2060.3, 1749.7])

# Define turbine characteristics
Cp = 0.7737 * 4.0 * 1.0/3.0 * math.pow((1 - 1.0/3.0),2)
Cp = np.array([Cp, Cp, Cp, Cp, Cp, Cp])
axialInd = np.array([1.0/3.0, 1.0/3.0, 1.0/3.0, 1.0/3.0, 1.0/3.0, 1.0/3.0])
rotorDiameter = np.array([126.4, 126.4, 126.4, 126.4, 126.4, 126.4])

# Define turbine measurements
Vinf = 8.0
k = 0.1

# Define site measurements
windDirection = 240
airDensity = 1.1716

Ueff = jensen_topHat(Vinf, rotorDiameter, axialInd, turbineX, turbineY, k, windDirection)

Power = powerCalc(Ueff, Cp, rotorDiameter, airDensity)

print "effective windspeeds (m/s): ", Ueff
print "Power of each turbine: ", Power
print "Wind farm total power: ", np.sum(Power)