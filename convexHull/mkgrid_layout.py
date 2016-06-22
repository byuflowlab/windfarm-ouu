import numpy as np


locations = np.genfromtxt('../WindFarms/layout_amalia.txt', delimiter=' ')
turbineX = locations[:,0]
turbineY = locations[:,1]

# Find the bounds of the amalia wind farm to 2 significant digits
# round_sig = lambda x, sig=2: np.round(x, sig-int(np.floor(np.log10(x)))-1)
# xlim = round_sig(np.max(turbineX))
# ylim = round_sig(np.max(turbineY))
xlim = np.max(turbineX)
ylim = np.max(turbineY)

# Find the bounds of the equivalent area grid to 3 significant digits
round_sig = lambda x, sig=3: np.round(x, sig-int(np.floor(np.log10(x)))-1)
rf = 0.884  # reduction factor to make sure the grid layout area is equivalent to the amalia.
xlim = round_sig(xlim*rf)
ylim = round_sig(ylim*rf)

# Grid farm (same number of turbines as Amalia 60)
nRows = 10   # number of rows and columns in grid
nCols = 6
# spacing = 5  # turbine grid spacing in diameters, original spacing for the grid
spacingX = xlim/(nCols)
spacingY = ylim/(nRows)
print 'xlim = ', xlim
print 'ylim = ', ylim
print 'spacingX = ', spacingX
print 'spacingY = ', spacingY
print 'spacingOriginal = ', 5*126.4

pointsx = np.linspace(start=0, stop=nCols*spacingX, num=nCols)
print pointsx
pointsy = np.linspace(start=0, stop=nRows*spacingY, num=nRows)
print pointsy
xpoints, ypoints = np.meshgrid(pointsx, pointsy)
turbineX = np.ndarray.flatten(xpoints)
turbineY = np.ndarray.flatten(ypoints)

# Center the reduce are grid, the offset equal half the difference between the original limits and the modified limits
turbineX = turbineX + 290
turbineY = turbineY + 220

# print turbineX
# print turbineY
np.savetxt('layout.txt', np.c_[turbineX, turbineY])



