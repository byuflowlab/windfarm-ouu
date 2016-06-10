import numpy as np
import matplotlib.pyplot as plt
import json


# Amalia baseline coordinates
tXama = np.array([973, 1553, 2158, 721, 1290, 1895, 2515, 3136, 469, 1033, 1623, 2238, 2853, 3483, 217, 776, 1356, 1956, 2571, 3186, 3801, 519, 1094, 1684, 2284, 2894, 3503, 257, 822, 1406, 1996, 2601, 3201, 3781, 0, 560, 1129, 1714, 2309, 2904, 3468, 287, 852, 1432, 2006, 2596, 3156, 3710, 20, 575, 1139, 1719, 2284, 2843, 297, 852, 1427, 1986, 1124, 1684])
tYama = np.array([10, 0, 20, 509, 494, 509, 539, 580, 1003, 978, 988, 1023, 1059, 1104, 1497, 1472, 1477, 1497, 1532, 1573, 1633, 1961, 1961, 1986, 2006, 2047, 2097, 2460, 2450, 2460, 2485, 2520, 2566, 2611, 2949, 2934, 2944, 2954, 2989, 3025, 3070, 3423, 3428, 3438, 3458, 3493, 3534, 3569, 3912, 3912, 3912, 3927, 3962, 3992, 4391, 4386, 4401, 4421, 4870, 4890])


method = ['dakota_opt_muactual.json', 'rect_opt_muactual.json']
samples = [['5', '10', '15', '20', '30', '40'], ['5', '10', '15', '20', '30', '40', '66', '100']]
# values = ['std', 'obj', 'tX', 'tY', 'iter', 'mu', 'mu_actual']
case = '0'
fig, ax = plt.subplots(2, 9, figsize=(24,8))
for j, m in enumerate(method):
    # Load the json file with all the information
    f = open(m, 'r')
    r = json.load(f)
    f.close()
    for i, n in enumerate(samples[j], 1):
        tX = r[n][case]['tX']
        tY = r[n][case]['tY']
        mu_actual = r[n][case]['mu_actual']
        mu = r[n][case]['mu']
        ax[j][i].scatter(tX, tY)
        ax[j][i].set_axis_off()
        ax[j][i].set_xlim([-190, 3990])
        ax[j][i].set_ylim([-245, 5145])
        ax[j][i].set_aspect('equal')
        # towrite = n #+ ' - ' + str(int(round(mu_actual))) + ' ' + str(int(round(mu)))
        # ax[j][i].set_title(towrite)
        ax[1][i].text(1600, -700, n)
        ax[j][i].text(3200, 5000, str(int(round(mu_actual))), fontsize=14)
        ax[j][i].text(3200, 0, str(int(round(mu))))


# Baseline plot
ax[0][0].scatter(tXama, tYama)
ax[0][0].set_axis_off()
ax[0][0].set_xlim([-190, 3990])
ax[0][0].set_ylim([-245, 5145])
ax[0][0].set_aspect('equal')
ax[1][0].scatter(tXama, tYama)
ax[1][0].set_axis_off()
ax[1][0].set_xlim([-190, 3990])
ax[1][0].set_ylim([-245, 5145])
ax[1][0].set_aspect('equal')

# Clear the last grids
ax[0][7].set_axis_off()
ax[0][8].set_axis_off()


# ax[0][0].text(3200, 5000, 'aloha')
# ax[0][0].text(3200, 0, 'aloha')

ax[0][4].text(4100, 5000, 'GWh - AEP of the layout as measured by using rectangular integration with n=100 directions')
#plt.savefig('optimizaton.pdf')


plt.show()
