
import json
import numpy as np
import matplotlib.pyplot as plt
import prettify

f = open('figure1.json', 'r')
r = json.load(f)
f.close()

p1s = np.array(r['speed_grid']['power'])
s1 = np.array(r['speed_grid']['speed'])
p2s = np.array(r['speed_amalia']['power'])
s2 = np.array(r['speed_amalia']['speed'])
p3s = np.array(r['speed_optimized']['power'])
s3 = np.array(r['speed_optimized']['speed'])
p4s = np.array(r['speed_random']['power'])
s4 = np.array(r['speed_random']['speed'])

p1d = np.array(r['dir_grid']['power'])
d1 = np.array(r['dir_grid']['direction'])
p2d = np.array(r['dir_amalia']['power'])
d2 = np.array(r['dir_amalia']['direction'])
p3d = np.array(r['dir_optimized']['power'])
d3 = np.array(r['dir_optimized']['direction'])
p4d = np.array(r['dir_random']['power'])
d4 = np.array(r['dir_random']['direction'])

# Move the turbine arrays to the json file too. Once I have the final set
locations = np.genfromtxt('../convergence_results/layout_grid.txt', delimiter='')
tXg = locations[:, 0]
tYg = locations[:, 1]

locations = np.genfromtxt('../convergence_results/layout_amalia.txt', delimiter='')
tXa = locations[:, 0]
tYa = locations[:, 1]

locations = np.genfromtxt('../convergence_results/layout_optimized.txt', delimiter='')
tXo = locations[:, 0]
tYo = locations[:, 1]

locations = np.genfromtxt('../convergence_results/layout_random.txt', delimiter='')
tXr = locations[:, 0]
tYr = locations[:, 1]

tx = [tXg, tXa, tXo, tXr]
ty = [tYg, tYa, tYo, tYr]

power_direction = [p1d, p2d, p3d, p4d]
power_speed = [p1s, p2s, p3s, p4s]
d = d1  # all the directions and the speeds are the same
s = s1


# fig, ax = plt.subplots(1, 3, figsize=(18, 6), gridspec_kw={'width_ratios': [0.2,1,1]})
# ax[0].scatter(tXg, tYg)
# ax[0].set_axis_off()
# ax[0].set_xlim([-190, 3990])
# ax[0].set_ylim([-245, 5145])
# ax[0].set_aspect('equal')
# ax[1].plot(d1, p1d)
# ax[2].plot(s1, p1s)
# fig.tight_layout()

# fig, ax = plt.subplots(1, 1, figsize=(8, 6))
# prettify.set_color_cycle(ax)
# ax.plot(d1, p1d)
# prettify.remove_junk(ax)



fig, ax = plt.subplots(4, 3, figsize=(12, 12), gridspec_kw={'width_ratios': [0.2,1,1]})
# Loop over all the axes and prettify the graph.  # later have that in the prettify
for Ax in ax:
    for AX in Ax:
        prettify.set_color_cycle(AX)
        prettify.remove_junk(AX)

for i, x in enumerate(tx):
    # The turbine locations plot
    ax[i][0].scatter(tx[i], ty[i])
    ax[i][0].set_axis_off()
    ax[i][0].set_xlim([-190, 3990])
    ax[i][0].set_ylim([-245, 5145])
    ax[i][0].set_aspect('equal')

    # The wind direction plot
    ax[i][1].plot(d, power_direction[i]/1000)
    ax[i][1].set_ylabel(r'$power\, (MW)$')
    ax[i][1].set_xlim([-10, 370])
    ax[i][1].set_ylim([38, 98])
    ax[i][1].set_xticks([0,90,180,270,360])

    # The wind speed plot
    ax[i][2].plot(s, power_speed[i]/1000)
    ax[i][2].set_xlim([-1, 31])
    ax[i][2].set_ylim([-10, 310])
    ax[i][2].grid(True)
    # gridlines = ax[i][2].get_xgridlines() + ax[i][2].get_ygridlines()
    # for line in gridlines:
    #     line.set_linestyle('-')
    #     line.set_color('w')

# Titles, labels and stuff.
ax[0][1].set_title(r'$wind\, direction\, (deg)$')
ax[3][1].set_xlabel(r'$wind\, direction\, (deg)$')
ax[0][2].set_title(r'$wind\, speed\, (m/s)$')
ax[3][2].set_xlabel(r'$wind\, speed\, (m/s)$')


fig.tight_layout()
plt.savefig('power_vs_winddirection_vs_windspeed.pdf', transparent=True)

plt.show()
