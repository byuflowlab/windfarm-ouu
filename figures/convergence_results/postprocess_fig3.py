
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import sys
sys.path.append('../')
import prettifylocal as prettify

locations = np.genfromtxt('../../WindFarms/layout_grid.txt', delimiter='')
tXg = locations[:, 0]
tYg = locations[:, 1]

locations = np.genfromtxt('../../WindFarms/layout_amalia.txt', delimiter='')
tXa = locations[:, 0]
tYa = locations[:, 1]

locations = np.genfromtxt('../../WindFarms/layout_optimized.txt', delimiter='')
tXo = locations[:, 0]
tYo = locations[:, 1]

locations = np.genfromtxt('../../WindFarms/layout_random.txt', delimiter='')
tXr = locations[:, 0]
tYr = locations[:, 1]

# turbine locations
tx = [tXg, tXa, tXo, tXr]
ty = [tYg, tYa, tYo, tYr]


# Figure Direction Mean Error, Speed Error and combined Error. (Log case)

# Different layouts

method_dir = ['dir_rect.json', 'dir_dakota.json']
# method_dir = ['dir_dakotar_average.json', 'dir_dakota.json']
# method_speed = ['speed_rect.json', 'speed_dakota.json']
method_speed = ['speed_recttrunc.json', 'speed_dakotaqtrunc.json']
#method_speed = ['speed_chaospy.json', 'speed_dakota.json']
method_2d = ['2d_rect_average.json', '2d_dakotaq_average_trunc.json']
# method_2d = ['2d_dakotaq_average.json', '2d_dakotaq_average_trunc.json']
layout = ['grid', 'amalia', 'optimized', 'random']

n = 45  # number of points to consider

fig, ax = plt.subplots(4, 4, figsize=(17, 12), gridspec_kw={'width_ratios': [0.2, 1, 1, 1]})
# Loop over all the axes and prettify the graph.  # later have that in the prettify
for Ax in ax:
    for AX in Ax:
        prettify.set_color_cycle(AX)
        prettify.remove_junk(AX)

ax[0][0].set_title('layout', position=(0.5, 1.6))
method = method_dir
for i, lay in enumerate(layout):
    # Get the baseline
    f = open(method[0], 'r')
    r = json.load(f)
    f.close()
    mu_base = r[lay]['average']['mu'][-1]

    # Plot the layout
    ax[i][0].scatter(tx[i], ty[i])
    ax[i][0].set_axis_off()
    ax[i][0].set_xlim([-190, 3990])
    ax[i][0].set_ylim([-245, 5145])
    ax[i][0].set_aspect('equal')

    for j, m in enumerate(method, 1):
        f = open(m, 'r')
        r = json.load(f)
        f.close()

        # Baseline values for error bounds and mean values
        s = r[lay]['average']['s'][:n]
        mu_ave = np.array(r[lay]['average']['mu'][:n])

        mu_max = np.array(r[lay]['max']['mu'][:n])
        mu_min = np.array(r[lay]['min']['mu'][:n])

        # If I want to include a particular error
        # mu_max = np.array(r[lay]['1']['mu'][:n])
        # mu_min = np.array(r[lay]['1']['mu'][:n])

        mu_err = np.abs((mu_ave-mu_base))/mu_base * 100
        mu_err_max = np.max((np.abs(mu_max-mu_base), np.abs(mu_min-mu_base)), axis=0)/mu_base * 100
        # Plot the average values and the bounds
        ax[i][1].plot(s, mu_err, linewidth=2) #, label=m.split('.')[0])
        # ax[i][1].plot(s, mu_err_max, '--')
        # ax[i][1].legend()
    # ax[i][1].legend(['rectangle rule', 'polynomial chaos'], frameon=False)
        # Plot the min, max range
        # ax[i][1].fill_between(s, mu_min, mu_max, facecolor='blue', alpha=0.2 )
    ax[i][1].plot(s, np.ones(len(s)), 'k--', label='1% bounds')
    ax[i][1].set_ylim([-0.2, 10.2])
    ax[i][1].set_xlim([-1, 46])
    ax[i][1].set_ylabel('% average error', rotation=90, fontsize=14)
    ax[i][1].yaxis.set_ticks([0, 1, 2, 5, 10])
    ax[i][1].xaxis.set_ticks([0, 5, 10, 20, 30, 40])
    ax[i][1].tick_params(axis='both', labelsize=14)
ax[0][1].legend(['rectangle rule', 'polynomial chaos'], frameon=False)
ax[0][1].set_title('wind direction')
ax[3][1].set_xlabel('number of wind directions', fontsize=14)

# Now add the speed information, later combine in just one loop
method = method_speed
for i, lay in enumerate(layout):
    # Get the baseline
    f = open(method[0], 'r')
    r = json.load(f)
    f.close()
    mu_base = r[lay]['0']['mu'][-1]

    for j, m in enumerate(method, 1):
        f = open(m, 'r')
        r = json.load(f)
        f.close()

        # Baseline values for error bounds and mean values
        s = r[lay]['0']['s'][:n]
        mu_ave = np.array(r[lay]['0']['mu'][:n])

        mu_err = np.abs((mu_ave-mu_base))/mu_base * 100
        # Plot the average values and the bounds
        ax[i][2].plot(s, mu_err, linewidth=2) #, label=m.split('.')[0])
        # ax[i][1].plot(s, mu_err_max, '--')
        # ax[i][1].legend()
    # ax[i][2].legend(['rectangle rule', 'polynomial chaos'], frameon=False)
    ax[i][2].plot(s, np.ones(len(s)), 'k--', label='1% bounds')
    ax[i][2].set_ylim([-0.2, 10.2])
    ax[i][2].set_xlim([-1, 46])
    ax[i][2].set_ylabel('% error', rotation=90)
    ax[i][2].yaxis.set_ticks([0, 1, 2, 5, 10])
    ax[i][2].xaxis.set_ticks([0, 5, 10, 20, 30, 40])
    ax[i][2].tick_params(axis='both', labelsize=14)

# ax[0][2].legend(['rectangle rule', 'polynomial chaos'], frameon=False)
ax[0][2].set_title('wind speed')
ax[3][2].set_xlabel('number of wind speeds')

# Now add the 2d information, later combine in just one loop
method = method_2d
n = 25
for i, lay in enumerate(layout):
    # Get the baseline
    f = open(method[0], 'r')
    r = json.load(f)
    f.close()
    mu_base = r[lay]['0']['mu'][-1]  # Originally I had these
    mu_base = r[lay]['average']['mu'][-1]

    for j, m in enumerate(method, 1):
        f = open(m, 'r')
        r = json.load(f)
        f.close()

        # Baseline values for error bounds and mean values
        s = r[lay]['0']['s'][:n]  # Originally I had these.
        mu_ave = np.array(r[lay]['0']['mu'][:n])  # Originally I had these
        s = r[lay]['average']['s'][:n]
        mu_ave = np.array(r[lay]['average']['mu'][:n])

        mu_err = np.abs((mu_ave-mu_base))/mu_base * 100
        # Plot the average values and the bounds
        ax[i][3].plot(s, mu_err, linewidth=2) #, label=m.split('.')[0])
        # ax[i][1].plot(s, mu_err_max, '--')
        # ax[i][1].legend()
    # ax[i][2].legend(['rectangle rule', 'polynomial chaos'], frameon=False)
    ax[i][3].plot(s, np.ones(len(s)), 'k--', label='1% bounds')
    ax[i][3].set_ylim([-0.2, 10.2])
    # ax[i][3].set_xlim([-10, 410])
    ax[i][3].set_xlim([-10, 635])
    ax[i][3].set_ylabel('% average error', rotation=90)
    ax[i][3].yaxis.set_ticks([0, 1, 2, 5, 10])
    # ax[i][3].xaxis.set_ticks([0, 5, 10, 20, 30, 40])
    ax[i][3].tick_params(axis='both', labelsize=14)

# ax[0][3].legend(['rectangle rule', 'polynomial chaos'], frameon=False)
ax[0][3].set_title('wind speed and direction')
ax[3][3].set_xlabel('number of wind speeds and directions')

fig.tight_layout()
plt.savefig('MeanError_statistics_convergence.pdf', transparent=True)
plt.savefig('MeanError_statistics_convergence.svg')


plt.show()
