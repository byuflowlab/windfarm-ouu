
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
tx = [tXo]
ty = [tYo]


# Figure Direction Mean, min, max

# Different layouts

methods = [['dir_rect.json', 'dir_dakota.json'], ['2d_rect_average.json', '2d_dakotaq_average_trunc.json']]
#methods = [['dir_dakotar_average.json', 'dir_dakota.json'], ['2d_rect_average.json', '2d_dakotaq_average_trunc.json']]
lay = 'optimized'

n = [45, 20]  # number of points to consider

fig, ax = plt.subplots(2, 2, figsize=(9, 6), gridspec_kw={'width_ratios': [1, 1]})
# Loop over all the axes and prettify the graph.  # later have that in the prettify
for Ax in ax:
    for AX in Ax:
        prettify.set_color_cycle(AX)
        prettify.remove_junk(AX)

for i, dim in enumerate(methods):

    # Get the baseline
    f = open(dim[0], 'r')
    r = json.load(f)
    f.close()
    mu_base = r[lay]['average']['mu']#[-1]
    print len(mu_base)
    mu_base = mu_base[-1]
    print mu_base  # The prints help to determine the axes range.

    for j, method in enumerate(dim):

        f = open(method, 'r')
        r = json.load(f)
        f.close()

        # Baseline values for error bounds and mean values
        s = r[lay]['average']['s'][:n[i]]
        mu_ave = r[lay]['average']['mu'][:n[i]]

        mu_max = r[lay]['max']['mu'][:n[i]]
        mu_min = r[lay]['min']['mu'][:n[i]]

        # Plot the average values and the bounds
        ax[i][j].plot(s, mu_ave, label='average', linewidth=2)
        ax[i][j].plot(s, np.ones(len(s))*mu_base+0.01*mu_base, 'k--', label='1% bounds')
        ax[i][j].plot(s, np.ones(len(s))*mu_base-0.01*mu_base, 'k--', label='1% bounds')

        # Plot the min, max range
        ax[i][j].fill_between(s, mu_min, mu_max, facecolor='blue', alpha=0.2)
        # ax[i][j].set_ylim([490, 810])
        # ax[i][j].set_xlim([-1, 46])
        ax[0][j].set_ylim([610, 770])
        ax[0][j].set_xlim([-1, 46])
        ax[0][j].xaxis.set_ticks([0, 5, 10, 20, 30, 40])
        ax[1][j].set_ylim([1290, 1640])
        ax[1][j].set_xlim([-10, 410])
        ax[1][j].xaxis.set_ticks([0, 50, 100, 200, 300, 400])

        print np.min(mu_min)
        print np.max(mu_max)


    # ax[i][0].set_ylabel(r'$AEP\, (GWh)$', rotation=90)
    ax[i][0].set_ylabel('AEP (GWh)', rotation=90)


ax[0][0].set_title('1d rectangle rule')
ax[0][1].set_title('1d polynomial chaos')
ax[1][0].set_title('2d rectangle rule')
ax[1][1].set_title('2d polynomial chaos')

# ax[0][1].get_yaxis().set_label_coords(-0.1,0.5)
ax[0][0].set_xlabel('number of wind directions')
ax[0][1].set_xlabel('number of wind directions')
ax[1][0].set_xlabel('number of wind speeds and directions')
ax[1][1].set_xlabel('number of wind speeds and directions')
# Put 1% annotation
ax[0][0].annotate('',
    xy=(40, 717), xycoords='data',
    xytext=(40, 757), textcoords='data',
    arrowprops=dict(arrowstyle="->",
                    connectionstyle="arc3"),
    )
ax[0][0].annotate('',
    xy=(40, 703), xycoords='data',
    xytext=(40, 663), textcoords='data',
    arrowprops=dict(arrowstyle="->",
                    connectionstyle="arc3"),
    )
ax[0][0].annotate(
    r'$\pm$1%', xy=(40, 737), xycoords='data',
    xytext=(5, 0), textcoords='offset points')

# Specify the cases
# ax[0][1].annotate(
#     '1d case', xy=(40, 690), xycoords='data',
#     xytext=(0, 0), textcoords='offset points')

fig.tight_layout()
plt.savefig('Statistics_convergence_mean_min_max_direction.pdf', transparent=True)

plt.show()
