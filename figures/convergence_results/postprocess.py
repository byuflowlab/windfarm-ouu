
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import sys
sys.path.append('../')
import prettifylocal as prettify

locations = np.genfromtxt('layout_grid.txt', delimiter='')
tXg = locations[:, 0]
tYg = locations[:, 1]

locations = np.genfromtxt('layout_amalia.txt', delimiter='')
tXa = locations[:, 0]
tYa = locations[:, 1]

locations = np.genfromtxt('layout_optimized.txt', delimiter='')
tXo = locations[:, 0]
tYo = locations[:, 1]

locations = np.genfromtxt('layout_random.txt', delimiter='')
tXr = locations[:, 0]
tYr = locations[:, 1]

# turbine locations
tx = [tXg, tXa, tXo, tXr]
ty = [tYg, tYa, tYo, tYr]


# Figure Direction Mean, min, max

# Different layouts

method = ['dir_rect.json', 'dir_dakota.json']
layout = ['grid', 'amalia', 'optimized', 'random']

n = 45  # number of points to consider

fig, ax = plt.subplots(4, 3, figsize=(12, 12), gridspec_kw={'width_ratios': [0.2, 1, 1]})
# Loop over all the axes and prettify the graph.  # later have that in the prettify
for Ax in ax:
    for AX in Ax:
        prettify.set_color_cycle(AX)
        prettify.remove_junk(AX)

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
        mu_ave = r[lay]['average']['mu'][:n]

        mu_max = r[lay]['max']['mu'][:n]
        mu_min = r[lay]['min']['mu'][:n]

        # Plot the average values and the bounds
        ax[i][j].plot(s, mu_ave, label='average', linewidth=2)
        ax[i][j].plot(s, np.ones(len(s))*mu_base+0.01*mu_base, 'k--', label='1% bounds')
        ax[i][j].plot(s, np.ones(len(s))*mu_base-0.01*mu_base, 'k--', label='1% bounds')

        # Plot the min, max range
        ax[i][j].fill_between(s, mu_min, mu_max, facecolor='blue', alpha=0.2)
        ax[i][j].set_ylim([490, 810])
        ax[i][j].set_xlim([-1, 46])

    # ax[i][1].set_ylabel(r'$AEP\, (GWh)$', rotation=90)
    ax[i][1].set_ylabel('AEP (GWh)', rotation=90)


ax[0][1].set_title('rectangle rule')
ax[0][2].set_title('polynomial chaos')
# ax[0][1].get_yaxis().set_label_coords(-0.1,0.5)
ax[3][1].set_xlabel('number of wind directions')
ax[3][2].set_xlabel('number of wind directions')
# Put 1% annotation
ax[0][1].annotate('',
    xy=(40, 731), xycoords='data',
    xytext=(40, 781), textcoords='data',
    arrowprops=dict(arrowstyle="->",
                    connectionstyle="arc3"),
    )
ax[0][1].annotate('',
    xy=(40, 717), xycoords='data',
    xytext=(40, 667), textcoords='data',
    arrowprops=dict(arrowstyle="->",
                    connectionstyle="arc3"),
    )
ax[0][1].annotate(
    r'$\pm$1%', xy=(40, 756), xycoords='data',
    xytext=(5, 0), textcoords='offset points')


fig.tight_layout()
plt.savefig('Statistics_convergence_mean_min_max_direction.pdf', transparent=True)


# Figure Direction Mean Error and Speed Error

# Different layouts

method = ['dir_rect.json', 'dir_dakota.json']
method_speed = ['speed_rect.json', 'speed_dakota.json']
layout = ['grid', 'amalia', 'optimized', 'random']

n = 45  # number of points to consider

fig, ax = plt.subplots(4, 3, figsize=(12, 12), gridspec_kw={'width_ratios': [0.2, 1, 1]})
# Loop over all the axes and prettify the graph.  # later have that in the prettify
for Ax in ax:
    for AX in Ax:
        prettify.set_color_cycle(AX)
        prettify.remove_junk(AX)

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

        mu_err = np.abs((mu_ave-mu_base))/mu_base *100
        mu_err_max = np.max((np.abs(mu_max-mu_base), np.abs(mu_min-mu_base)), axis=0)/mu_base *100
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
    ax[i][1].set_ylabel('% average error', rotation=90)
    ax[i][1].yaxis.set_ticks([0, 1, 2, 5, 10])
    ax[i][1].xaxis.set_ticks([0, 5, 10, 20, 30, 40])
ax[0][1].legend(['rectangle rule', 'polynomial chaos'], frameon=False)
ax[0][1].set_title('number of wind directions')
ax[3][1].set_xlabel('number of wind directions')

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

ax[0][2].legend(['rectangle rule', 'polynomial chaos'], frameon=False)
ax[0][2].set_title('number of wind speeds')
ax[3][2].set_xlabel('number of wind speeds')

fig.tight_layout()
plt.savefig('MeanError_statistics_convergence.pdf', transparent=True)


plt.show()
