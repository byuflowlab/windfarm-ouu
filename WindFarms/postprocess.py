import numpy as np
import matplotlib.pyplot as plt
import prettify

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

diameter = 126.4  # (m)
colors = prettify.tableau_colors()

fig, ax = plt.subplots(figsize=(10, 10))
# ax.scatter(tXg, tYg, label='grid', color=colors[0])
# ax.scatter(tXa, tYa, label='amalia', color=colors[1])
# ax.scatter(tXo, tYo, label='optimized', color=colors[2])
# ax.scatter(tXr, tYr, label='random', color=colors[3])

# Make the figure with the size of the circle corresponding to the size of the turbine.

# Grid
# for x, y, r in zip(tXg, tYg, [diameter]*len(tXg)):
#     ax.add_artist(plt.Circle((x, y), radius=diameter/2, edgecolor='none', color=colors[7]))

# Amalia
# for x, y, r in zip(tXa, tYa, [diameter]*len(tXa)):
#     ax.add_artist(plt.Circle((x, y), radius=diameter/2, edgecolor='none', color=colors[7]))

# Optimized
for x, y, r in zip(tXo, tYo, [diameter]*len(tXo)):
    ax.add_artist(plt.Circle((x, y), radius=diameter/2, edgecolor='none', color=colors[7]))

# Random
# for x, y, r in zip(tXr, tYr, [diameter]*len(tXr)):
#     ax.add_artist(plt.Circle((x, y), radius=diameter/2, edgecolor='none', color=colors[7]))

# # Add circle around everything
cx = 1900  # Basically tXa/2
cy = 2445  # Basically tYa/2
r = 2690  # With the extra 5% variation

# Add some arrows to showcase directions
l = 0.3  # arrow length % of radius
grey = [64/255., 64/255., 64/255.]

angles = range(0, 360, 45)
deg2rad = np.pi/180

for angle in angles:
    angle = angle*deg2rad
    x = r*np.cos(angle)
    y = r*np.sin(angle)
    # For arrow use an empty annotate instead of arrow. The annotate is more versatile.
    arrowprops=dict(fc=grey, ec=grey, width=5, headwidth=18, headlength=18)
    ax.annotate('', (cx+x, cy+y), (cx+(1+l)*x, cy+(1+l)*y), arrowprops=arrowprops)

angles = range(-10, 350, 45)  # Minus because the zero is defined from x-axis not North
deg2rad = np.pi/180

for angle in angles:
    angle = angle*deg2rad
    x = r*np.cos(angle)
    y = r*np.sin(angle)
    # For arrow use an empty annotate instead of arrow. The annotate is more versatile.
    arrowprops=dict(fc=colors[3], ec=colors[3], width=5, headwidth=18, headlength=18)
    ax.annotate('', (cx+x, cy+y), (cx+(1+l)*x, cy+(1+l)*y), arrowprops=arrowprops)

# ax.add_artist(plt.Circle((cx, cy), r, fill=False, linestyle='dashed', edgecolor=colors[7]))
# ax.arrow(cx, cy, r*np.sin(np.pi/4), r*np.cos(np.pi/4), head_width=0.05, head_length=0.1, fc='k', ec='k')


# Range long enough to cover the arrows as well.
ax.set_xlim([-1600, 5400])
ax.set_ylim([-1052, 5942])
# Range just to encircle the turbines
# ax.set_xlim([-190, 3990]) # Range of variation of the Amalia times +0.05% on each side.
# ax.set_ylim([-245, 5145])
ax.set_aspect('equal')
ax.set_axis_off()

fig.tight_layout()
plt.savefig('amalia_layout.pdf')

plt.show()
