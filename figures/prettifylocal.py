
def set_color_cycle(ax):
    '''Change the default color cycle of matplotlib with one of tableau'''
    color_cycle = tableau_colors()
    # Change for newer matplotlib 1.5
    #ax.set_color_cycle(color_cycle)
    ax.set_prop_cycle('color',color_cycle)

def remove_junk(ax):
    '''Remove the extra stuff in the plots and lighten the axes'''

    # Color for the axes
    #light_grey = (127/255.,127/255.,127/255.)
    light_grey = (64/255.,64/255.,64/255.)

    # Remove the top and right axes lines ('splines')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Get rid of ticks only on the right and top boundary
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    # Remove the ticks on all boundaries
    #ax.xaxis.set_ticks_position('none')
    #ax.yaxis.set_ticks_position('none')

    # Color the axes
    axis_color = light_grey
    ax.spines['bottom'].set_color(axis_color)
    ax.spines['left'].set_color(axis_color)
    ax.xaxis.label.set_color(axis_color)
    ax.yaxis.label.set_color(axis_color)
    ax.title.set_color(axis_color)
    [t.set_color(axis_color) for t in ax.xaxis.get_ticklines()]
    [t.set_color(axis_color) for t in ax.xaxis.get_ticklabels()]
    [t.set_color(axis_color) for t in ax.yaxis.get_ticklines()]
    [t.set_color(axis_color) for t in ax.yaxis.get_ticklabels()]

def remove_junk3D(ax):
    '''Remove the extra stuff in the plots and lighten the axes'''

    # Color for the axes
    light_grey = (127/255.,127/255.,127/255.)

    # Remove the top and right axes lines ('splines')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Get rid of ticks only on the right and top boundary
    #ax.get_xaxis().tick_bottom()
    #ax.get_yaxis().tick_left()
    # Remove the ticks on all boundaries
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.zaxis.set_ticks_position('none')

    # Color the axes
    axis_color = light_grey
    ax.spines['bottom'].set_color(axis_color)
    ax.spines['left'].set_color(axis_color)
    ax.xaxis.label.set_color(axis_color)
    ax.yaxis.label.set_color(axis_color)
    ax.zaxis.label.set_color(axis_color)
    [t.set_color(axis_color) for t in ax.xaxis.get_ticklines()]
    [t.set_color(axis_color) for t in ax.xaxis.get_ticklabels()]
    [t.set_color(axis_color) for t in ax.yaxis.get_ticklines()]
    [t.set_color(axis_color) for t in ax.yaxis.get_ticklabels()]
    [t.set_color(axis_color) for t in ax.zaxis.get_ticklines()]
    [t.set_color(axis_color) for t in ax.zaxis.get_ticklabels()]

def tableau_colors():
    '''Return a tableau color set'''

    # Tableau colors
    # These are the "Tableau 20" colors as RGB.
    tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
    # Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
    for i in range(len(tableau20)):
        r, g, b = tableau20[i]
        tableau20[i] = (r / 255., g / 255., b / 255.)

    # Tableau Color Blind 10
    tableau10blind = [(0, 107, 164), (255, 128, 14), (171, 171, 171), (89, 89, 89),
             (95, 158, 209), (200, 82, 0), (137, 137, 137), (163, 200, 236),
             (255, 188, 121), (207, 207, 207)]
    for i in range(len(tableau10blind)):
        r, g, b = tableau10blind[i]
        tableau10blind[i] = (r / 255., g / 255., b / 255.)

    # Tableau 10
    tableau10 = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40),
             (148, 103, 189), (140, 86, 75), (227, 119, 194), (127, 127, 127),               (188,189,34), (23,190,207)]

    for i in range(len(tableau10)):
        r, g, b = tableau10[i]
        tableau10[i] = (r / 255., g / 255., b / 255.)

    return tableau10

def color_legend_text(leg):
    """Color legend texts based on color of corresponding lines"""
    for line, txt in zip(leg.get_lines(), leg.get_texts()):
        txt.set_color(line.get_color())

### Other Stuff ###
# Ideally for an xy plot you probably want a ratio of 1.33
# The default size in matplotlib is (8in,6in)
# fig.set_size_inches(10,7.5) to set size
#
# ax.annotate for annotating plot
# #ax.annotate('Bartels', xy=(15, -0.19),  xycoords='data',
#             xytext=(-10, 50), textcoords='offset points',
#             arrowprops=dict(arrowstyle="->") )
# ax.add_patch(mpl.patches.Rectangle((50, -0.195), 8, 0.1,color='white',zorder=10)) To cover up the symbols of the legend.
# Make labels look nice, get rid of zeros
# major_formatter = mpl.ticker.FormatStrFormatter('%g')
# ax.yaxis.set_major_formatter(major_formatter)
