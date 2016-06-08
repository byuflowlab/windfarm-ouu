
import json
import numpy as np
import matplotlib.pyplot as plt
import prettify

f = open('figure1.json', 'r')
r = json.load(f)
f.close()

p1s = np.array(r['grid_speed']['power'])
s1 = np.array(r['grid_speed']['speed'])
p2s = np.array(r['amalia_speed']['power'])
s2 = np.array(r['amalia_speed']['speed'])
p3s = np.array(r['opt_speed']['power'])
s3 = np.array(r['opt_speed']['speed'])
p4s = np.array(r['lhs_speed']['power'])
s4 = np.array(r['lhs_speed']['speed'])

p1d = np.array(r['grid_dir']['power'])
d1 = np.array(r['grid_dir']['direction'])
p2d = np.array(r['amalia_dir']['power'])
d2 = np.array(r['amalia_dir']['direction'])
p3d = np.array(r['opt_dir']['power'])
d3 = np.array(r['opt_dir']['direction'])
p4d = np.array(r['lhs_dir']['power'])
d4 = np.array(r['lhs_dir']['direction'])

# Move the turbine arrays to the json file too. Once I have the final set
tXlhs = np.array([2313, 289, 1205, 644, 2957, 939, 2173, 627, 2262, 3749, 352, 3379, 2608, 3026, 3481, 1978, 1145, 3585, 743, 207, 3273, 2123, 826, 2352, 2419, 2090, 1933, 3140, 47, 2694, 564, 1761, 469, 2809, 1327, 3315, 1828, 433, 1871, 1008, 1526, 3617, 1058, 3096, 1593, 81, 1119, 2544, 171, 791, 1690, 1363, 1396, 1471, 2886, 2478, 3198, 2732, 3677, 3495])
tYlhs = np.array([2813, 3747, 3916, 4286, 1099, 705, 517, 4746, 2699, 1359, 66, 2668, 2177, 3138, 4709, 4427, 136, 942, 878, 1151, 3576, 4874, 3977, 4532, 488, 1650, 1903, 2060, 984, 2279, 170, 2336, 4380, 603, 2941, 1620, 1501, 3393, 385, 1443, 1853, 3213, 3790, 1980, 3346, 2882, 2462, 2379, 812, 274, 1775, 1300, 2589, 4033, 4650, 4143, 3613, 3089, 4227, 3462])

tXgrid = np.array([0, 760, 1520, 2280, 3040, 3800, 0, 760, 1520, 2280, 3040, 3800, 0, 760, 1520, 2280, 3040, 3800, 0, 760, 1520, 2280, 3040, 3800, 0, 760, 1520, 2280, 3040, 3800, 0, 760, 1520, 2280, 3040, 3800, 0, 760, 1520, 2280, 3040, 3800, 0, 760, 1520, 2280, 3040, 3800, 0, 760, 1520, 2280, 3040, 3800, 0, 760, 1520, 2280, 3040, 3800])
tYgrid = np.array([0, 0, 0, 0, 0, 0, 544, 544, 544, 544, 544, 544, 1089, 1089, 1089, 1089, 1089, 1089, 1633, 1633, 1633, 1633, 1633, 1633, 2178, 2178, 2178, 2178, 2178, 2178, 2722, 2722, 2722, 2722, 2722, 2722, 3267, 3267, 3267, 3267, 3267, 3267, 3811, 3811, 3811, 3811, 3811, 3811, 4356, 4356, 4356, 4356, 4356, 4356, 4900, 4900, 4900, 4900, 4900, 4900])

tXama = np.array([973, 1553, 2158, 721, 1290, 1895, 2515, 3136, 469, 1033, 1623, 2238, 2853, 3483, 217, 776, 1356, 1956, 2571, 3186, 3801, 519, 1094, 1684, 2284, 2894, 3503, 257, 822, 1406, 1996, 2601, 3201, 3781, 0, 560, 1129, 1714, 2309, 2904, 3468, 287, 852, 1432, 2006, 2596, 3156, 3710, 20, 575, 1139, 1719, 2284, 2843, 297, 852, 1427, 1986, 1124, 1684])
tYama = np.array([10, 0, 20, 509, 494, 509, 539, 580, 1003, 978, 988, 1023, 1059, 1104, 1497, 1472, 1477, 1497, 1532, 1573, 1633, 1961, 1961, 1986, 2006, 2047, 2097, 2460, 2450, 2460, 2485, 2520, 2566, 2611, 2949, 2934, 2944, 2954, 2989, 3025, 3070, 3423, 3428, 3438, 3458, 3493, 3534, 3569, 3912, 3912, 3912, 3927, 3962, 3992, 4391, 4386, 4401, 4421, 4870, 4890])

tXopt = np.array([1270, 1879, 2214, 840, 1308, 1929, 2639, 3136, 556, 850, 1617, 2253, 3273, 3725, 238, 354, 1240, 2424, 2954, 3409, 3790, 307, 941, 2030, 2561, 2737, 3616, 121, 1124, 1624, 2059, 2371, 3094, 3707, 48, 9, 1136, 1165, 1879, 3202, 3727, 9, 510, 1193, 1885, 2871, 3210, 3710, 17, 621, 1262, 1885, 2363, 2631, 297, 727, 1502, 1978, 1215, 1752])
tYopt = np.array([5, 11, 53, 271, 319, 476, 296, 580, 830, 616, 1024, 558, 1014, 1507, 1455, 1227, 1323, 1471, 1771, 1583, 2153, 1930, 1544, 1761, 1958, 2916, 2336, 2137, 2302, 2348, 2136, 3077, 2714, 2810, 2630, 2995, 2699, 3153, 3232, 3553, 3170, 3324, 2834, 3492, 3665, 4116, 3895, 3569, 3765, 4031, 4125, 4284, 4447, 4273, 4391, 4640, 4883, 4698, 4873, 4845])

# Values

# import matplotlib.gridspec as gridspec
# gs = gridspec.GridSpec(2, 2,
#                        #width_ratios=[1,2],
#                        height_ratios=[4,1]
#                        )
#
# ax1 = plt.subplot(gs[0])
# ax2 = plt.subplot(gs[1])
# ax3 = plt.subplot(gs[2])
# ax4 = plt.subplot(gs[3])

# fig, ax = plt.subplots(1, 3, figsize=(18, 6))
# ax[0].scatter(tX, tY)
# ax[0].set_xlim([-190, 3990])
# ax[0].set_ylim([-245, 5145])
# # ax[0].set_aspect('equal')
# ax[1].plot(s1, mu1, label='rect')
# ax[2].plot(s2, mu2, label='PC')
#
# fig, ax = plt.subplots(1, 3, figsize=(18, 6), gridspec_kw={'width_ratios': [0.2,1,1]})
# ax[0].scatter(tXgrid, tYgrid)
# ax[0].set_axis_off()
# ax[0].set_xlim([-190, 3990])
# ax[0].set_ylim([-245, 5145])
# ax[0].set_aspect('equal')
# ax[1].plot(d1, p1d)
# ax[2].plot(s1, p1s)
# fig.tight_layout()

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(d1, p1d)
prettify.set_color_cycle(ax)
# prettify.remove_junk(ax)

# fig, ax = plt.subplots(4, 3, figsize=(12, 12), gridspec_kw={'width_ratios': [0.2,1,1]})
# # Row 1
# ax[0][0].scatter(tXgrid, tYgrid)
# ax[0][0].set_axis_off()
# ax[0][0].set_xlim([-190, 3990])
# ax[0][0].set_ylim([-245, 5145])
# ax[0][0].set_aspect('equal')
# ax[0][1].plot(d1, p1d)
# ax[0][1].set_title('direction (deg)')
# ax[0][1].set_ylabel('power (kW)')
# ax[0][1].set_xlim([-10, 370])
# ax[0][1].set_xticks([0,90,180,270,360])
# ax[0][2].plot(s1, p1s)
# ax[0][2].set_title('speed (m/s)')
#
# # Row 2
# ax[1][0].scatter(tXama, tYama)
# ax[1][0].set_axis_off()
# ax[1][0].set_xlim([-190, 3990])
# ax[1][0].set_ylim([-245, 5145])
# ax[1][0].set_aspect('equal')
# ax[1][1].plot(d2, p2d)
# ax[1][1].set_xlim([-10, 370])
# ax[1][1].set_xticks([0,90,180,270,360])
# ax[1][2].plot(s2, p2s)
#
#
# # Row 4
# ax[3][0].scatter(tXopt, tYopt)
# ax[3][0].set_axis_off()
# ax[3][0].set_xlim([-190, 3990])
# ax[3][0].set_ylim([-245, 5145])
# ax[3][0].set_aspect('equal')
# ax[3][1].plot(d3, p3d)
# ax[3][1].set_xlim([-10, 370])
# ax[3][1].set_xticks([0,90,180,270,360])
# ax[3][2].plot(s3, p3s)
#
#
# # Row 3
# ax[2][0].scatter(tXlhs, tYlhs)
# ax[2][0].set_axis_off()
# ax[2][0].set_xlim([-190, 3990])
# ax[2][0].set_ylim([-245, 5145])
# ax[2][0].set_aspect('equal')
# ax[2][1].plot(d4, p4d)
# ax[2][1].set_xlim([-10, 370])
# ax[2][1].set_xticks([0,90,180,270,360])
# ax[2][2].plot(s4, p4s)
#
# # Loop over all the axes, later have that in the prettify
# for Ax in ax:
#     for AX in Ax:
#         prettify.set_color_cycle(AX)
#         prettify.remove_junk(AX)
# # prettify.remove_junk(ax[0][1])
# fig.tight_layout()
# plt.savefig('test.pdf')
# plt.savefig('power_vs_winddirection_vs_windspeed.pdf', transparent=True)

# fig, ax = plt.subplots(1, 3, figsize=(18, 6))
# # ax[0].scatter(tX, tY)
# ax[0].set_axis_off()
# ax2 = fig.add_axes([0.1, 0.1, 0.2, 0.2])
# ax2.set_aspect('equal')
# ax2.scatter(tX, tY)
# ax[1].plot(s1, mu1, label='rect')
# ax[2].plot(s2, mu2, label='PC')
# # fig.tight_'optimized'()
# plt.savefig('test.pdf')

# a = fig.get_axes()
# for x in a:
#     print x.get_position()
# print a
# ax[0].plot(s, np.ones(len(s))*base_mu+0.01*base_mu, 'k--', label='1% bounds')
# ax[0].plot(s, np.ones(len(s))*base_mu-0.01*base_mu, 'k--', label='1% bounds')
# ax[0].fill_between(s, mu1_min, mu1_max, facecolor='blue', alpha=0.2 )
# ax[0].fill_between(s, mu2_min, mu2_max, facecolor='green', alpha=0.2 )
#ax2 = plt.axes([.50, .35, .35, .35]) #, axisbg='y')
# ax[2].scatter(tX, tY)
# ax[2].set_axis_off()
# i can make the points not as bright, I need to colors of the lines, thickness and other stuff.

# ax.text(0.6, 0.3, towrite, transform=ax.transAxes)


# ax[0].set_xlabel('# directions')
# ax[0].set_ylabel('mean')
# ax[0].set_title('averaged mean vs # directions')
# ax[0].legend()
# name = 'optimized' + '_mean.pdf'
# plt.savefig(name)

# # Error
# err1 = np.abs((mu1-base_mu))/base_mu *100
# err2 = np.abs((mu2-base_mu))/base_mu *100
#
# err1_max = np.abs((mu1_max-base_mu))/base_mu *100
# err2_max = np.abs((mu2_max-base_mu))/base_mu *100
#
# err1_min = np.abs((mu1_min-base_mu))/base_mu *100
# err2_min = np.abs((mu2_min-base_mu))/base_mu *100
#
# fig, ax = plt.subplots()
# ax.plot(s1, err1, label='rect')
# ax.plot(s2, err2, label='PC')
# #ax.fill_between(s, err1_min, err1_max, facecolor='blue', alpha=0.2 )
# #ax.fill_between(s, err2_min, err2_max, facecolor='green', alpha=0.2 )
# ax.set_xlabel('# directions')
# ax.set_ylabel('% error mean')
# # ax.set_yscale('log')
# ax.set_title('averaged mean vs # directions')
# ax.legend()
# ax2 = plt.axes([.50, .35, .35, .35]) #, axisbg='y')
# ax2.scatter(tX, tY)
# ax2.set_axis_off()
# towrite = 'optimized' + ' 'optimized''
# ax.text(0.6, 0.3, towrite, transform=ax.transAxes)
# name = 'optimized' + '_errormean.pdf'
# plt.savefig(name)
#
#
# ### STD results
# # Values
# fig, ax = plt.subplots()
# ax.plot(s1[1:], std1[1:], label='rect')
# ax.plot(s2[1:], std2[1:], label='PC')
# ax.plot(s, np.ones(len(s))*base_std+0.01*base_std, 'k--', label='1% bounds')
# ax.plot(s, np.ones(len(s))*base_std-0.01*base_std, 'k--', label='1% bounds')
# ax.fill_between(s, std1_min, std1_max, facecolor='blue', alpha=0.2 )
# ax.fill_between(s, std2_min, std2_max, facecolor='green', alpha=0.2 )
# ax.set_xlabel('# directions')
# ax.set_ylabel('std')
# ax.set_title('averaged std vs # directions')
# ax.legend()
# ax2 = plt.axes([.50, .35, .35, .35]) #, axisbg='y')
# ax2.scatter(tX, tY)
# ax2.set_axis_off()
# towrite = 'optimized' + ' 'optimized''
# ax.text(0.6, 0.3, towrite, transform=ax.transAxes)
# name = 'optimized' + '_std.pdf'
# plt.savefig(name)
#
# # Error
# err1 = np.abs((std1-base_std))/base_std *100
# err2 = np.abs((std2-base_std))/base_std *100
#
# err1_max = np.abs((std1_max-base_std))/base_std *100
# err2_max = np.abs((std2_max-base_std))/base_std *100
#
# err1_min = np.abs((std1_min-base_std))/base_std *100
# err2_min = np.abs((std2_min-base_std))/base_std *100
#
# fig, ax = plt.subplots()
# ax.plot(s1[1:], err1[1:], label='rect')
# ax.plot(s2[1:], err2[1:], label='PC')
# #ax.fill_between(s, err1_min, err1_max, facecolor='blue', alpha=0.2 )
# #ax.fill_between(s, err2_min, err2_max, facecolor='green', alpha=0.2 )
# ax.set_xlabel('# directions')
# ax.set_ylabel('% error std')
# # ax.set_yscale('log')
# ax.set_title('averaged std vs # directions')
# ax.legend()
# ax2 = plt.axes([.50, .35, .35, .35]) #, axisbg='y')
# ax2.scatter(tX, tY)
# ax2.set_axis_off()
# towrite = 'optimized' + ' 'optimized''
# ax.text(0.6, 0.3, towrite, transform=ax.transAxes)
# name = 'optimized' + '_errorstd.pdf'
# plt.savefig(name)
#
plt.show()
