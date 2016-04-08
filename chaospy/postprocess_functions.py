import matplotlib.pyplot as plt
import json
import numpy as np

f1 = open('record_weighted_power_uniform_direction.json', 'r')
f2 = open('record_weighted_power_windrose_direction.json', 'r')
f3 = open('record_weighted_power_uniform_speed.json', 'r')
f4 = open('record_weighted_power_weibull_speed.json', 'r')

r1 = json.load(f1)
r2 = json.load(f2)
r3 = json.load(f3)
r4 = json.load(f4)

f1.close()
f2.close()
f3.close()
f4.close()

w1 = np.array(r1['winddirections'])
p1 = np.array(r1['power'])
w2 = np.array(r2['winddirections'])
p2 = np.array(r2['power'])
w3 = np.array(r3['windspeeds'])
p3 = np.array(r3['power'])
w4 = np.array(r4['windspeeds'])
p4 = np.array(r4['power'])

step = 0.03
y1range = p1.max() - p1.min()
y1lim = [p1.min()-y1range*step, p1.max()+y1range*step]
y2range = p2.max() - p2.min()
y2lim = [p2.min()-y2range*step, p2.max()+y2range*step]
y3range = p3.max() - p3.min()
y3lim = [p3.min()-y3range*step, p3.max()+y3range*step]
y4range = p4.max() - p4.min()
y4lim = [p4.min()-y4range*step, p4.max()+y4range*step]


fig, ax = plt.subplots()
ax.plot(w1, p1)
ax.set_xlabel('wind direction')
ax.set_ylabel('power (kW)')
ax.set_title('Uniform weighted power')
ax.set_xlim([-10, 370])
ax.set_xticks(range(0,361,45))
ax.set_ylim(y1lim)
# plt.savefig('AEPdirectionConvergence.pdf')

fig, ax = plt.subplots()
ax.plot(w2, p2)
ax.set_xlabel('wind direction')
ax.set_ylabel('power (kW)')
ax.set_title('Windrose weighted power')
ax.set_xlim([-10, 370])
ax.set_xticks(range(0, 361, 45))
ax.set_ylim(y2lim)
# plt.savefig('AEPdirectionConvergence.pdf')

fig, ax = plt.subplots()
ax.plot(w3, p3)
ax.set_xlabel('wind speed (m/s)')
ax.set_ylabel('power (kW)')
ax.set_title('Uniform weighted power')
ax.set_xlim([-1, 31])
# ax.set_xticks(range(0,361,45))
ax.set_ylim(y3lim)
# plt.savefig('AEPdirectionConvergence.pdf')

fig, ax = plt.subplots()
ax.plot(w4, p4)
ax.set_xlabel('wind speed (m/s)')
ax.set_ylabel('power (kW)')
ax.set_title('Weibull weighted power')
ax.set_xlim([-1, 31])
# ax.set_xticks(range(0, 361, 45))
ax.set_ylim(y4lim)
# plt.savefig('AEPdirectionConvergence.pdf')

plt.show()
