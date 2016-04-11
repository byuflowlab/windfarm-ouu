
import json
import numpy as np
import matplotlib.pyplot as plt

f1 = open('record_2d.json', 'r') 
f2 = open('record_2dsparse.json', 'r') 

r1 = json.load(f1)
r2 = json.load(f2)

f1.close()
f2.close()

s1 = np.array(r1['samples'])
mu1 = np.array(r1['mean'])
std1 = np.array(r1['std'])

s2 = np.array(r2['samples'])
mu2 = np.array(r2['mean'])
std2 = np.array(r2['std'])

base = mu1[-1]
fig, ax = plt.subplots()
ax.plot(s1, mu1, label='uniform-uniform')
ax.plot(s1, np.ones(len(s1))*base+0.01*base, 'k--', label='1% bounds')
ax.plot(s1, np.ones(len(s1))*base-0.01*base, 'k--', label='1% bounds')
ax.set_xlabel('# points per direction')
ax.set_ylabel('mean')
ax.set_title('mean vs # points per direction')
ax.legend()
#plt.savefig('AEPspeedConvergence.pdf')

base = mu2[-1]
fig, ax = plt.subplots()
ax.plot(s2, mu2, label='uniform-uniform')
ax.plot(s2, np.ones(len(s2))*base+0.01*base, 'k--', label='1% bounds')
ax.plot(s2, np.ones(len(s2))*base-0.01*base, 'k--', label='1% bounds')
ax.set_xlabel('# sparse grid level')
ax.set_ylabel('mean')
ax.set_title('mean vs # sparse grid level')
ax.legend()

plt.show()





