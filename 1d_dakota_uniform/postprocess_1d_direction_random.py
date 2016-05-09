
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Used 5 starting points for the direction
f1 = open('record_direction_random_rectanglem2.json', 'r')
f2 = open('record_direction_random_rectanglem1.json', 'r')
f3 = open('record_direction_random_rectangle0.json', 'r')
f4 = open('record_direction_random_rectangle1.json', 'r')
f5 = open('record_direction_random_rectangle2.json', 'r')

# f2 = open('record_direction_random_dakota.json', 'r')

r1 = json.load(f1)
r2 = json.load(f2)
r3 = json.load(f3)
r4 = json.load(f4)
r5 = json.load(f5)

f1.close()
f2.close()
f1.close()
f2.close()
f1.close()

s1 = np.array(r1['samples'])
mu1 = np.array(r1['mean'])
std1 = np.array(r1['std'])
n1 = s1.size

s2 = np.array(r2['samples'])
mu2 = np.array(r2['mean'])
std2 = np.array(r2['std'])
n2 = s2.size

s3 = np.array(r3['samples'])
mu3 = np.array(r3['mean'])
std3 = np.array(r3['std'])
n3 = s3.size

s4 = np.array(r4['samples'])
mu4 = np.array(r4['mean'])
std4 = np.array(r4['std'])
n4 = s4.size

s5 = np.array(r5['samples'])
mu5 = np.array(r5['mean'])
std5 = np.array(r5['std'])
n5 = s5.size


# Update the length of the rectangle array and get baseline values
s = np.average([s1, s2, s3, s4, s5], axis=0)
mu = np.average([mu1, mu2, mu3, mu4, mu5], axis=0)
std = np.average([std1, std2, std3, std4, std5], axis=0)
base_mu = mu[-1]
base_std = std[-1]

# Only plot up to 50
n = 100
s1 = s1[:n]
mu1 = mu1[:n]
std1 = std1[:n]

s2 = s2[:n]
mu2 = mu2[:n]
std2 = std2[:n]

s3 = s3[:n]
mu3 = mu3[:n]
std3 = std3[:n]

s4 = s4[:n]
mu4 = mu4[:n]
std4 = std4[:n]

s5 = s5[:n]
mu5 = mu5[:n]
std5 = std5[:n]

s = s[:n]
mu = mu[:n]
std = std[:n]


# Mean results

# Values
fig, ax = plt.subplots()
ax.plot(s1, mu1, label='rectanglem2')
ax.plot(s2, mu2, label='rectanglem1')
ax.plot(s3, mu3, label='rectangle0')
ax.plot(s4, mu4, label='rectangle1')
ax.plot(s5, mu5, label='rectangle2')
ax.plot(s, mu, label='average', linewidth=2, color='k')
ax.plot(s, np.ones(len(s))*base_mu+0.01*base_mu, 'k--', label='1% bounds')
ax.plot(s, np.ones(len(s))*base_mu-0.01*base_mu, 'k--', label='1% bounds')
ax.set_xlabel('# directions')
ax.set_ylabel('mean')
ax.set_title('mean vs # directions')
ax.legend()
#plt.savefig('AEPspeedConvergence.pdf')

# Error
err1 = np.abs((mu1-base_mu))/base_mu *100
err2 = np.abs((mu2-base_mu))/base_mu *100
err3 = np.abs((mu3-base_mu))/base_mu *100
err4 = np.abs((mu4-base_mu))/base_mu *100
err5 = np.abs((mu5-base_mu))/base_mu *100
err = np.abs((mu-base_mu))/base_mu *100

fig, ax = plt.subplots()
ax.plot(s1, err1, label='rectanglem2')
ax.plot(s2, err2, label='rectanglem1')
ax.plot(s3, err3, label='rectangle0')
ax.plot(s4, err4, label='rectangle1')
ax.plot(s5, err5, label='rectangle2')
ax.plot(s, err, label='average', linewidth=2, color='k')
ax.set_xlabel('# directions')
ax.set_ylabel('% error mean')
# ax.set_yscale('log')
ax.set_title('mean vs # directions')
ax.legend()


# Moving average error
# N = 5  # window size
# roll_err1 = pd.rolling_mean(err1, N, center=True)
# roll_err2 = pd.rolling_mean(err2, N, center=True)
# fig, ax = plt.subplots()
# ax.plot(s1, roll_err1, label='rectangle')
# ax.plot(s2, roll_err2, label='dakota')
# ax.set_xlabel('# speeds')
# ax.set_ylabel('% error rolling mean')
# # ax.set_yscale('log')
# ax.set_title('error rolling mean N=5 vs # speeds')
# ax.legend()
#
# # Moving average error
# N = 7  # window size
# roll_err1 = pd.rolling_mean(err1, N, center=True)
# roll_err2 = pd.rolling_mean(err2, N, center=True)
# fig, ax = plt.subplots()
# ax.plot(s1, roll_err1, label='rectangle')
# ax.plot(s2, roll_err2, label='dakota')
# ax.set_xlabel('# speeds')
# ax.set_ylabel('% error rolling mean')
# # ax.set_yscale('log')
# ax.set_title('error rolling mean N=7 vs # speeds')
# ax.legend()



### STD results
# Values
fig, ax = plt.subplots()
ax.plot(s1, std1, label='rectanglem2')
ax.plot(s2, std2, label='rectanglem1')
ax.plot(s3, std3, label='rectangle0')
ax.plot(s4, std4, label='rectangle1')
ax.plot(s5, std5, label='rectangle2')
ax.plot(s, std, label='average', linewidth=2, color='k')
ax.plot(s, np.ones(len(s))*base_std+0.01*base_std, 'k--', label='1% bounds')
ax.plot(s, np.ones(len(s))*base_std-0.01*base_std, 'k--', label='1% bounds')
ax.set_xlabel('# directions')
ax.set_ylabel('std')
ax.set_title('std vs # directions')
ax.legend()
#plt.savefig('AEPspeedConvergence.pdf')

# Error
err1 = np.abs((std1-base_std))/base_std *100
err2 = np.abs((std2-base_std))/base_std *100
err3 = np.abs((std3-base_std))/base_std *100
err4 = np.abs((std4-base_std))/base_std *100
err5 = np.abs((std5-base_std))/base_std *100
err = np.abs((std-base_std))/base_std *100

fig, ax = plt.subplots()
ax.plot(s1, err1, label='rectanglem2')
ax.plot(s2, err2, label='rectanglem1')
ax.plot(s3, err3, label='rectangle0')
ax.plot(s4, err4, label='rectangle1')
ax.plot(s5, err5, label='rectangle2')
ax.plot(s, err, label='average', linewidth=2, color='k')
ax.set_xlabel('# directions')
ax.set_ylabel('% error std')
# ax.set_yscale('log')
ax.set_title('std vs # directions')
ax.legend()


# windspeed = np.array(r1['windspeeds'])
# power = np.array(r1['power'])
# fig, ax = plt.subplots()
# ax.plot(windspeed, power)
# ax.set_xlabel('wind speed (m/s)')
# ax.set_ylabel('power (kWhs)')


plt.show()





