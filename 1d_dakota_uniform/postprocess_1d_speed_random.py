
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

f1 = open('record_raw.json', 'r')
f2 = open('record_smooth.json', 'r')

r1 = json.load(f1)
r2 = json.load(f2)

f1.close()
f2.close()

s1 = np.array(r1['samples'])
mu1 = np.array(r1['mean'])
std1 = np.array(r1['std'])
n1 = s1.size

s2 = np.array(r2['samples'])
#mu2 = np.array(r2['mean']) * 0.9918  # weighted by how much of probability is between 0 and 30
mu2 = np.array(r2['mean'])  # weighted by how much of probability is between 0 and 30
#std2 = np.array(r2['std']) * np.sqrt(0.9918)  # weighted by how much of probability is between 0 and 30, if you look at PC formula for std you see why it is sqrt.
std2 = np.array(r2['std'])   # weighted by how much of probability is between 0 and 30, if you look at PC formula for std you see why it is sqrt.
n2 = s2.size
n = n2

# Update the length of the rectangle array and get baseline values
base_mu = mu1[-1]
base_std = std1[-1]
s1 = s1[:n]
mu1 = mu1[:n]
std1 = std1[:n]


# Values
fig, ax = plt.subplots()
ax.plot(s1, mu1, label='rectangle')
ax.plot(s2, mu2, label='dakota')
ax.plot(s1, np.ones(len(s1))*base_mu+0.01*base_mu, 'k--', label='1% bounds')
ax.plot(s1, np.ones(len(s1))*base_mu-0.01*base_mu, 'k--', label='1% bounds')
ax.set_xlabel('# speeds')
ax.set_ylabel('mean')
ax.set_title('mean vs # speeds')
ax.legend()
#plt.savefig('AEPspeedConvergence.pdf')

# Error
err1 = np.abs((mu1-base_mu))/base_mu *100
err2 = np.abs((mu2-base_mu))/base_mu *100

fig, ax = plt.subplots()
ax.plot(s1, err1, label='rectangle')
ax.plot(s2, err2, label='dakota')
ax.set_xlabel('# speeds')
ax.set_ylabel('% error mean')
# ax.set_yscale('log')
ax.set_title('mean vs # speeds')
ax.legend()


# Moving average error
N = 5  # window size
roll_err1 = pd.rolling_mean(err1, N, center=True)
roll_err2 = pd.rolling_mean(err2, N, center=True)
fig, ax = plt.subplots()
ax.plot(s1, roll_err1, label='rectangle')
ax.plot(s2, roll_err2, label='dakota')
ax.set_xlabel('# speeds')
ax.set_ylabel('% error rolling mean')
# ax.set_yscale('log')
ax.set_title('error rolling mean N=5 vs # speeds')
ax.legend()

# Moving average error
N = 7  # window size
roll_err1 = pd.rolling_mean(err1, N, center=True)
roll_err2 = pd.rolling_mean(err2, N, center=True)
fig, ax = plt.subplots()
ax.plot(s1, roll_err1, label='rectangle')
ax.plot(s2, roll_err2, label='dakota')
ax.set_xlabel('# speeds')
ax.set_ylabel('% error rolling mean')
# ax.set_yscale('log')
ax.set_title('error rolling mean N=7 vs # speeds')
ax.legend()



### STD results
# Values

fig, ax = plt.subplots()
ax.plot(s1, std1, label='rectangle')
ax.plot(s2, std2, label='dakota')
ax.plot(s1, np.ones(len(s1))*base_std+0.01*base_std, 'k--', label='1% bounds')
ax.plot(s1, np.ones(len(s1))*base_std-0.01*base_std, 'k--', label='1% bounds')
ax.set_xlabel('# speeds')
ax.set_ylabel('std')
ax.set_title('std vs # speeds')
ax.legend()

# Error
err1 = np.abs((std1-base_std))/base_std *100
err2 = np.abs((std2-base_std))/base_std *100

fig, ax = plt.subplots()
ax.plot(s1, err1, label='rectangle')
ax.plot(s2, err2, label='dakota')
ax.set_xlabel('# speeds')
ax.set_ylabel('% error std')
# ax.set_yscale('log')
ax.set_title('std vs # speeds')
ax.legend()

# Moving average error
err1 = err1[1:]  # remove the first point of the error
err2 = err2[1:]
s1 = s1[1:]
s2 = s2[1:]

N = 5  # window size
roll_err1 = pd.rolling_mean(err1, N, center=True)
roll_err2 = pd.rolling_mean(err2, N, center=True)
fig, ax = plt.subplots()
ax.plot(s1, roll_err1, label='rectangle')
ax.plot(s2, roll_err2, label='dakota')
ax.set_xlabel('# speeds')
ax.set_ylabel('% error rolling std')
# ax.set_yscale('log')
ax.set_title('error rolling std N=5 vs # speeds')
ax.legend()

# Moving average error
N = 7  # window size
roll_err1 = pd.rolling_mean(err1, N, center=True)
roll_err2 = pd.rolling_mean(err2, N, center=True)
fig, ax = plt.subplots()
ax.plot(s1, roll_err1, label='rectangle')
ax.plot(s2, roll_err2, label='dakota')
ax.set_xlabel('# speeds')
ax.set_ylabel('% error rolling std')
# ax.set_yscale('log')
ax.set_title('error rolling std N=7 vs # speeds')
ax.legend()

windspeed = np.array(r1['windspeeds'])
power = np.array(r1['power'])
fig, ax = plt.subplots()
ax.plot(windspeed, power)
ax.set_xlabel('wind speed (m/s)')
ax.set_ylabel('power (kWhs)')


plt.show()





