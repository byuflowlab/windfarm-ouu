
import json
import numpy as np
import matplotlib.pyplot as plt

f1 = open('record_direction_numerical_uniformdakota.json', 'r') 
f2 = open('record_direction_numerical_uniformSA.json', 'r') 
f3 = open('record_direction_numerical_numericaldakota.json', 'r') 
f4 = open('record_direction_numerical_MC.json', 'r')
#f4 = open('record_direction_numerical_numericaldakotalong.json', 'r') 
f5 = open('record_direction_chaospy.json', 'r')

r1 = json.load(f1)  
r2 = json.load(f2)  
r3 = json.load(f3)  
r4 = json.load(f4)
r5 = json.load(f5)

f1.close()
f2.close()
f3.close()
f4.close()
f5.close()

s1 = np.array(r1['samples'])
mu1 = np.array(r1['AEP'])
std1 = np.array(r1['std_energy'])

s2 = np.array(r2['samples'])
mu2 = np.array(r2['AEP'])
std2 = np.array(r2['std_energy'])

s3 = np.array(r3['samples'])
mu3 = np.array(r3['AEP'])
std3 = np.array(r3['std_energy'])

s4 = np.array(r4['samples'])
mu4 = np.array(r4['mean'])
std4 = np.array(r4['std'])

s5 = np.array(r5['samples'])
mu5 = np.array(r5['mean'])
std5 = np.array(r5['std'])

mu_ref = mu2[-1]
n = 5
fig, ax = plt.subplots()
ax.plot(s1[n:], mu1[n:], label='numerical_uniform_PC')
ax.plot(s2[n:], mu2[n:], label='numerical_rectangle_rule')
ax.plot(s3[n:], mu3[n:], label='numerical_numerical_PC')
ax.plot(s4[n:], mu4[n:], label='numerical_MC')
ax.plot(s5[n:], mu5[n:], label='chaospy')
ax.plot(s2, np.ones(len(s2))*mu_ref+0.01*mu_ref, 'k--', label='1% bounds')
ax.plot(s2, np.ones(len(s2))*mu_ref-0.01*mu_ref, 'k--')
ax.set_xlabel('# wind directions')
ax.set_ylabel('AEP')
#ax.set_ylim([mu_ref-0.1*mu_ref, mu_ref+0.1*mu_ref])
ax.set_title('AEP vs # wind directions')
ax.legend()
plt.savefig('AEPdirectionConvergence.pdf')

std_ref = std2[-1]
n = 5
fig, ax = plt.subplots()
ax.plot(s1[n:], std1[n:], label='numerical_uniform_PC')
ax.plot(s2[n:], std2[n:], label='numerical_rectangle_rule')
ax.plot(s3[n:], std3[n:], label='numerical_numerical_PC')
ax.plot(s4[n:], std4[n:], label='numerical_MC')
ax.plot(s5[n:], std5[n:], label='chaospy')
ax.plot(s2, np.ones(len(s2))*std_ref+0.01*std_ref, 'k--', label='1% bounds')
ax.plot(s2, np.ones(len(s2))*std_ref-0.01*std_ref, 'k--')
ax.set_xlabel('# wind directions')
ax.set_ylabel('Stanford deviation of energy')
#ax.set_ylim([std_ref-0.1*std_ref, std_ref+0.1*std_ref])
ax.set_title('STD vs # wind directions')
ax.legend()
plt.savefig('AEPdirectionConvergence.pdf')

plt.show()





