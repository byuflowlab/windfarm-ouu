
import json
import numpy as np
import matplotlib.pyplot as plt

file1 = open('record_speed_uniform30dakota.json', 'r') 
file2 = open('record_speed_uniform30SA.json', 'r') 

recordd = json.load(file1)  
recordSA = json.load(file2)  

file1.close()
file2.close()

AEP = recordSA['AEP'][-1]
fig, ax = plt.subplots()
#ax.plot(recordd['samples'], np.array(recordd['AEP']), label='uniform PC')
#ax.plot(recordSA['samples'], np.array(recordSA['AEP']), label='uniform SA')
ax.plot(recordd['samples'][1:], np.array(recordd['AEP'][1:]), label='uniform PC')
ax.plot(recordSA['samples'][1:], np.array(recordSA['AEP'][1:]), label='uniform SA')
ax.plot(recordSA['samples'], np.ones(len(recordSA['samples']))*AEP+0.01*AEP, 'k--', label='1% bounds')
ax.plot(recordSA['samples'], np.ones(len(recordSA['samples']))*AEP-0.01*AEP, 'k--')
ax.set_xlabel('# wind speeds')
ax.set_ylabel('AEP')
ax.set_title('AEP vs # wind speeds')
ax.legend()
plt.savefig('AEPspeedConvergence.pdf')


std = recordSA['std_energy'][-1]
fig, ax = plt.subplots()
#ax.plot(recordd['samples'], np.array(recordd['std_energy']), label='uniform PC')
#ax.plot(recordSA['samples'], np.array(recordSA['std_energy']), label='uniform SA')
ax.plot(recordd['samples'][1:], np.array(recordd['std_energy'][1:]), label='uniform PC')
ax.plot(recordSA['samples'][1:], np.array(recordSA['std_energy'][1:]), label='uniform SA')
ax.plot(recordSA['samples'], np.ones(len(recordSA['samples']))*std+0.01*std, 'k--', label='1% bounds')
ax.plot(recordSA['samples'], np.ones(len(recordSA['samples']))*std-0.01*std, 'k--')
ax.set_xlabel('# wind speeds')
ax.set_ylabel('std')
ax.set_title('std energy vs # wind speeds')
ax.legend()
plt.savefig('STDspeedConvergence.pdf')


plt.show()





