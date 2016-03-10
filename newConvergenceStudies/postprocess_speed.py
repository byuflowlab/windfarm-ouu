
import json
import numpy as np
import matplotlib.pyplot as plt

file1 = open('recordSAspeedw.json', 'r')  # w for weighted
file2 = open('recordPCspeedw.json', 'r')

recordSAw = json.load(file1)    # SA simple average
recordPCw = json.load(file2)   # PC polynomial chaos

fig, ax = plt.subplots()
#ax.plot(recordSAw['samples'], np.array(recordSAw['AEP'])/recordSAw['AEP'][-1], label='SA')
#ax.plot(recordPCw['samples'], np.array(recordPCw['AEP'])/recordPCw['AEP'][-1], label='PC')
ax.plot(recordSAw['samples'], np.array(recordSAw['AEP']), label='SA')
ax.plot(recordPCw['samples'], np.array(recordPCw['AEP']), label='PC')
ax.set_xlabel('# wind speeds')
ax.set_ylabel('AEP')
ax.set_title('AEP vs # wind speeds')
ax.legend()
plt.savefig('AEPspeedConvergence.pdf')


plt.show()





