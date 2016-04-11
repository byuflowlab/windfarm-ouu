
import json
import numpy as np
import matplotlib.pyplot as plt

file1 = open('recordSAdirectionw.json', 'r')  # w for weighted
file2 = open('recordPCdirectionw.json', 'r')

recordSAw = json.load(file1)    # SA simple average
recordPCw = json.load(file2)   # PC polynomial chaos

fig, ax = plt.subplots()
ax.plot(recordSAw['samples'], np.array(recordSAw['AEP'])/recordSAw['AEP'][-1], label='SA')
ax.plot(recordPCw['samples'], np.array(recordPCw['AEP'])/recordPCw['AEP'][-1], label='PC')
ax.set_xlabel('# directions')
ax.set_ylabel('AEP')
ax.set_title('AEP vs # directions')
ax.legend()
plt.savefig('AEPdirectionsConvergence.pdf')


plt.show()





