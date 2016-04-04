
import json
import numpy as np
import matplotlib.pyplot as plt

file1 = open('record_direction_uniformdakota.json', 'r') 
file2 = open('record_direction_uniformSA.json', 'r') 

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
ax.set_xlabel('# wind directions')
ax.set_ylabel('AEP')
ax.set_title('AEP vs # wind directions')
ax.legend()
plt.savefig('AEPdirectionConvergence.pdf')


plt.show()





