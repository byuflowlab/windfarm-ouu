
import json
import numpy as np
import matplotlib.pyplot as plt

file1 = open('record_speed_weibull_weibulldakota.json', 'r') 
file2 = open('record_speed_weibull_uniform30dakota.json', 'r') 
file3 = open('record_speed_weibull_uniform100dakota.json', 'r') 
file4 = open('record_speed_weibull_uniform30SA.json', 'r') 
file5 = open('record_speed_weibull_uniform100SA.json', 'r') 

recordw = json.load(file1) 
recordu30d = json.load(file2)  
recordu100d = json.load(file3)  
recordu30SA = json.load(file4)  
recordu100SA = json.load(file5)  

file1.close()
file2.close()
file3.close()
file4.close()
file5.close()

AEP = recordu100SA['AEP'][-1]
fig, ax = plt.subplots()
ax.plot(recordw['samples'], np.array(recordw['AEP']), label='weibull-weibull')
#ax.plot(recordu30d['samples'], np.array(recordu30d['AEP']), label='weibull-uniform30')
ax.plot(recordu100d['samples'], np.array(recordu100d['AEP']), label='weibull-uniform100')
#ax.plot(recordu30SA['samples'], np.array(recordu30SA['AEP']), label='weibull-uniform30SA')
ax.plot(recordu100SA['samples'], np.array(recordu100SA['AEP']), label='weibull-uniform100SA')
ax.plot(recordu100SA['samples'], np.ones(len(recordu100SA['samples']))*AEP+0.01*AEP, 'k--', label='1% bounds')
ax.plot(recordu100SA['samples'], np.ones(len(recordu100SA['samples']))*AEP-0.01*AEP, 'k--')
ax.set_xlabel('# wind speeds')
ax.set_ylabel('AEP')
ax.set_title('AEP vs # wind speeds')
ax.legend()
plt.savefig('AEPspeedConvergence.pdf')


plt.show()





