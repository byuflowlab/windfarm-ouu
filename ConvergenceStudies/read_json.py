import json
import numpy as np

f1 = open('recordPCspeed.json', 'r')
f2 = open('recordPCdirection.json', 'r')
record1 = json.load(f1)
record2 = json.load(f2)
f1.close()
f2.close()

AEPspeed = np.array(record1['AEP'])
speed_points = np.array(record1['points'])
AEPdirection = np.array(record2['AEP'])
direction_points = np.array(record2['points'])

print type(AEPspeed)
print AEPspeed.size
print AEPspeed


