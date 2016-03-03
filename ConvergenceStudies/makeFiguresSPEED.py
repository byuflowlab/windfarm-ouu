import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

if __name__=="__main__":
    filename = "WindSpeedConvergence.txt"
    file = open(filename)
    data = np.loadtxt(file)
    x = np.zeros(len(data))
    y = np.zeros(len(data))
    variance = np.zeros(len(data))
    error = np.zeros(len(data))
    
    f1 = open('recordPCspeed.json', 'r')
    f2 = open('recordPCdirection.json', 'r')
    record1 = json.load(f1)
    record2 = json.load(f2)
    f1.close()
    f2.close()

    AEPspeed = np.array(record1['AEP'])/1e6
    speed_points = np.array(record1['points'])
    AEPdirection = np.array(record2['AEP'])/1e6
    direction_points = np.array(record2['points'])

    # print type(AEPspeed)
    # print AEPspeed.size
    # print AEPspeed


    # converged = 1212031792.82

    for i in range(len(data)):
        x[i] = data[i][0]
        y[i] = data[i][1]/1e6
    print np.max(y)
    print np.max(AEPspeed)
    print y[len(y)-1]
    print AEPspeed[len(AEPspeed)-1]
    plt.plot(x[1:100], AEPspeed[1:], 'g', label='Polynomial Chaos')
    plt.plot(x[1:], y[1:], label='Trapezoidal Integration')
    plt.xlim([0,50])

    # plt.ylim([2000, 3000])
    plt.xlabel('Number of Wind Speeds')
    plt.ylabel('AEP (GWhrs)')
    # plt.title('AEP vs Number of Wind directions: 10x10 grid')

    converged = y[len(y)-1]
    
    top1percent = converged+0.01*converged
    bottom1percent = converged - 0.01*converged
    
    topy = (top1percent, top1percent)
    bottomy = (bottom1percent, bottom1percent)	
    topx = (0,150)
    plt.plot(topx, topy, 'r--', label='1% error')
    plt.plot(topx, bottomy, 'r--')		
    plt.legend(loc=1)
    font = {'family' : 'sans',
        'weight' : 'bold',
        'size'   : 15}
    matplotlib.rc('font', **font)
    plt.show()
 
