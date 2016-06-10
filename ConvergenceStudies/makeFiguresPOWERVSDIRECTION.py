import matplotlib
import json
import numpy as np
import matplotlib.pyplot as plt

if __name__=="__main__":
    filename = "powerVSdirectionAMALIA.txt"
    file = open(filename)
    data = np.loadtxt(file)
    x = np.zeros(len(data))
    y = np.zeros(len(data))
    variance = np.zeros(len(data))
    error = np.zeros(len(data))
    """
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
    
    # print type(AEPspeed)
    # print AEPspeed.size
    print "Speed Points: ", speed_points
    print "AEPspeed: ", AEPspeed
    """

    # converged = 1212031792.82

    for i in range(len(data)):
        x[i] = data[i][0]
        y[i] = data[i][1]/1000

    font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 15}
    matplotlib.rc('font', **font)

    # plt.plot(speed_points, AEPspeed, 'g', label='Santiago')
    plt.plot(x, y)
    plt.axis([0,360,20,100])
    plt.xlabel('Wind Direction (degrees)')
    plt.ylabel('Power (MW)')
    # plt.title('AEP vs Number of Wind directions: 10x10 grid')
    """
    converged = y[len(y)-1]
    
    top1percent = converged+0.01*converged
    bottom1percent = converged - 0.01*converged
    
    topy = (top1percent, top1percent)
    bottomy = (bottom1percent, bottom1percent)	
    topx = (0,150)
    """
    


    plt.show()

