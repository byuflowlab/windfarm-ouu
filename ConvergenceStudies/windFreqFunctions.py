import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import scipy.integrate as integrate
from math import exp


def wind_frequency_funcion(windrose):
    input_file = open(windrose+".txt")
    wind_data = np.loadtxt(input_file)
    
    probability = wind_data[:, 2]
    length_data = np.linspace(0,len(wind_data)+0.01,len(probability))
    f = interp1d(length_data, probability)
    return f, len(probability)


def wind_frequency_cubic():
    input_file = open("amalia_windrose_8.txt")
    wind_data = np.loadtxt(input_file)

    length_data = np.linspace(0,72.01,len(wind_data))
    f = interp1d(length_data, wind_data, kind='cubic')
    return f


def frequ(windrose, bins):
    f, size = wind_frequency_funcion(windrose)
    L = size/1.
    print "L: ", L
    bin_size = L/bins
    dx = 0.01
    x1 = 0.
    x2 = x1+dx
    bin_location = bin_size
    frequency = np.zeros(bins)
    for i in range(0, bins):
        while x1 <= bin_location:
            dfrequency = dx*(f(x1)+f(x2))/2
            frequency[i] += dfrequency
            x1 = x2
            x2 += dx
        bin_location += bin_size
    total = np.sum(frequency)
    frequency = frequency/total
    return frequency

def npfrequ(bins):
    f = wind_frequency_funcion()
    dx = 0.001
    bin_size = 72./bins
    bin_size_int = 72/bins
    bin_start = 0
    frequency = np.zeros(bins)
    for i in range(0, bins):
        x = np.linspace(bin_start, bin_start+bin_size, num = bin_size_int/dx)
        frequency[i] = np.trapz(f(x), x, dx=dx)
        bin_start += bin_size
    return frequency
	

def weibull_prob(x):
    a = 1.8
    avg = 8.
    lamda = avg/(((a-1)/a)**(1/a))
    return a/lamda*(x/lamda)**(a-1)*exp(-(x/lamda)**a)


def speed_frequ(speeds):
    x = speeds
    size = 30./(speeds)
    dx = 0.01
    x1 = 0.
    x2 = x1+dx
    location = size
    frequency = np.zeros(speeds)
    for i in range(0, speeds):
        while x1 <= location:
            dfrequency = dx*(weibull_prob(x1)+weibull_prob(x2))/2
            frequency[i] += dfrequency
            x1 = x2
            x2 += dx
        location += size
    return frequency


if __name__ == '__main__':

    # bins1 = 5
    # x1 = np.linspace(0,72-72/bins1, bins1)
    # bins2 = 15
    # x2 = np.linspace(0,72-72/bins2, bins2)
    # bins3 = 30
    # x3 = np.linspace(0,72-72/bins3, bins3)
    # bins4 = 50
    # x4 = np.linspace(0,72-72/bins4, bins4)
    # bins5 = 72
    # x5 = np.linspace(0,72-72/bins5, bins5)

    # f = wind_frequency_funcion()

    # print frequ(bins3)
    # print npfrequ(bins3)
    # print np.sum(npfrequ(bins3))


    # print np.sum(frequ(bins3))
    #
    # dx=72./bins3
    # f1 = f(x3)
    # print np.trapz(f1, dx=dx)
    frequencies = frequ('windrose_lissett_single_average_speed', 5)
    print frequencies
    print np.sum(frequencies)
    """f = wind_frequency_cubic()
    x = np.linspace(0.,72.,1000)
    x_wind = np.linspace(0.,72,72)

    input_file = open("amalia_windrose_8.txt")
    wind_data = np.loadtxt(input_file)

    wind_freq = f(x)
    for i in range(0,len(wind_freq)):
        if wind_freq[i]<np.min(wind_data):
            wind_freq[i] = np.min(wind_data)
    print np.min(wind_freq)

    plt.figure(1)
    plt.plot(x,wind_freq)
    plt.scatter(x_wind, wind_data)
    plt.title('Frequency Function')
    plt.xlabel('Wind Direction')
    plt.ylabel('Frequency')
    plt.show()"""

    """f = speed_frequ(1000)
    print f
    print np.sum(f)


    
    x = np.linspace(0, 30, 1000)
    y = np.zeros(len(x))
    for i in range(len(x)):
        y[i] = weibull_prob(x[i])
    plt.plot(x,y)
    plt.xlabel('Wind Speed (m/s)')
    plt.ylabel('Probability')
    plt.show()"""



