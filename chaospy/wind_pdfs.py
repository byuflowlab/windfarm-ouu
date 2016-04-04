import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def wind_direction_pdf(xnew):
    input_file = open("amalia_windrose_8.txt")
    wind_data = np.loadtxt(input_file)
    step = 360/len(wind_data)
    wind_data = np.append(wind_data, wind_data[0])  # Include the value at 360, which is the same as 0.
    wind_data = wind_data/step  # normalize for the [0, 360] range.
    x = np.array(range(0,360+1,step))
    f = interp1d(x, wind_data)
    # fc = interp1d(x, wind_data, kind='cubic')
    y = f(xnew)
    return y


def wind_speed_pdfweibull(x):
    a = 1.8
    avg = 8.
    lamda = avg/(((a-1)/a)**(1/a))  # is the beta in Dakota
    return a/lamda*(x/lamda)**(a-1)*np.exp(-(x/lamda)**a)


def wind_speed_pdfweibull_dakota(x):
    a = 1.8
    b = 12.552983
    f = a/b * (x/b)**(a-1) * np.exp(-(x/b)**a)
    return f


if __name__ == '__main__':

    x = np.linspace(0, 360, 361)
    y = wind_direction_pdf(x)
    dx = x[1]-x[0]
    print 'integral pdf = ', sum(y)*dx  # This should integrate to 1.
    plt.figure()
    plt.plot(x, y)


    n = 101
    x = np.linspace(0,30,n)
    y = wind_speed_pdfweibull(x)
    dx = x[1]-x[0]
    print 'integral pdf = ', sum(y)*dx  # This should integrate to 1.
    # y1 = wind_speed_pdfweibull_dakota(x)
    plt.figure()
    # plt.plot(x, y, x, y1)
    plt.plot(x, y)



    plt.show()
