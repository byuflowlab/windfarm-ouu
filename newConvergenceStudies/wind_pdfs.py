import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def wind_direction_pdf(xnew):
    input_file = open("amalia_windrose_8.txt")
    wind_data = np.loadtxt(input_file)
    step = 360/len(wind_data)
    wind_data = np.append(wind_data, wind_data[0])  # Include the value at 360, which is the same as 0.
    x = np.array(range(0,360+1,step))
    f = interp1d(x, wind_data)
    # fc = interp1d(x, wind_data, kind='cubic')
    y = f(xnew)
    return y


def wind_speed_pdfweibull(x):
    a = 1.8
    avg = 8.
    lamda = avg/(((a-1)/a)**(1/a))
    return a/lamda*(x/lamda)**(a-1)*np.exp(-(x/lamda)**a)


if __name__ == '__main__':


    f = wind_direction_pdf()
    xnew = np.linspace(0,360,361)
    plt.figure()
    plt.plot(xnew, f(xnew))


    x = np.linspace(0,30)
    y = wind_speed_pdfweibull(x)
    plt.figure()
    plt.plot(x,y)


    plt.show()
