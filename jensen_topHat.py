import numpy as np


def jensen_topHat(Uinf, D, axialInd, X, Y, k, WindDirDeg):
    """
    :param Uinf: float; inflow wind velocity
    :param D: numpy array; diameter of each turbine
    :param axialInd: numpy array; axial induction of each turbine
    :param X: numpy array; x position of each turbine
    :param Y: numpy array; y position of each turbine
    :param k: float; wake decay constant
    :param WindDirDeg: float; wind direction TO in deg.
    :return: numpy array; effective wind speed at each turbine
    """

    n = X.size
    R = D/2.
    UTilde = np.zeros([X.size, X.size])             # UTilde[i, j] of turb-i at turb-j
    Ueff = np.zeros(X.size)                         # Ueff[i, j] of turb-i at turb-j
    OLRatio = np.zeros([X.size, X.size])            # Overlap ratio
    WindDirRad = np.pi*WindDirDeg/180.0             # inflow wind direction in radians

    # adjust coordinates to wind direction reference frame
    x = np.zeros_like(X)
    y = np.zeros_like(Y)
    for i in range(0, n):
        x[i] = X[i]*np.cos(-WindDirRad)-Y[i]*np.sin(-WindDirRad)
        y[i] = X[i]*np.sin(-WindDirRad)+Y[i]*np.cos(-WindDirRad)

    # print x, y

    # overlap calculations as per Jun et. al 2012
    for turbI in range(0, n):
        for turb in range(0, n):
            dx = x[turbI] - x[turb]         # downwind turbine separation
            dy = abs(y[turbI] - y[turb])    # crosswind turbine separation
            if turb != turbI and dx > 0:
                Rw = R[turb] + k*dx
                OLArea = 0.0
                if dy <= Rw - R[turbI]:
                    OLArea = np.pi*R[turbI]**2

                if Rw - R[turbI] < dy < Rw + R[turbI]:

                    # print R[turb], dy, Rw

                    a = (R[turb]**2+dy**2-Rw**2)/(2.0*dy*R[turb])
                    b = (Rw**2+dy**2-R[turb]**2)/(2.0*dy*Rw)

                    alpha1 = 2.0*np.arccos(a)
                    alpha2 = 2.0*np.arccos(b)
                    # print alpha1, alpha2
                    OLArea = 0.5*(R[turbI]**2)*(alpha1 - np.sin(alpha1)) + 0.5*(Rw**2)*(alpha2 - np.sin(alpha2))

                Ar = np.pi*R[turbI]**2                      # rotor area of waked rotor
                OLRatio[turb, turbI] = OLArea/Ar            # ratio of area of wake-i and turb-j to rotor area of turb-j

    # Single wake effective windspeed calculations as per N.O. Jensen 1983
    for turbI in range(0, n):
        for turb in range(0, n):
            if turb != turbI and OLRatio[turb, turbI] != 0:
                dx = x[turbI] - x[turb]
                UTilde[turb, turbI] = Uinf*(1.0-2.0*axialInd[turb]*(R[turb]/(R[turb]+k*dx))**2)

    # Wake Combination as per Jun et. al 2012 (I believe this is essentially what is done in WAsP)
    for turbI in range(0, n):
        sumterm = 0.0
        for turb in range(0, n):
            if turb != turbI and OLRatio[turb, turbI] != 0:
                sumterm += (OLRatio[turb, turbI]*(1-UTilde[turb, turbI]/Uinf))**2
        Ueff[turbI] = Uinf*(1.0-np.sqrt(sumterm))

    return Ueff


def powerCalc(Ueff, Cp, D, airDensity):

    Ar = 0.25*np.pi*D**2                    # Rotor area

    power = 0.5*airDensity*Ar*Cp*Ueff**3

    return power