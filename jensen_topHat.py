"""
This file contains an implementation of the top hat Jensen model

Created by: Jared J. Thomas
Date: 2015

"""

import numpy as np


def jensen_topHat(Uinf, rotorDiameter, axialInd, turbineX, turbineY, k, WindDirDeg):
    """
    :param Uinf: float; inflow wind velocity
    :param rotorDiameter: numpy array; diameter of each turbine
    :param axialInd: numpy array; axial induction of each turbine
    :param turbineX: numpy array; X position of each turbine
    :param turbineY: numpy array; Y position of each turbine
    :param k: float; wake decay constant
    :param WindDirDeg: float; wind direction TO in deg.
    :return: numpy array; effective wind speed at each turbine
    """

    nTurbines = turbineX.size
    Rr = rotorDiameter/2.
    UTilde = np.zeros([nTurbines, nTurbines])             # UTilde[i, j] of turb-i at turb-j
    wt_velocity = np.zeros(nTurbines)                         # wt_velocity[i, j] of turb-i at turb-j
    OLRatio = np.zeros([nTurbines, nTurbines])            # Overlap ratio
    WindDirRad = np.pi*WindDirDeg/180.0             # inflow wind direction in radians

    # adjust coordinates to wind direction reference frame
    turbineXw = turbineX*np.cos(-WindDirRad)-turbineY*np.sin(-WindDirRad)
    turbineYw = turbineX*np.sin(-WindDirRad)+turbineY*np.cos(-WindDirRad)

    # print turbineXw, turbineYw

    # overlap calculations as per Jun et. al 2012
    for turbI in range(0, nTurbines):
        for turb in range(0, nTurbines):
            dx = turbineXw[turbI] - turbineXw[turb]         # downwind turbine separation
            dy = abs(turbineYw[turbI] - turbineYw[turb])    # crosswind turbine separation
            if turb != turbI and dx > 0:
                Rw = Rr[turb] + k*dx
                OLArea = 0.0
                if dy <= Rw - Rr[turbI]:
                    OLArea = np.pi*Rr[turbI]**2

                if Rw - Rr[turbI] < dy < Rw + Rr[turbI]:

                    # print Rr[turb], dy, Rw

                    a = (Rr[turb]**2+dy**2-Rw**2)/(2.0*dy*Rr[turb])
                    b = (Rw**2+dy**2-Rr[turb]**2)/(2.0*dy*Rw)

                    alpha1 = 2.0*np.arccos(a)
                    alpha2 = 2.0*np.arccos(b)
                    # print alpha1, alpha2
                    OLArea = 0.5*(Rr[turbI]**2)*(alpha1 - np.sin(alpha1)) + 0.5*(Rw**2)*(alpha2 - np.sin(alpha2))

                Ar = np.pi*Rr[turbI]**2                      # rotor area of waked rotor
                OLRatio[turb, turbI] = OLArea/Ar            # ratio of area of wake-i and turb-j to rotor area of turb-j

    # Single wake effective windspeed calculations as per N.O. Jensen 1983
    for turbI in range(0, nTurbines):
        for turb in range(0, nTurbines):
            if turb != turbI and OLRatio[turb, turbI] != 0:
                dx = turbineXw[turbI] - turbineXw[turb]
                UTilde[turb, turbI] = Uinf*(1.0-2.0*axialInd[turb]*(Rr[turb]/(Rr[turb]+k*dx))**2)

    # Wake Combination as per Jun et. al 2012 (I believe this is essentially what is done in WAsP)
    for turbI in range(0, nTurbines):
        sumterm = 0.0
        for turb in range(0, nTurbines):
            if turb != turbI and OLRatio[turb, turbI] != 0:
                sumterm += (OLRatio[turb, turbI]*(1-UTilde[turb, turbI]/Uinf))**2
        wt_velocity[turbI] = Uinf*(1.0-np.sqrt(sumterm))

    return wt_velocity


def powerCalc(wt_velocity, Cp, rotorDiameter, airDensity):

    Ar = 0.25*np.pi*rotorDiameter**2                    # Rotor area

    power = 0.5*airDensity*Ar*Cp*wt_velocity**3

    return power