"""
This file contains an implementation of the top hat version of the Jensen model
presented in N. O. Jensen "A note on wind generator interaction" 1983

Created by: Jared J. Thomas
Date: 2015

"""

import numpy as np


def jensen_topHat(Uinf, rotorDiameter, axialInd, turbineX, turbineY, k, WindDirDeg, Cp, airDensity):
    """
    :param Uinf: float; inflow wind velocity
    :param rotorDiameter: numpy array; diameter of each turbine
    :param axialInd: numpy array; axial induction of each turbine
    :param turbineX: numpy array; X position of each turbine
    :param turbineY: numpy array; Y position of each turbine
    :param k: float; wake decay constant
    :param WindDirDeg: float; wind direction FROM in deg. CW from north (as in meteorological data)
    :return: numpy array; effective wind speed at each turbine
    """

    # adjust reference frame to wind direction
    turbineXw, turbineYw = referenceFrameConversion(turbineX, turbineY, WindDirDeg)

    # overlap calculations as per Jun et. al 2012
    OLRatio = wakeOverlapJensen(turbineXw, turbineYw, rotorDiameter, k)

    # Single wake effective windspeed calculations as per N.O. Jensen 1983
    UTilde = velocityDeficitJensen(Uinf, axialInd, turbineXw, rotorDiameter, k, OLRatio)

    # Wake Combination as per Jun et. al 2012 (I believe this is essentially what is done in WAsP)
    wt_velocity = wakeCombinationSumSquares(Uinf, OLRatio, UTilde)

    # simple power calculations
    wt_power = powerCalc(wt_velocity, Cp, rotorDiameter, airDensity)

    return wt_power, wt_velocity


def referenceFrameConversion(turbineX, turbineY, WindDirDeg):
    """ adjust turbine positions to wind direction reference frame """

    WindDirDeg = 270. - WindDirDeg
    if WindDirDeg < 0.:
        WindDirDeg += 360.
    WindDirRad = np.pi*WindDirDeg/180.0             # inflow wind direction in radians
    turbineXw = turbineX*np.cos(-WindDirRad)-turbineY*np.sin(-WindDirRad)
    turbineYw = turbineX*np.sin(-WindDirRad)+turbineY*np.cos(-WindDirRad)

    return turbineXw, turbineYw


def wakeOverlapJensen(turbineXw, turbineYw, rotorDiameter, k):
    """ overlap calculations as per Jun et. al 2012 and WAsP """

    nTurbines = np.size(turbineXw)
    Rr = rotorDiameter/2.0
    OLRatio = np.zeros([nTurbines, nTurbines])      # Overlap ratio

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

    return OLRatio


def velocityDeficitJensen(Uinf, axialInd, turbineXw, rotorDiameter, k, OLRatio):
    """ Single wake effective windspeed calculations as per N.O. Jensen 1983 """

    nTurbines = np.size(turbineXw)                  # number of turbines in the farm
    Rr = rotorDiameter/2.0                          # radii of the rotors
    UTilde = np.zeros([nTurbines, nTurbines])       # UTilde[i, j] of turb-i at turb-j

    for turbI in range(0, nTurbines):
        for turb in range(0, nTurbines):
            if turb != turbI and OLRatio[turb, turbI] != 0:
                dx = turbineXw[turbI] - turbineXw[turb]
                UTilde[turb, turbI] = Uinf*(1.0-2.0*axialInd[turb]*(Rr[turb]/(Rr[turb]+k*dx))**2)

    return UTilde


def wakeCombinationSumSquares(Uinf, OLRatio, UTilde):
    """ Wake Combination as per Jun et. al 2012 (I believe this is essentially what is done in WAsP) """

    nTurbines = np.size(OLRatio[0])
    wt_velocity = np.zeros(nTurbines)               # wt_velocity[i, j] of turb-i at turb-j

    for turbI in range(0, nTurbines):
        sumterm = 0.0
        for turb in range(0, nTurbines):
            if turb != turbI and OLRatio[turb, turbI] != 0:
                sumterm += (OLRatio[turb, turbI]*(1-UTilde[turb, turbI]/Uinf))**2
        wt_velocity[turbI] = Uinf*(1.0-np.sqrt(sumterm))

    return wt_velocity


def powerCalc(wt_velocity, Cp, rotorDiameter, airDensity):

    Ar = 0.25*np.pi*rotorDiameter**2                    # Rotor area of each turbine
    wt_power = 0.5*airDensity*Ar*Cp*wt_velocity**3      # power of each turbine

    return wt_power