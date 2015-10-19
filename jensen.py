import numpy as np


def jensen_bak(X, Y, axialInd, rotorDiams, Cp_in, effU_in, wind, rho):
    """
        X is an ndarray of the x-coordinates of all turbines in the plant
        Y is an ndarray of the y-coordinates of all turbines in the plant
        R is an integer representing a consistent turbine radius
        Uinf is the free stream velocity
        The cosine-bell method [N.O.Jensen] is  employed
        Assume no wake deflection. Set wake center equal to hub center
        Assume effective wind speed at rotor hub is the effective windspeed for the entire rotor
    """

    D = np.average(rotorDiams)
    a = np.average(axialInd)
    Cp = np.average(Cp_in)
    effU = np.average(effU_in)
    alpha = 0.1                 # Entrainment constant per N. O. Jensen "A Note on Wind Generator Interaction".
    # alpha = 0.01              # Entrainment constant per Andersen et al "Comparison of Engineering Wake Models with CFD Simulations"
    # alpha = 0.084647            # Entrainment constant optimized to match FLORIS at 7*D
    # alpha = 0.399354
    # alpha = 0.216035
    R = D/2
    # Ct = (4/3)*(1-1/3)
    # Uin = 8
    Uin = effU
    Uinf = np.ones(X.size)*Uin
    # a = (1-sqrt(1-Ct))/2          # this assumes an optimal rotor. For a more general equation, see FLORIS paper
    # eta = 0.768
    # Cp = 4*a*((1-a)**2)*eta
    # rho = 1.1716
    n = np.size(X)
    Area = np.pi*pow(R, 2)

    Phi = (90-wind)*np.pi/180           # define inflow direction relative to North = 0 CCW

    # adjust coordinates to wind direction
    x = np.zeros(n)
    y = np.zeros(n)
    for i in range(0, n):
        # x[i], y[i] = np.dot(np.array([[cos(-Phi), sin(-Phi)], [-sin(-Phi), cos(-Phi)]]), np.array([X[i], Y[i]]))
        x[i] = X[i]*np.cos(-Phi)-Y[i]*np.sin(-Phi)
        y[i] = X[i]*np.sin(-Phi)+Y[i]*np.cos(-Phi)
    # print x, y

    Ueff = np.zeros(n)     # initialize effective wind speeds array

    # find wake overlap
    Overlap_adjust = WakeOverlap(x, y, R)

    # find upwind turbines and calc power for them
    Pow = np.zeros(n)
    front = np.zeros(n)
    for j in range(0, n):
        for i in range(0, n):
            if Overlap_adjust[i][j] != 0:
                front[j] = 0

           # print j #Pow[j]

    # calc power for downstream turbines

    xcopy = x-max(x)

    for j in range(0, n):

        q = np.argmin(xcopy)
        # print q, xcopy[q]
        if front[q] == 1:
           Ueff[q] = Uin

        elif front[q] == 0:
            G = 0
            for i in range(0, n):

                if x[q] - x[i] > 0:

                    z = abs(x[q] - x[i])

                    # V = Ueff[i]*(1-(2.0/3.0)*(R/(R+alpha*z))**2)
                    V = Uin*(1-Overlap_adjust[i][q]*(R/(R+alpha*z))**2)
                    print V
                    G = G + (1.0-V/Uin)**2

                # print 'G is:', G
            Ueff[q] = (1-np.sqrt(G))*Uin
            # print Ueff[q]
        Pow[q] = 0.5*rho*Area*Cp*Ueff[q]**3
            # print Pow[q]
        xcopy[q] = 1
    Pow_tot = np.sum(Pow)

    # return Pow_tot  # Ueff is an ndarray of effective windspeeds at each turbine in the plant
    return Pow


def jensen_tune(X, Y, rotorDiams, Cp_in, effU_in, wind, rho, alpha, boundAngle, axialInd):
    # last modified June 10 2015
    """
        X is an ndarray of the x-coordinates of all turbines in the plant
        Y is an ndarray of the y-coordinates of all turbines in the plant
        R is an integer representing a consistent turbine radius
        Uinf is the free stream velocity
        The cosine-bell method [N.O.Jensen] is  employed
        Assume no wake deflection. Set wake center equal to hub center
        Assume effective wind speed at rotor hub is the effective windspeed for the entire rotor
    """

    D = np.average(rotorDiams)
    Cp = np.average(Cp_in)
    effU = np.average(effU_in)
    # alpha = 0.1                 # Entrainment constant per N. O. Jensen "A Note on Wind Generator Interaction".
    # alpha = 0.01              # Entrainment constant per Andersen et al "Comparison of Engineering Wake Models with CFD Simulations"
    R = D/2.
    # Ct = (4/3)*(1-1/3)
    # Uin = 8
    Uin = effU
    Uinf = np.ones(X.size)*Uin
    # a = (1-sqrt(1-Ct))/2          # this assumes an optimal rotor. For a more general equation, see FLORIS paper
    # eta = 0.768
    # Cp = 4*a*((1-a)**2)*eta
    # rho = 1.1716
    n = np.size(X)
    Area = np.pi*pow(R, 2)

    Phi = (90-wind)*np.pi/180.           # define inflow direction relative to North = 0 CCW

    # adjust coordinates to wind direction
    x = np.zeros(n)
    y = np.zeros(n)
    for i in range(0, n):
        # x[i], y[i] = np.dot(np.array([[cos(-Phi), sin(-Phi)], [-sin(-Phi), cos(-Phi)]]), np.array([X[i], Y[i]]))
        x[i] = X[i]*np.cos(-Phi)-Y[i]*np.sin(-Phi)
        y[i] = X[i]*np.sin(-Phi)+Y[i]*np.cos(-Phi)
    # print x, y

    Ueff = np.zeros(n)     # initialize effective wind speeds array

    # find wake overlap
    Overlap_adjust = WakeOverlap_tune(x, y, R, boundAngle)

    # find upwind turbines and calc power for them
    Pow = np.zeros(n)
    front = np.zeros(n)
    for j in range(0, n):
        for i in range(0, n):
            if Overlap_adjust[i][j] != 0:
                front[j] = 0

           # print j #Pow[j]

    # calc power for downstream turbines

    xcopy = x-max(x)

    for j in range(0, n):

        q = np.argmin(xcopy)
        # print q, xcopy[q]
        if front[q] == 1:
           Ueff[q] = Uin

        elif front[q] == 0:
            G = 0
            for i in range(0, n):

                if x[q] - x[i] > 0:

                    z = x[q] - x[i]

                    # V = Ueff[i]*(1-(2.0/3.0)*Overlap_adjust[i][q]*(R/(R+alpha*z))**2)
                    V = Uin*(1-Overlap_adjust[i][q]*(R/(R+alpha*z))**2)
                    # print V
                    G += (1-V/Uin)**2

                # print 'G is:', G
            Ueff[q] = (1-np.sqrt(G))*Uin
            # print Ueff[q]
        Pow[q] = 0.5*rho*Area*Cp*Ueff[q]**3
            # print Pow[q]
        xcopy[q] = 1
    Pow_tot = np.sum(Pow)

    # return Pow_tot  # Ueff is an ndarray of effective windspeeds at each turbine in the plant
    return Pow, Ueff


def jensen(X, Y, rotorDiams, Cp, Uin, wind, rho, ke, boundAngle, axialInd):
    # last modified June 10 2015
    # X is an ndarray of the x-coordinates of all turbines in the plant
    # Y is an ndarray of the y-coordinates of all turbines in the plant
    # R is an ndarray or the rotor radii
    # ke is an ndarray of the axial induction factors
    # Uin is an ndarray of the free stream velocity
    # The cosine-bell method [N.O.Jensen] is  employed
    # Assume no wake deflection. Set wake center equal to hub center
    # Assume effective wind speed at rotor hub is the effective windspeed for the entire rotor

    # alpha = 0.1                 # Entrainment constant per N. O. Jensen "A Note on Wind Generator Interaction".
    # alpha = 0.01              # Entrainment constant per Andersen et al "Comparison of Engineering Wake Models with CFD Simulations"
    # Ct = (4/3)*(1-1/3)
    # Uin = 8
    # a = (1-sqrt(1-Ct))/2          # this assumes an optimal rotor. For a more general equation, see FLORIS paper
    # eta = 0.768
    # Cp = 4*a*((1-a)**2)*eta
    # rho = 1.1716
    effU = Uin
    n = np.size(X)
    rotorArea = np.pi*pow(rotorDiams, 2)/4.

    Phi = (90-wind)*np.pi/180.           # define inflow direction relative to North = 0 CCW

    # adjust coordinates to wind direction
    x = np.zeros(n)
    y = np.zeros(n)
    for i in range(0, n):
        # x[i], y[i] = np.dot(np.array([[cos(-Phi), sin(-Phi)], [-sin(-Phi), cos(-Phi)]]), np.array([X[i], Y[i]]))
        x[i] = X[i]*np.cos(-Phi)-Y[i]*np.sin(-Phi)
        y[i] = X[i]*np.sin(-Phi)+Y[i]*np.cos(-Phi)
    # print x, y

    # find wake overlap
    R = rotorDiams/2.
    wakeOverlap, cosine_adjust = WakeOverlap_areas(x, y, R, boundAngle)

    # find upwind turbines and calc power for them
    Pow = np.zeros(n)
    front = np.zeros(n)
    for j in range(0, n):
        if all(vals == 0 for vals in cosine_adjust[:, j]):
            front[j] = 1.
            Pow[j] = 0.5*rho*rotorArea[j]*Cp[j]*effU[j]**3

    # calc power for downstream turbines
    for j in range(0, n):
        if front[j] == 0:
            summation = 0
            for i in range(0, n):
                z = x[j] - x[i]
                c = (rotorDiams[i]/(rotorDiams[i]+2.*ke*z))*(wakeOverlap[j, i]/rotorArea[j])
                summation += (axialInd[i]*c)**2.

            effU[j] = 2*(1-2*np.sqrt(summation))

            Pow[j] = 0.5*rho*rotorArea[j]*Cp[j]*effU[j]**3.

    return Pow, effU


def jensen_TH(X, Y, rotorDiams, Cp_in, effU_in, wind, rho, alpha, boundAngle):
    # last modified June 10 2015
    # X is an ndarray of the x-coordinates of all turbines in the plant
    # Y is an ndarray of the y-coordinates of all turbines in the plant
    # R is an integer representing a consistent turbine radius
    # Uinf is the free stream velocity
    # The cosine-bell method [N.O.Jensen] is  employed
    # Assume no wake deflection. Set wake center equal to hub center
    # Assume effective wind speed at rotor hub is the effective windspeed for the entire rotor

    D = np.average(rotorDiams)
    Cp = np.average(Cp_in)
    effU = np.average(effU_in)
    # alpha = 0.1                 # Entrainment constant per N. O. Jensen "A Note on Wind Generator Interaction".
    # alpha = 0.01                # Entrainment constant per Andersen et al "Comparison of Engineering Wake Models with CFD Simulations"
    R = D/2.0
    # Ct = (4/3)*(1-1/3)
    # Uin = 8
    Uin = effU
    Uinf = np.ones(X.size)*Uin
    # a = (1-sqrt(1-Ct))/2          # this assumes an optimal rotor. For a more general equation, see FLORIS paper
    # eta = 0.768
    # Cp = 4*a*((1-a)**2)*eta
    # rho = 1.1716
    n = np.size(X)
    Area = np.pi*pow(R, 2)

    Phi = (90.-wind)*np.pi/180.           # define inflow direction relative to North = 0 CCW

    # adjust coordinates to wind direction
    x = np.zeros(n)
    y = np.zeros(n)
    for i in range(0, n):
        # x[i], y[i] = np.dot(np.array([[cos(-Phi), sin(-Phi)], [-sin(-Phi), cos(-Phi)]]), np.array([X[i], Y[i]]))
        x[i] = X[i]*np.cos(-Phi)-Y[i]*np.sin(-Phi)
        y[i] = X[i]*np.sin(-Phi)+Y[i]*np.cos(-Phi)
    # print x, y

    Ueff = np.zeros(n)     # initialize effective wind speeds array

    # find wake overlap
    Overlap_adjust = WakeOverlap_tune(x, y, R, boundAngle)

    # find upwind turbines and calc power for them
    Pow = np.zeros(n)
    front = np.zeros(n)
    for j in range(0, n):
        for i in range(0, n):
            if Overlap_adjust[i][j] != 0:
                front[j] = 0

           # print j #Pow[j]

    # calc power for downstream turbines

    xcopy = x-max(x)

    for j in range(0, n):

        q = np.argmin(xcopy)
        # print q, xcopy[q]
        if front[q] == 1:
           Ueff[q] = Uin

        elif front[q] == 0:
            G = 0
            for i in range(0, n):

                if x[q] - x[i] > 0:

                    z = x[q] - x[i]

                    # V = Ueff[i]*(1-(2.0/3.0)*Overlap_adjust[i][q]*(R/(R+alpha*z))**2)
                    V = Uin*(1-2*(1/3)*(R/(R+alpha*z))**2)
                    # print V
                    G += (1-V/Uin)**2

                # print 'G is:', G
            Ueff[q] = (1-np.sqrt(G))*Uin
            # print Ueff[q]
        Pow[q] = 0.5*rho*Area*Cp*Ueff[q]**3
            # print Pow[q]
        xcopy[q] = 1
    Pow_tot = np.sum(Pow)

    # return Pow_tot  # Ueff is an ndarray of effective windspeeds at each turbine in the plant
    return Pow, Ueff


def WakeOverlap(X, Y, R):

    n = np.size(X)

    # theta = np.zeros((n, n), dtype=np.float)        # angle of wake from fulcrum
    f_theta = np.zeros((n, n), dtype=np.float)      # smoothing values for smoothing

    for i in range(0, n):
        for j in range(0, n):
            if X[i] < X[j]:
                z = R/np.tan(0.34906585)
                # print z
                theta = np.arctan((Y[j] - Y[i]) / (X[j] - X[i] + z))
                # print 'theta =', theta
                if -0.34906585 < theta < 0.34906585:
                    f_theta[i][j] = (1 + np.cos(9*theta))/2
                    # print f_theta

    # print z
    # print f_theta
    return f_theta


def WakeOverlap_areas(X, Y, R, boundAngle):
    """
    :param X: x positions of turbines
    :param Y: y positions of turbines
    :param R: radius of turbine rotors (ndarray(nTurbines))
    :param boundAngle: bounding angle for Jensen wake definition (20 deg by NOJ 1983)
    :return: wakeOverlap = overlap ratio of each wake, f_theta = adjustment parameter for cosine Jensen
    """

    n = np.size(X)
    wakeOverlap = np.zeros((n, n))
    boundAngle *= np.pi/180.0
    #
    # for i in range(1, n):
    #     for j in range(1, n):
    #
    #         # # center distance between wake center i and rotor j
    #         z = abs(Y[j]-Y[i])
    #         zx = abs(X[j]-X[i])
    #         # print 'z = %s' % z
    #         # radius of rotor j
    #         Rr = R[j]
    #         Rw = zx*np.tan(boundAngle)
    #
    #         # calculate overlap areas
    #         if z >= (Rw + Rr):
    #             wakeOverlap[i, j] = 0.
    #         elif z < R[j]*1E-6:
    #             if Rw < Rr:
    #                 wakeOverlap[i, j] = np.pi*Rw**2.
    #             else:
    #                 wakeOverlap[i, j] = np.pi*Rr**2.
    #         else:
    #               # print (z**2+Rw**2-Rr**2)/(2*z*Rw)
    #               thetaw = np.real(np.arccos(complex((z**2+Rw**2-Rr**2)/(2*z*Rw))))
    #               # print thetaw
    #               thetar = np.real((np.arccos(complex((z**2+Rr**2-Rw**2)/(2*z*Rr)))))
    #               # print thetar
    #               Aw = thetaw*Rw**2 - Rw**2*np.sin(thetaw)*np.cos(thetaw)
    #               Ar = thetar*Rr**2 - Rr**2*np.sin(thetar)*np.cos(thetar)
    #               wakeOverlap[j, i] = Aw + Ar
    #
    for turb in range(0, n):
        for turbI in range(0, n):
            if X[turbI] > X[turb]:
                OVdYd = Y[turb]-Y[turbI]
                OVr = R[turbI]
                wakeDiameter = R[turb]+(X[turbI]-X[turb])*np.arctan(boundAngle)
                OVR = wakeDiameter/2.
                OVdYd = abs(OVdYd)
                if OVdYd != 0:
                    OVL = (-np.power(OVr, 2.0)+np.power(OVR, 2.0)+np.power(OVdYd, 2.0))/(2.0*OVdYd)
                else:
                    OVL = 0

                OVz = np.power(OVR, 2.0)-np.power(OVL, 2.0)

                if OVz > 0:
                    OVz = np.sqrt(OVz)
                else:
                    OVz = 0

                if OVdYd < (OVr+OVR):
                    if OVL < OVR and (OVdYd-OVL) < OVr:
                        wakeOverlap[turbI, turb] = np.power(OVR, 2.0)*np.arccos(OVL/OVR) + np.power(OVr, 2.0)*np.arccos((OVdYd-OVL)/OVr) - OVdYd*OVz
                    elif OVR > OVr:
                        wakeOverlap[turbI, turb] = np.pi*np.power(OVr, 2.0)
                    else:
                        wakeOverlap[turbI, turb] = np.pi*np.power(OVR, 2.0)
                else:
                    wakeOverlap[turbI, turb] = 0

    f_theta = np.zeros((n, n), dtype=np.float)      # smoothing values for smoothing
    q = np.pi/boundAngle                            # factor inside the cos term of the smooth Jensen (see Jensen1983 eq.(3))

    for i in range(0, n):
        for j in range(0, n):
            if X[i] < X[j]:
                z = R[i]/np.tan(boundAngle)               # distance from fulcrum to wake producing turbine
                theta = np.arctan((Y[j] - Y[i]) / (X[j] - X[i] + z))

                if -boundAngle < theta < boundAngle:
                    # For Coseine Jensen
                    # f_theta[i][j] = (1. + np.cos(q*theta))/2.
                    # For flat (top-hat) Jensen
                    f_theta[i][j] = 1.

    return wakeOverlap, f_theta


def WakeOverlap_tune(X, Y, R, boundAngle):

    n = np.size(X)
    boundAngle = boundAngle*np.pi/180.0
    # theta = np.zeros((n, n), dtype=np.float)      # angle of wake from fulcrum
    f_theta = np.zeros((n, n), dtype=np.float)      # smoothing values for smoothing
    q = np.pi/boundAngle                            # factor inside the cos term of the smooth Jensen (see Jensen1983 eq.(3))
    # print 'boundAngle = %s' %boundAngle, 'q = %s' %q
    for i in range(0, n):
        for j in range(0, n):
            if X[i] < X[j]:
                # z = R/tan(0.34906585)
                z = R/np.tan(boundAngle)               # distance from fulcrum to wake producing turbine
                # print z
                theta = np.arctan((Y[j] - Y[i]) / (X[j] - X[i] + z))
                # print 'theta =', theta

                if -boundAngle < theta < boundAngle:

                    f_theta[i][j] = (1. + np.cos(q*theta))/2.
                    # print f_theta

    # print z
    # print f_theta
    return f_theta


def WakeOverlap_TH(X, Y, R, boundAngle):

    n = np.size(X)
    boundAngle = boundAngle*np.pi/180.0
    # theta = np.zeros((n, n), dtype=np.float)      # angle of wake from fulcrum
    f_theta = np.zeros((n, n), dtype=np.float)      # smoothing values for smoothing
    q = np.pi/boundAngle
    # print 'boundAngle = %s' %boundAngle, 'q = %s' %q
    for i in range(0, n):
        for j in range(0, n):
            if X[i] < X[j]:
                z = R/np.tan(boundAngle)
                # print z
                theta = np.arctan((Y[j] - Y[i]) / (X[j] - X[i] + z))
                # print 'theta =', theta

                if -boundAngle < theta < boundAngle:

                    f_theta[i][j] = (1. + np.cos(q*theta))/2.
                    # print f_theta

    # print z
    # print f_theta
    return f_theta


if __name__ == "__main__":
    # data from example.py from FLORISSE downloaded from gitHub WISDEM project
    Y = np.array([1024.7, 1335.3, 1387.2, 1697.8, 2060.3, 1749.7])
    X = np.array([1164.7, 947.2,  1682.4, 1464.9, 1982.6, 2200.1])


    Pow = jensen(X, Y)

    print Pow