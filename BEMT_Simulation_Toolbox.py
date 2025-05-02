import numpy as np
import csv
from matplotlib import pyplot as plt
from scipy.interpolate import make_interp_spline

########################################################################################################################
############################ Constants and Setup #######################################################################
########################################################################################################################

rpm2radps = 0.10472
deg2rad = np.pi/180
rad2deg = 180/np.pi

# Set Constants
increments = 117
rho = 1.25  # Density of Air
R = 117  # Blade Length
Rhub = 3  # hub radius
dr = R / increments  # Length of Analyzed Blade Segment
Nb = 3.0  # Number of Blades
H_tower = 150  # Hub Height in meters

# The arrays smooth_BEMT_Rot and smooth_BEMT_Pit contain the interpolated values

# Create arrays for chord length and twist from ref geometry
with open('AF/BladeData.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    columns_as_lists = [list(c) for c in zip(*csv_reader)]
    span = columns_as_lists[0]
    span = np.array(span[2:]).astype(np.float64)
    twist = columns_as_lists[4]
    twist = np.array(twist[2:]).astype(np.float64)
    twist = twist * np.pi / 180
    chord = columns_as_lists[5]
    chord = np.array(chord[2:]).astype(np.float64)
    AFID = columns_as_lists[6]
    AFID = np.array(AFID[2:]).astype(np.float64)

CL = [None] * 50
CD = [None] * 50
alpha = [None] * 50
for i in range(50):
    with open('AF/AF' + str(i + 1) + '.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        columns_as_lists = [list(c) for c in zip(*csv_reader)]
        CL[i] = np.array(columns_as_lists[1][1:]).astype(np.float64)
        CD[i] = np.array(columns_as_lists[2][1:]).astype(np.float64)
        alpha[i] = np.array(columns_as_lists[0][1:]).astype(np.float64)
        alpha[i] = alpha[i] * np.pi / 180

# Set increments along the blade geometry
radius = np.linspace(0, R, round(increments) + 1)

def BEMT_Solve_ROSCO(WS, time_arr, blade_pitch, rot_speed):
    # mot_amp in rad and mot_freq in rad/s
    time_increments = len(time_arr)  # length of time array

    pitch_sched = np.ones_like(time_arr) * blade_pitch
    rotor_speed = np.ones_like(time_arr) * rot_speed

    WS_adj = [[0 for t in range(time_increments)] for r in range(increments)]
    for t in range(time_increments):
        for r in range(increments):
            WS_adj[r][t] = WS[t]


    dQ = [[0 for t in range(time_increments)] for r in range(increments)]
    dP = [[0 for t in range(time_increments)] for r in range(increments)]
    dT = [[0 for t in range(time_increments)] for r in range(increments)]

    tols = 0.0001
    tols_ap = 0.0001
    iters = 1000

    for r in range(int(increments)):
        mid = radius[r] + dr / 2  # Define midpoint radius of blade element
        print("Anlayzing Annulus at Radius = " + str(mid) + "...")

        ch = np.interp(mid, span, chord)  # Find local blade element chord and twist from arrays
        tw = np.interp(mid, span, twist)
        sig_prime = (Nb * ch) / (2 * np.pi * mid)  # Local Solidity Factor
        A = np.zeros(time_increments)  # Initialize axial, angular induction arrays for each time step
        AP = np.zeros(time_increments)
        alphaData, Cl_data, Cd_data = GetAirfoilPolars(mid)  # get airfoil polars as func of alpha

        for t in range(int(time_increments)):
            U1 = WS_adj[r][t]
            pitch_angle = pitch_sched[t]
            pit = tw + pitch_angle
            omega = rotor_speed[t]
            lam_r = omega * mid / U1
            current_time = time_arr[t]

            # Starting guesses for BEM convergence
            a_0 = 0.25 * (2 + np.pi * lam_r * sig_prime - np.sqrt(np.abs(
                4 - (4 * np.pi * lam_r * sig_prime) + (
                        np.pi * lam_r ** 2 * sig_prime * (8 * pit + np.pi * sig_prime)))))
            ap_0 = 0.0

            err_a, err_ap, a_1, ap_1, Glauert_on = cost_func_basic_F_G(a_0, ap_0, pit, mid, alphaData, Cl_data, Cd_data,
                                                       U1, omega, ch)
            count_a = 0

            while (err_a > tols or err_ap > tols_ap) and count_a < iters:
                a_0 = a_1
                ap_0 = ap_1
                err_a, err_ap, a_1, ap_1, Glauert_on = cost_func_basic_F_G(a_0, ap_0, pit, mid, alphaData, Cl_data, Cd_data,
                                                           U1, omega, ch)
                count_a = count_a + 1

            if count_a >= iters and err_a > tols:
                print('WARNING AXIAL INDUCTION FACTOR DID NOT CONVERGE AT TIME = ' + str(t))
            if count_a >= iters and err_ap > tols_ap:
                print('WARNING ANGULAR INDUCTION FACTOR DID NOT CONVERGE AT TIME = ' + str(t))

            A[t] = a_1
            AP[t] = ap_1
            u = U1 * (1 - A[t])  # Local Out of Plane Air Velocity
            w = omega * mid * (1 + AP[t])  # Local In Plane Air Velocity
            phi = np.arctan(u / w)
            F = getHubTipLoss(Nb, mid, phi)

            dQ[r][t] = 4 * np.pi * mid ** 3 * rho * U1 * omega * (1 - A[t]) * AP[t] * dr * F
            dP[r][t] = dQ[r][t] * omega
            dT[r][t] = 4 * np.pi * mid * rho * U1 ** 2 * A[t] * (1 - A[t]) * dr * F

            if Glauert_on:
                # Modified TWM Correction
                #ac = 0.17
                #b0 = 0.0705
                #b2 = (b0 / ac**2) - (4 * F)
                #b1 = (4 * F) - (8 * F * ac) - (2 * b2 * ac)
                #Ct_w_Glauert = b0 + b1 * a_1 + b2 * A[t]**2

                # Buhl TWM Correction
                ac = 0.4
                b0 = 8 / 9
                b2 = (b0 / ac**2) - (4 * F)
                b1 = (4 * F) - (8 * F * ac) - (2 * b2 * ac)
                Ct_w_Glauert = b0 + b1 * a_1 + b2 * A[t]**2

                # Hansen Correction
                #ac = 0.2
                #Ct_w_Glauert = 4 * (ac**2 + (1 - 2 * ac) * A[t]) * F

                dT[r][t] = Ct_w_Glauert * 0.5 * rho * U1**2 * 2 * np.pi * mid * dr


    pwr = np.zeros_like(time_arr)
    thr = np.zeros_like(time_arr)
    tor = np.zeros_like(time_arr)

    for t in range(time_increments):
        for r in range(increments):
            pwr[t] = pwr[t] + dP[r][t]
            tor[t] = tor[t] + dQ[r][t]
            thr[t] = thr[t] + dT[r][t]

    pwr = pwr * 0.9655

    return pwr, thr, tor


def BEMT_Solve(WS, time_arr, pitch, rotSpeed):
    # mot_amp in rad and mot_freq in rad/s
    time_increments = len(time_arr)  # length of time array

    pitch_sched = np.ones_like(time_arr) * pitch
    rotor_speed = np.ones_like(time_arr) * rotSpeed

    WS_adj = [[0 for t in range(time_increments)] for r in range(increments)]
    for t in range(time_increments):
        for r in range(increments):
            WS_adj[r][t] = WS[t]


    dQ = [[0 for t in range(time_increments)] for r in range(increments)]
    dP = [[0 for t in range(time_increments)] for r in range(increments)]
    dT = [[0 for t in range(time_increments)] for r in range(increments)]

    tols = 0.0001
    tols_ap = 0.0001
    iters = 1000

    for r in range(int(increments)):
        mid = radius[r] + dr / 2  # Define midpoint radius of blade element
        #print("Anlayzing Annulus at Radius = " + str(mid) + "...")

        ch = np.interp(mid, span, chord)  # Find local blade element chord and twist from arrays
        tw = np.interp(mid, span, twist)
        sig_prime = (Nb * ch) / (2 * np.pi * mid)  # Local Solidity Factor
        A = np.zeros(time_increments)  # Initialize axial, angular induction arrays for each time step
        AP = np.zeros(time_increments)
        alphaData, Cl_data, Cd_data = GetAirfoilPolars(mid)  # get airfoil polars as func of alpha

        for t in range(int(time_increments)):
            U1 = WS_adj[r][t]
            pitch_angle = pitch_sched[t]
            pit = tw + pitch_angle
            omega = rotor_speed[t]
            lam_r = omega * mid / U1
            current_time = time_arr[t]

            # Starting guesses for BEM convergence
            radican = 4 - (4 * np.pi * lam_r * sig_prime) + (np.pi * lam_r ** 2 * sig_prime * (8 * pit + np.pi * sig_prime))
            if radican > 0:
                a_0 = 0.25 * (2 + np.pi * lam_r * sig_prime - np.sqrt(radican))
            else:
                a_0 = 0.25 * (2 + np.pi * lam_r * sig_prime)
            ap_0 = 0.0

            err_a, err_ap, a_1, ap_1, Glauert_on = cost_func_basic_F_G(a_0, ap_0, pit, mid, alphaData, Cl_data, Cd_data,
                                                       U1, omega, ch)
            count_a = 0

            while (err_a > tols or err_ap > tols_ap) and count_a < iters:
                a_0 = a_1
                ap_0 = ap_1
                err_a, err_ap, a_1, ap_1, Glauert_on = cost_func_basic_F_G(a_0, ap_0, pit, mid, alphaData, Cl_data, Cd_data,
                                                           U1, omega, ch)
                count_a = count_a + 1

            if count_a >= iters and err_a > tols:
                print('WARNING AXIAL INDUCTION FACTOR DID NOT CONVERGE AT TIME = ' + str(t))
            if count_a >= iters and err_ap > tols_ap:
                print('WARNING ANGULAR INDUCTION FACTOR DID NOT CONVERGE AT TIME = ' + str(t))

            A[t] = a_1
            AP[t] = ap_1
            u = U1 * (1 - A[t])  # Local Out of Plane Air Velocity
            w = omega * mid * (1 + AP[t])  # Local In Plane Air Velocity
            phi = np.arctan(u / w)
            F = getHubTipLoss(Nb, mid, phi)

            dQ[r][t] = 4 * np.pi * mid ** 3 * rho * U1 * omega * (1 - A[t]) * AP[t] * dr * F
            dP[r][t] = dQ[r][t] * omega
            dT[r][t] = 4 * np.pi * mid * rho * U1 ** 2 * A[t] * (1 - A[t]) * dr * F

            if Glauert_on:
                # Modified TWM Correction
                #ac = 0.17
                #b0 = 0.0705
                #b2 = (b0 / ac**2) - (4 * F)
                #b1 = (4 * F) - (8 * F * ac) - (2 * b2 * ac)
                #Ct_w_Glauert = b0 + b1 * a_1 + b2 * A[t]**2

                # Buhl TWM Correction
                ac = 0.4
                b0 = 8 / 9
                b2 = (b0 / ac**2) - (4 * F)
                b1 = (4 * F) - (8 * F * ac) - (2 * b2 * ac)
                Ct_w_Glauert = b0 + b1 * a_1 + b2 * A[t]**2

                # Hansen Correction
                #ac = 0.2
                #Ct_w_Glauert = 4 * (ac**2 + (1 - 2 * ac) * A[t]) * F

                dT[r][t] = Ct_w_Glauert * 0.5 * rho * U1**2 * 2 * np.pi * mid * dr


    pwr = np.zeros_like(time_arr)
    thr = np.zeros_like(time_arr)
    tor = np.zeros_like(time_arr)

    for t in range(time_increments):
        for r in range(increments):
            pwr[t] = pwr[t] + dP[r][t]
            tor[t] = tor[t] + dQ[r][t]
            thr[t] = thr[t] + dT[r][t]

    pwr = pwr * 0.9655

    return pwr, thr, tor

def cost_func_basic_F_G(a_0, ap_0, pit, mid, alphaData, Cl_data, Cd_data, U1, omega, ch):
    phi = np.arctan(U1 * (1 - a_0) / (omega * mid * (1 + ap_0)))
    alpha = phi - pit
    Cl, Cd = getClCd(alpha, alphaData, Cl_data, Cd_data)
    sig_prime = 3 * ch / (2 * np.pi * mid)
    F = getHubTipLoss(Nb, mid, phi)

    # Define a_crit, Modified TWM = 0.17, Buhl TWM = 0.4, Hansen = 0.2, Aerodyn defined based on Ct
    a_crit = 0.4
    b0 = 8 / 9
    b2 = (b0 / (a_crit ** 2)) - (4 * F)
    b1 = (4 * F) - (8 * F * a_crit) - (2 * b2 * a_crit)

    # Assume BEMT without empirical corrections
    Glauert = False
    a_new = 1 / (1 + ((4 * F * np.sin(phi) * np.sin(phi)) / (sig_prime * (Cl * np.cos(phi) + Cd * np.sin(phi)))))
    ap_new = 1 / (-1 + ((4 * F * np.sin(phi) * np.cos(phi)) / (sig_prime * (Cl * np.sin(phi) - Cd * np.cos(phi)))))
    # Ct_new = 4 * a_new * (1 - a_new) * F

    # if empirical correction required for new a
    if a_new >= a_crit:
        Glauert = True
        # K = (4 * F * np.sin(phi)**2) / (sig_prime * (Cl * np.cos(phi) + Cd * np.sin(phi)))
        # radican = (K * (1 - (2 * a_crit)) + 2)**2 + 4 * (K * a_crit**2 - 1)
        # a_new = 0.5 * (2 + K * (1 - (2 * a_crit)) - np.sqrt(radican))
        Ct_new = b0 + b1 * a_new + b2 * a_new ** 2
        a_new = (np.sqrt(-1 * 4 * b0 * b2 + b1 ** 2 + 4 * b2 * Ct_new) - b1) / (2 * b2)

    err_a = a_new - a_0
    err_ap = ap_new - ap_0

    return err_a, err_ap, a_new, ap_new, Glauert


########################################################################################################################
########################################### SUPPORTING FUNCTIONS #######################################################
########################################################################################################################

def GetAirfoilPolars(mid, CL=CL, CD=CD, alpha=alpha):
    idx = find_nearest(span, mid)

    return alpha[idx], CL[idx], CD[idx]


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def getHubTipLoss(Nb, mid, phi, Rh=Rhub, R=R):
    f_tip = (Nb / 2) * ((R - mid) / (mid * np.sin(phi)))
    F_tip_input = np.exp(-abs(f_tip))
    f_hub = (Nb / 2) * (mid - Rh) / (mid * np.sin(phi))
    F_hub_input = np.exp(-abs(f_hub))
    F_tip = (2 / np.pi) * np.arccos(F_tip_input)
    F_hub = (2 / np.pi) * np.arccos(F_hub_input)
    F = F_tip * F_hub
    return F


def getClCd(alpha_adj, alphaData, ClData, CdData):
    Cl = np.interp(alpha_adj, alphaData, ClData)
    Cd = np.interp(alpha_adj, alphaData, CdData)
    return Cl, Cd


# Testing
#BEMT_time_array = np.linspace(0, 100, 1000)
#WindSpeed = 10
#WS = np.ones_like(BEMT_time_array) * WindSpeed

#for i in range(len(WS)):
#    WS[i] = WS[i] + np.sin(BEMT_time_array[i])

#Power, Aero_Thrust, Aero_Torque = BEMT_Solve(WS, BEMT_time_array)

#plt.figure(1)
#plt.title('Thrust')
#plt.plot(BEMT_time_array, Aero_Thrust / 10**6, label='T')
#plt.plot(BEMT_time_array, WS, label='WS')
#plt.legend()

#plt.figure(2)
#plt.title('Power')
#plt.plot(BEMT_time_array, Power / 10**6, label='P')
#plt.plot(BEMT_time_array, WS, label='WS')
#plt.legend()
#plt.show()
