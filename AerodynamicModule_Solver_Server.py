########################################################################################################
#
#  This code provides users of Ansys Aqwa with examples of how to use the interface allowing them to
#  apply forces onto Aqwa structures at runtime. Please do not use if you are not a registered user of Aqwa.
#
########################################################################################################
from AqwaServerMgr import *
from BEMT_Simulation_Toolbox import *
import csv
import numpy as np
import platform, os
from ROSCO_toolbox import controller as ROSCO_controller
from ROSCO_toolbox import turbine as ROSCO_turbine
from ROSCO_toolbox import sim as ROSCO_sim
from ROSCO_toolbox import control_interface as ROSCO_ci
import ROSCO_toolbox

def kaimal_spectrum(f, U, z, I):
    """
    Compute the IEC 61400-1 Kaimal turbulence spectrum for given frequency f, mean wind speed U, and height z.
    """
    L = 0.7 * z  # Turbulence length scale
    #sigma_u = I * U  # Standard deviation of turbulence from IEC model
    sigma_u = I * ((0.75 * U) + 5.6)  # Standard deviation of turbulence from IEC model
    S_u = (4 * sigma_u**2 * L / U) / ((1 + 6 * f * L / U) ** (5 / 3))
    return S_u


def generate_kaimal_wind_series(U, z, dt, total_time, I):
    """
    Generate a 14000s long wind speed time series using Kaimal spectrum.
    """
    N = int(total_time / dt)  # Number of time steps
    T = total_time  # Total simulation time
    df = 1 / T  # Frequency resolution
    f = np.fft.rfftfreq(N, d=dt)[1:]  # Frequency array (exclude zero frequency to avoid division errors)

    # Compute spectral density
    S_u = kaimal_spectrum(f, U, z, I)

    phi = 2 * np.pi * np.random.rand(len(f))

    # Compute Fourier coefficients
    sqrt_Su = np.sqrt(2 * S_u * df)  # Magnitude scaling
    wind_spectrum = np.zeros(N // 2 + 1, dtype=complex)
    wind_spectrum[1:] = sqrt_Su * np.exp(1j * phi)  # Assign nonzero frequencies
    wind_spectrum = np.concatenate(([0], wind_spectrum))  # Include zero frequency

    # Convert back to time domain using IFFT
    wind_speed_series = np.fft.irfft(wind_spectrum, n=N)

    # Normalize to have correct standard deviation
    wind_speed_series = wind_speed_series / np.std(wind_speed_series) * (I * U)

    # Add mean wind speed
    wind_speed_series += U

    return wind_speed_series, np.linspace(0, total_time, N)


def create_wind_speed_array(U1, z, dt, ten_min, turbulence_intensity, endTime):

    ten_min_turb_wind, ten_min_time = generate_kaimal_wind_series(U1, z, dt, ten_min, turbulence_intensity)

    number_of_10min_chunks = round(endTime / ten_min)

    wind_speed_array = np.tile(ten_min_turb_wind, number_of_10min_chunks)

    wind_speed_array = np.concatenate([wind_speed_array, [U1]])

    return wind_speed_array

Run_index = 0
Uinf_run_array = np.array([
    10.09285201, 9.440898615, 15.53738903, 9.79333741, 4.962839891,
    7.171379166, 6.691489992, 16.75561289, 13.44621354, 8.316359763,
    6.228783518, 8.227358326, 10.99279412, 14.154364, 5.852751049,
    13.99968469, 7.30901649, 12.23475599, 3.393388086, 5.791376392,
    6.406359241, 6.06178846, 4.636675599, 6.347603875, 15.92109071,
    11.43765152, 8.718111991, 6.578153699, 8.068540629, 9.259450942,
    6.015263986, 6.702899643, 4.16845219, 10.91011173, 5.306589326,
    5.076046966, 19.18905229, 10.79281317, 11.23001047, 10.71987067,
    8.334287989, 10.6303028, 6.895889654, 5.561838647, 2.505032838,
    12.28063753, 3.933188604, 10.45875116, 5.517096689, 8.827419278
])
Uangle_run_array = np.array([
    221.44788, 46.43142127, 255.5371938, 347.560593, 260.3816575,
    60.56178119, 129.2394185, 223.2977382, 183.7997084, 41.48447986,
    258.2750622, 184.1840651, 330.3387861, 5.90035728, 234.7517581,
    160.7246141, 4.654499276, 225.5081198, 322.8931114, 11.47891089,
    93.18163078, 260.6128115, 357.0203146, 31.09232056, 90.46906599,
    79.76082293, 259.8830293, 292.103416, 274.5451108, 26.66159422,
    186.9219246, 223.1527321, 61.16902811, 29.40298702, 88.64644143,
    264.7349877, 187.1398614, 333.5834171, 245.6100475, 259.1817533,
    4.405953189, 24.20838411, 255.9991752, 180.9405921, 331.4160073,
    288.7068415, 236.111398, 234.5206277, 72.64256788, 219.451065
])
Wvangle_run_array = np.array([
    195.794549, 59.63239355, 222.2266299, 0.161170894, 143.1494083,
    82.46709364, 98.86966818, 202.7370766, 174.244025, 48.14582132,
    83.25349173, 158.1174181, 278.6385719, 29.53433197, 122.3725105,
    147.0263145, 63.56949941, 210.2677231, 81.02567607, 159.3431182,
    94.18458227, 180.9348327, 179.0663338, 81.91049957, 97.658965,
    89.53834652, 187.0839287, 355.8928459, 272.6215082, 53.09404449,
    136.7525457, 156.3339187, 54.80386755, 41.5416247, 140.8807254,
    46.8768851, 169.8278169, 334.0055403, 216.5759031, 238.9951811,
    21.7506482, 32.77629492, 215.9774667, 96.07115896, 110.7232918,
    262.3603629, 106.566005, 139.7171405, 59.60724229, 180.057766
])
T_run_array = np.array([
    5.248870859, 6.457414027, 6.865043946, 5.4350899, 6.322966859,
    5.710411919, 6.7000076, 6.618754896, 6.748287617, 5.214535276,
    7.615453812, 5.576614255, 5.93171769, 7.000305912, 6.892201336,
    6.219939638, 7.183043948, 5.327895355, 7.047467406, 6.487204837,
    5.892846376, 5.890261507, 6.005926842, 6.860292891, 7.660820576,
    6.427559501, 7.396577515, 5.923856876, 4.84986771, 7.237900693,
    6.094458751, 5.946161191, 5.952828425, 6.203170525, 5.860540028,
    6.875117586, 8.528070021, 5.19125338, 5.546501631, 5.432761424,
    5.719589877, 5.367616897, 5.209205765, 7.263804846, 11.05943145,
    6.103999126, 6.816059534, 7.747951384, 7.42054717, 5.497613747
])
A_run_array = np.array([
    1.293494195, 1.72113168, 2.90064413, 1.616010153, 0.899642828,
    1.138675811, 1.404089521, 2.678897258, 2.421114555, 1.286414372,
    1.274805145, 1.15867522, 1.900531675, 2.863968394, 1.162631871,
    2.318248375, 1.57444159, 1.564625819, 0.984750591, 1.344147027,
    1.184028624, 1.062583676, 1.080396134, 1.240760945, 3.365450675,
    2.118967017, 1.962592058, 1.086896948, 1.014353622, 1.9728641,
    0.994528958, 0.987374221, 0.987680181, 1.978486127, 1.016990402,
    1.059562053, 4.646680816, 1.708488889, 1.468473593, 1.559570342,
    1.451940351, 1.650818012, 0.905885558, 1.284314635, 1.795072814,
    2.106991093, 0.840679758, 2.10328282, 1.413189129, 1.20331447
])
Fr_run_array = 1 / T_run_array

endTime = 14400 #total time of sim in seconds (first hour will be thrown out to avoid transients)

z = 150
turbulence_intensity = 0.06  #IEC recommendation of Iref for class B turbines
dt = 0.25
ten_min = 600 #length of time of 10-min segement in seconds
BEMT_time_array = np.linspace(0, 1, 1)
GenEff = 96.55
Ng = 1.0
J = 318628138.0

beta = Uangle_run_array[Run_index] * deg2rad

endIndex = round(endTime / dt)
simulation_time_array = np.linspace(0, endTime, endIndex + 1)
cog_data = np.empty([endIndex + 1, 22])
cog_data = np.zeros_like(cog_data)

wind_speed_array = create_wind_speed_array(Uinf_run_array[Run_index], z, dt, ten_min, turbulence_intensity, endTime)

# Specify controller dynamic library path and name
this_dir = os.path.dirname(os.path.abspath(__file__))
example_out_dir = os.path.join(this_dir,'examples_out')
if not os.path.isdir(example_out_dir):
  os.makedirs(example_out_dir)

if platform.system() == 'Windows':
    lib_name = os.path.join(this_dir, '/ROSCO_toolbox/build/libdiscon.dll')
elif platform.system() == 'Darwin':
    lib_name = os.path.join(this_dir, '../ROSCO/build/libdiscon.dylib')
else:
    lib_name = os.path.join(this_dir, '../ROSCO/build/libdiscon.so')

param_filename = os.path.join(this_dir,'DISCON.IN')

controller_int = ROSCO_ci.ControllerInterface(lib_name,param_filename=param_filename)



nac_FA_pos = np.ones_like(simulation_time_array) * 0.0
nac_FA_vel = np.ones_like(simulation_time_array) * 0.0
nac_FA_Acc = np.ones_like(simulation_time_array) * 0.0
pitch = np.ones_like(simulation_time_array) * 0.0
Aero_Thrust = np.ones_like(simulation_time_array) * 1000
Aero_Torque = np.ones_like(simulation_time_array) * 1000
Power = np.ones_like(simulation_time_array) * 0.0
rotorSpeed = np.ones_like(simulation_time_array) * 10 * rpm2radps
genSpeed = np.ones_like(simulation_time_array) * 10 * rpm2radps
genTorque = np.ones_like(simulation_time_array) * 1000
Nac_FA_Acc_fromCon = 0


def UF1(Analysis, Mode, Stage, Time, TimeStep, Pos, Vel):
    # Check that input file is the one expected without .DAT extension
    dt = TimeStep
    Error = 0

    ExpectedFileName = "TIMERESPONSE"
    ActualFileName = Analysis.InputFileName.split("\\")[
        -1]  # We strip the path part of the filename for easy comparison
    if (ActualFileName != ExpectedFileName):
        print("Error. Incorrect input file !")
        print("Expected : " + ExpectedFileName)
        print("Actual : " + ActualFileName)
        Error = 1  # Will cause Aqwa to stop

    # If this passed, we create an empty container for AddMass and Force
    AddMass = BlankAddedMass(Analysis.NOfStruct)
    Force = BlankForce(Analysis.NOfStruct)

    starting_Hub_coords = [-1 * 4 * np.cos(beta), -1 * 4 * np.sin(beta), 150]
    curr_Hub_pos = Analysis.GetNodeCurrentPosition(Struct=0, DefAxesX=starting_Hub_coords[0],
                                                   DefAxesY=starting_Hub_coords[1], DefAxesZ=starting_Hub_coords[2])

    nac_FA_pos[int(Time / TimeStep)] = np.sqrt(curr_Hub_pos[0] ** 2 + curr_Hub_pos[1] ** 2)
    nac_FA_vel[int(Time / TimeStep)] = (nac_FA_pos[int(Time / TimeStep)] - nac_FA_pos[int(Time / TimeStep) - 1]) / dt
    nac_FA_Acc[int(Time / TimeStep)] = (nac_FA_vel[int(Time / TimeStep)] - nac_FA_vel[int(Time / TimeStep) - 1]) / dt
    print('The Nacelle Acceleration at t = ' + str(Time) + ', is ' + str(nac_FA_Acc[int(Time / TimeStep)]))

    # User defined code here
    dWS_tran = (-1 * Vel[0][1] * np.sin(beta)) + (-1 * Vel[0][0] * np.cos(beta))
    dWS_rot = (Vel[0][3] * starting_Hub_coords[2] * np.sin(beta)) + (
                -1 * Vel[0][4] * starting_Hub_coords[2] * np.cos(beta))
    WindSpeed = (wind_speed_array[int(Time / TimeStep)] * np.cos(Pos[0][4]) * np.cos(Pos[0][5])) + dWS_tran + dWS_rot
    WS = np.ones_like(BEMT_time_array) * WindSpeed

    # pitch[int(Time / TimeStep) - 1] = np.interp(WindSpeed, new_ws_Pit, BEMT_Pit_spline(new_ws_Pit)) * deg2rad
    # rotorSpeed[int(Time / TimeStep) - 1] = np.interp(WindSpeed, new_ws_Rot, BEMT_Rot_spline(new_ws_Rot)) * rpm2radps
    # print('Rotor Speed = ' + str(rotorSpeed[int(Time / TimeStep) - 1]))
    # print('pitch (deg) = ' + str(pitch[int(Time / TimeStep) - 1] * rad2deg))

    Power_array, Aero_Thrust_array, Aero_Torque_array = BEMT_Solve(WS, BEMT_time_array, pitch[int(Time / TimeStep) - 1],
                                                                   rotorSpeed[int(Time / TimeStep) - 1])

    Aero_Thrust[int(Time / TimeStep) - 1] = Aero_Thrust_array[len(BEMT_time_array) - 1]
    Aero_Torque[int(Time / TimeStep) - 1] = Aero_Torque_array[len(BEMT_time_array) - 1]
    Power[int(Time / TimeStep) - 1] = Power_array[len(BEMT_time_array) - 1]

    rotorSpeed[int(Time / TimeStep)] = rotorSpeed[int(Time / TimeStep) - 1] + (dt / J) * (
            Aero_Torque[int(Time / TimeStep) - 1] * GenEff / 100 - Ng * genTorque[int(Time / TimeStep) - 1])
    genSpeed[int(Time / TimeStep)] = rotorSpeed[int(Time / TimeStep)] * Ng

    genTorque[int(Time / TimeStep)], pitch[int(Time / TimeStep)], Nac_FA_Acc_fromCon = controller_int.call_controller(
        Time, dt, pitch[int(Time / TimeStep) - 1], genTorque[int(Time / TimeStep) - 1], genSpeed[int(Time / TimeStep)],
        GenEff / 100, rotorSpeed[int(Time / TimeStep)], WindSpeed, NacIMU_FA_Acc=nac_FA_Acc[int(Time / TimeStep)])

    # genTorque[int(Time / TimeStep)] = Aero_Torque[int(Time / TimeStep) - 1]
    T_frc = Aero_Thrust[int(Time / TimeStep) - 1]
    Q_frc = Aero_Torque[int(Time / TimeStep) - 1] / 150
    tht = Pos[0][4]
    gam = Pos[0][5]
    Force[0][0] = (T_frc * np.cos(tht) * np.cos(gam) * np.cos(beta)) + (Q_frc * np.sin(beta))
    Force[0][1] = (T_frc * np.cos(tht) * np.sin(gam) * np.sin(beta)) - (Q_frc * np.sin(beta))
    Force[0][2] = Aero_Thrust[int(Time / TimeStep) - 1] * np.sin(tht)
    Force[0][3] = Aero_Thrust[int(Time / TimeStep) - 1] * starting_Hub_coords[2] * np.sin(beta)
    Force[0][4] = Aero_Thrust[int(Time / TimeStep) - 1] * starting_Hub_coords[2] * np.cos(beta)
    Force[0][5] = 0

    # Now return the results
    cog_data[int(Time / TimeStep) - 1][0] = Time
    cog_data[int(Time / TimeStep) - 1][1] = Pos[0][0]
    cog_data[int(Time / TimeStep) - 1][2] = Pos[0][1]
    cog_data[int(Time / TimeStep) - 1][3] = Pos[0][2]
    cog_data[int(Time / TimeStep) - 1][4] = Pos[0][3]
    cog_data[int(Time / TimeStep) - 1][5] = Pos[0][4]
    cog_data[int(Time / TimeStep) - 1][6] = Pos[0][5]
    cog_data[int(Time / TimeStep) - 1][7] = Vel[0][0]
    cog_data[int(Time / TimeStep) - 1][8] = Vel[0][1]
    cog_data[int(Time / TimeStep) - 1][9] = Vel[0][2]
    cog_data[int(Time / TimeStep) - 1][10] = Vel[0][3]
    cog_data[int(Time / TimeStep) - 1][11] = Vel[0][4]
    cog_data[int(Time / TimeStep) - 1][12] = Vel[0][5]
    cog_data[int(Time / TimeStep) - 1][13] = genTorque[int(Time / TimeStep) - 1] * rotorSpeed[int(Time / TimeStep) - 1]
    cog_data[int(Time / TimeStep) - 1][14] = Aero_Thrust[int(Time / TimeStep) - 1]
    cog_data[int(Time / TimeStep) - 1][15] = Aero_Torque[int(Time / TimeStep) - 1]
    cog_data[int(Time / TimeStep) - 1][16] = pitch[int(Time / TimeStep) - 1]
    cog_data[int(Time / TimeStep) - 1][17] = rotorSpeed[int(Time / TimeStep) - 1]
    cog_data[int(Time / TimeStep) - 1][18] = genTorque[int(Time / TimeStep) - 1]
    cog_data[int(Time / TimeStep) - 1][19] = WindSpeed
    cog_data[int(Time / TimeStep) - 1][20] = Nac_FA_Acc_fromCon
    cog_data[int(Time / TimeStep) - 1][21] = nac_FA_Acc[int(Time / TimeStep)]

    if Time == endTime - 1:
        with open('TimeResponse_CarolinaCoast_5D_Center_' + str(Run_index + 1) + '.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(cog_data)

    return Force, AddMass, Error


Server = AqwaUserForceServer()

for UF in [UF1, UF1, UF1, UF1, UF1, UF1, UF1, UF1, UF1, UF1, UF1, UF1, UF1, UF1, UF1, UF1, UF1, UF1, UF1, UF1, UF1, UF1, UF1, UF1, UF1, UF1, UF1, UF1, UF1, UF1, UF1, UF1, UF1, UF1, UF1, UF1, UF1, UF1, UF1, UF1, UF1, UF1, UF1, UF1, UF1, UF1, UF1, UF1, UF1, UF1]:
    print('This is run Number: ' + str(Run_index))
    wind_speed_array = create_wind_speed_array(Uinf_run_array[Run_index], z, dt, ten_min, turbulence_intensity, endTime)
    beta = Uangle_run_array[Run_index] * deg2rad
    try:
        print("Now running user function {0}".format(UF.__name__))
        Server.Run(UF)
    except Exception as E:  # If an error occurred, we print it but continue
        print("Caught error : ", E)
        with open('TimeResponse_export.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(cog_data)
        print("Skipping to next case")
    Run_index = Run_index + 1

