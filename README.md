To correctly run this model ANSYS AQWA needs to be installed
The hydrodynamic model for the desired turbine platform must be created in aqwa independently
The code in this repository should be run from "...\ANSYSACADEMICSTUDENT_2024R1_WINX64\aqwa\WINX64\aqwa\utils\ExternalForceCalculation\"

This code is written for the example of the IEA 15 MW RWT and the UMaine Volturn-US floater
The enviornmental paramters are 50 GMM cluster centers for 1 year of data from the Celtic Sea PDA 1 lease site in the UK

AerodynamicModule_Solver_Server -- This file creates the server that communicates between AQWA and the user code. This file is made with irregular wave loading in mind, but is easily repurposed for the regular wave case
AqwaServerMgr -- Class for server
BEMT_Pitch.csv -- rated pitch values for IEA 15 MW RWT, used when bypassing ROSCO control
BEMT_RotSpeed.csv -- rated rotor speed values for IEA 15 MW RWT, used when bypassing ROSCO control
BEMT_Simulation_Toolbox -- functions for BEMT aerodynamic solver
DISCON.IN -- rosco control deifnition file
ROSCO_toolbox.zip -- this is the file that contains all ROSCO source code. Needs to be unzipped to work.
GMM_Cluser_Centers.zip -- Contains the csv files of the Gaussian Mixture Model cluster centers for the ERA 5 data at four case study locations. These cluster centers are inputs to AQWA and for the AerodynamicModule_Solver_Server file.

I would highly recommend consulting the AQWA reference manual external user force section for creating the external user force server workflow
