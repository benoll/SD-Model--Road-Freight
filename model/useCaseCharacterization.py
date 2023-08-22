# FILE: useCaseCharacterization.py
# PROJECT: Global System-Dynamics Freight Model
# MODULE DESCRIPTION: This is the use case characterization module that randomely generates vocational parameters for each of the
# applications. For freight, this means for each year, region and application segment, and for every investor within those segments,
# a weight and range value is randomely selected thus giving a corresponding power and energy ratio.

# Import statements
import pandas as pd
import numpy as np
import time
import externalParameters as ep

# Global constants
NUM_INVESTORS = ep.getNumInvestors()
BASE_YEAR = ep.getYears()['Years'][0]
INIT_USE_CASE_BOOL = ep.getUseCaseParametersInitBool()        # 1 means run the use case initialization, 0 means the use case parameters are pre-initialized

################################################################################
# EXTERNAL FUNCTIONS TO THE MODULE:
################################################################################

def initialize():
    """
    This function initializes all relevant global parameters used in the experienceCurves.py module.
    """

    # Initialize the global variables for the main use case dataframe and average power and energy dataframes
    global USE_CASE_DF
    global POWER_AVG
    global ENERGY_AVG

    # Check to see if the use case parameters have already been pre-initialized:
    # The marker is set in the externalParameters.py module. If the market is equal to 1, all use case parameters must be initialized. If the market is equal to zero, the use case parameters have already been set.
    if INIT_USE_CASE_BOOL:
        # Time the process
        start_time = time.time()

        print(' ')
        print('-------------------------------------------')
        print(' ')
        print("Initializing Use Case Parameters (main) ...")
        print(' ')
        print('-------------------------------------------')

        # Setup the empty storage dataframe
        USE_CASE_DF = ep.getEmptyDataFrame('technologies')
        # Calculate values (could be constant or distributed) for all use case parameters (weight, range, power, energy, payload, etc.) for all investors
        USE_CASE_DF = USE_CASE_DF.loc[(BASE_YEAR, slice(None), slice(None)), :].apply(selectTechnology, axis=1)

        print(' ')
        print('-------------------------------------------')
        print(' ')
        print('Storing use case parameters ... ')
        print(' ')
        print('-------------------------------------------')

        # Save the main use case parameters dataframe to the '_02_Intermediate Output Files' Folder
        if ep.getParameterType('PerformanceCharacteristics') == 'constant':
            USE_CASE_DF.to_pickle(ep.getUseCaseCharacterizationOutputDirectory() + '\_USE_CASE_DF_constant_' + str(NUM_INVESTORS) + '.pkl')
        elif ep.getParameterType('PerformanceCharacteristics') == 'distributed':
            USE_CASE_DF.to_pickle(ep.getUseCaseCharacterizationOutputDirectory() + '\_USE_CASE_DF_distributed_' + str(NUM_INVESTORS) + '.pkl')

        # Store all other uce case parameters (power, energy, weight, range, payload factor, etc.) for data analysis or plotting
        storeUseCaseParametersMean()
        storeUseCaseParametersAll('Weight')
        storeUseCaseParametersAll('Range')
        storeUseCaseParametersAll('Power')
        storeUseCaseParametersAll('Energy')
        storeUseCaseParametersAll('Payload Factor')
        storeUseCaseParametersAll('Loading Factor')
        storeUseCaseParametersAll('Empty Run Share')

        end_time = time.time()

        print('---------------------------------------------------------------')
        print('   ')
        print('UCC INITIALIZATION RUN TIME:')
        print('--- ',np.around((end_time - start_time), decimals=3), 'seconds ---')
        print('--- ',np.around((end_time - start_time)/60, decimals=3), 'minutes ---')
        print('---------------------------------------------------------------')

    else:
        print('---------------------------------------------------------------')
        print('   ')
        print("USE_CASE_DF was pre-initialized.")
        print('   ')
        print('---------------------------------------------------------------')

        if ep.getParameterType('PerformanceCharacteristics') == 'constant':
            USE_CASE_DF = pd.read_pickle(ep.getUseCaseCharacterizationOutputDirectory() + '\_USE_CASE_DF_constant_' + str(NUM_INVESTORS) + '.pkl')
        elif ep.getParameterType('PerformanceCharacteristics') == 'distributed':
            USE_CASE_DF = pd.read_pickle(ep.getUseCaseCharacterizationOutputDirectory() + '\_USE_CASE_DF_distributed_' + str(NUM_INVESTORS) + '.pkl')


    # Create an "avg" USE_CASE_DF dataframe that returns the average power and energy values for all modelled investors
    POWER_AVG = ep.getEmptyDataFrame('technologies').loc[(BASE_YEAR, slice(None), slice(None)), :]
    ENERGY_AVG = ep.getEmptyDataFrame('technologies').loc[(BASE_YEAR, slice(None), slice(None)), :]
    POWER_AVG = POWER_AVG.apply(computeParameterAverage, axis=1, args=['Power'])
    ENERGY_AVG = ENERGY_AVG.apply(computeParameterAverage, axis=1, args=['Energy'])

def getAvgPowerEnergyValues(region, application, parameter):
    """
    This function returns the power and energy values for all investors for a specific region and
    application.
    """
    if parameter == 'Power':
        return POWER_AVG.loc[(BASE_YEAR, region, application), :]
    else:
        return ENERGY_AVG.loc[(BASE_YEAR, region, application), :]

def getUseCaseParameters(region, application, technology, parameter):
    """
    This function is called from other external modules and will return the use-case parameters
    for a specified year, region, application, technology. A specific
    parameter can be additionally specified.

    The parameters returned are: [Weight, Range, Power, Energy, Payload Factor, Loading Factor, Empty Run Share]

    These parameters are returned for ALL investors as a series with row length NUM_INVESTORS. The
    columns as the above returned parameters.

    NOTE: if the useCaseCharacterization.py module is running with 'constant' parameters,
    the returned parameter values for every investor will be the same. Only
    when the module is running with 'distributed' parameters will there be a
    difference between returned investor-specific use-case parameters.
    """

    if parameter == 'all':
        return USE_CASE_DF.loc[(BASE_YEAR, region, application), technology]
    else:
        return USE_CASE_DF.loc[(BASE_YEAR, region, application), technology].loc[:, parameter]

def printUseCaseParameters(year, region, application, technology, investor):
    """
    This function prints the specific use-case parametrs specified by the passed
    parameters. This funciton would be called primarily from the main.py module for analysis.
    """
    # First find the [technologies x parameters] matrix for the given state and investor
    technology_specific_use_case_param = USE_CASE_DF.loc[(year, region, application), technology]

    print('For the following segment: [', year, region, application, technology, investor, '] the use-case parameter values are:')
    print('Power: ', np.around(technology_specific_use_case_param.loc[(investor), 'Power'], decimals=2), ' [kW]')
    print('Energy: ', np.around(technology_specific_use_case_param.loc[(investor), 'Energy'], decimals=2), ' [kWh]')
    print('Weight: ', np.around(technology_specific_use_case_param.loc[(investor), 'Weight'], decimals=2), ' [kg]')
    print('Range: ', np.around(technology_specific_use_case_param.loc[(investor), 'Range'], decimals=2), ' [km/day]')

################################################################################
# INTERNAL FUNCTIONS TO THE MODULE:
################################################################################

def computeParameterAverage(technologyRow, parameter):
    """
    This function calculates the average value for all modelled investors for each application in
    each region and year for the passed parameter. The two passed parameters are either 'power'
    or 'energy' and the passed dataframe 'technologyRow' is an empty single column dataframe with
    a technology as the column and the year, region and application as the multiindexed rows.
    """
    year = technologyRow.name[0]
    region = technologyRow.name[1]
    application = technologyRow.name[2]

    # Iterate through the technologies to assign use case parameters for each of the investors
    for technology, element in technologyRow.items():
        # Get the investor array of power values
        parameter_array = USE_CASE_DF.loc[(BASE_YEAR, region, application), technology].loc[:,parameter]
        # Take the mean and assign it to the technologyRow
        technologyRow[technology] = parameter_array.mean()

    return technologyRow

def storeUseCaseParametersMean():
    """
    This function stores only the weight, range, power and energy values together as the main use case
    parameters for each technology in each application and region in an organized matrix so that
    the data can be easily analysed and ploted later. Here, the mean power and energy
    values for all investors is stored. NOTE: values are only stored for the base year
    (2020) because the assumption is that all use case parameters remain constant throughout
    all modelling years.
    """

    # Initialize the empty dataframe
    df = pd.DataFrame(columns=['Year','Region','Application','Technology', 'Weight','Range','Power','Energy'])
    # Loop through all regions, applications and technologies and input data
    for region in ep.getRegionsString():
        for application in ep.getApplicationsString():
            for technology in ep.getTechnologiesString():

                weight = USE_CASE_DF.loc[(BASE_YEAR, region, application), technology]['Weight'].mean()
                range = USE_CASE_DF.loc[(BASE_YEAR, region, application), technology]['Range'].mean()
                power = USE_CASE_DF.loc[(BASE_YEAR, region, application), technology]['Power'].mean()
                energy = USE_CASE_DF.loc[(BASE_YEAR, region, application), technology]['Energy'].mean()

                dict = {'Year':[BASE_YEAR],
                        'Region':[region],
                        'Application':[application],
                        'Technology':[technology],
                        'Weight':[weight],
                        'Range':[range],
                        'Power':[power],
                        'Energy':[energy]}
                df_dict = pd.DataFrame(dict)
                df = pd.concat([df, df_dict], ignore_index=True)

    # Set the dataframe index
    df.set_index(['Year','Region', 'Application','Technology'], inplace=True)

    # Save the dataframe as a pickle and xlsx file
    df.to_pickle(ep.getUseCaseCharacterizationOutputDirectory() + '\_USE_CASE_PARAMETERS_MEAN_DF.pkl')
    df.to_excel(ep.getUseCaseCharacterizationOutputDirectory() + '\_use_case_parameters_mean.xlsx')

def storeUseCaseParametersAll(parameter):
    """
    This function stores all use case parameter values separately for each technology
    in each application and region in an organized matrix so that the data can be easily analysed
    and ploted later. Here, parameter values are stored for all investors. NOTE: values are only
    stored for the base year (2020) because the assumption is that all use case parameters remain
    constant throughout all modelling years.
    """

    # Initialize the empty dataframe
    df = pd.DataFrame()
    # Loop through all regions, applications and technologies and input data
    for region in ep.getRegionsString():
        for application in ep.getApplicationsString():
            for technology in ep.getTechnologiesString():
                    parameter_series = pd.Series([BASE_YEAR, region, application, technology])
                    values_series = USE_CASE_DF.loc[(BASE_YEAR, region, application), technology][parameter]
                    parameter_series = pd.concat([parameter_series,values_series], ignore_index=True)
                    # Append the dimensions
                    df = pd.concat([df, parameter_series.to_frame().T],ignore_index=True)



    # Relable the dataframe columns
    columns = pd.Series(['Year','Region','Application','Technology'])
    investor_col = pd.Series(range(0,NUM_INVESTORS))
    columns = pd.concat([columns, investor_col], ignore_index=True)
    df.rename(columns=columns, inplace=True)

    # Set the dataframe index
    df.set_index(['Year','Region', 'Application','Technology'], inplace=True)


    # Save the dataframe as a pickle and xlsx file
    df.to_pickle(ep.getUseCaseCharacterizationOutputDirectory() + '\_' + parameter.upper().replace(" ", "_") + '_' + str(NUM_INVESTORS) + '_INVESTORS_DF.pkl')
    df.to_excel(ep.getUseCaseCharacterizationOutputDirectory() + '\_' + parameter.lower().replace(" ", "_") + '_' + str(NUM_INVESTORS) + '_investors.xlsx')

def setTechnologyIndependentParameters(tech_independent_parameters_series, region, application):
    """
    This function calculates the use case characterization parameters, for an individual investor,
    that do NOT vary between technology but do vary between application and region.

    These parameters are: weight, range, vehicle performance characteristics,
    payload, loading factor (and fuel consumption).
    """

    # If the 'MASTER' boolean is set to true, change the region to 'MASTER'
    if ep.getMasterBoolean():
        region = 'MASTER'

    # Get the stochastic parameters for each investor
    weight_min = ep.getApplicationData(region, application, 'Min Weight')
    weight_avg = ep.getApplicationData(region, application, 'Avg Weight')
    weight_max = ep.getApplicationData(region, application, 'Max Weight')
    weight = ep.distributionFunction(np.array([weight_min, weight_avg, weight_max]), 'PERT', 1)
    range_min = ep.getApplicationData(region, application, 'Min Range')
    range_avg = ep.getApplicationData(region, application, 'Avg Range')
    range_max = ep.getApplicationData(region, application, 'Max Range')
    range = ep.distributionFunction(np.array([range_min, range_avg, range_max]), 'PERT', 1)
    payload_factor_min = ep.getApplicationData(region, application, 'Min Payload % of GVW')
    payload_factor_avg = ep.getApplicationData(region, application, 'Avg Payload % of GVW')
    payload_factor_max = ep.getApplicationData(region, application, 'Max Payload % of GVW')
    payload_factor = ep.distributionFunction(np.array([payload_factor_min, payload_factor_avg, payload_factor_max]), 'PERT', 1)
    loading_factor_min = ep.getApplicationData(region, application, 'Min Loading % of Payload')
    loading_factor_avg = ep.getApplicationData(region, application, 'Avg Loading % of Payload')
    loading_factor_max = ep.getApplicationData(region, application, 'Max Loading % of Payload')
    loading_factor = ep.distributionFunction(np.array([loading_factor_min, loading_factor_avg, loading_factor_max]), 'PERT', 1)
    empty_run_share_min = ep.getApplicationData(region, application, 'Min Empty Run Share')
    empty_run_share_avg = ep.getApplicationData(region, application, 'Avg Empty Run Share')
    empty_run_share_max = ep.getApplicationData(region, application, 'Max Empty Run Share')
    empty_run_share = ep.distributionFunction(np.array([empty_run_share_min, empty_run_share_avg, empty_run_share_max]), 'PERT', 1)
    C_d_min = ep.getDynamicVehicleParametersData(region, application, 'Min C_d')
    C_d_avg = ep.getDynamicVehicleParametersData(region, application, 'Avg C_d')
    C_d_max = ep.getDynamicVehicleParametersData(region, application, 'Max C_d')
    C_d = ep.distributionFunction(np.array([C_d_min, C_d_avg, C_d_max]), 'PERT', 1)
    A_f_min = ep.getDynamicVehicleParametersData(region, application, 'Min A_f')
    A_f_avg = ep.getDynamicVehicleParametersData(region, application, 'Avg A_f')
    A_f_max = ep.getDynamicVehicleParametersData(region, application, 'Max A_f')
    A_f = ep.distributionFunction(np.array([A_f_min, A_f_avg, A_f_max]), 'PERT', 1)
    C_rr_min = ep.getDynamicVehicleParametersData(region, application, 'Min C_rr')
    C_rr_avg = ep.getDynamicVehicleParametersData(region, application, 'Avg C_rr')
    C_rr_max = ep.getDynamicVehicleParametersData(region, application, 'Max C_rr')
    C_rr = ep.distributionFunction(np.array([C_rr_min, C_rr_avg, C_rr_max]), 'PERT', 1)

    # Insert the values into the dataframe depending on whether the module is set to constant or distributed (NOTE: this is set in the externalParameters.py module)
    if ep.getParameterType('PerformanceCharacteristics') == 'constant':
        tech_independent_parameters_series.at['Weight'] = weight_avg
        tech_independent_parameters_series.at['Range'] = range_avg
        tech_independent_parameters_series.at['Payload Factor'] = payload_factor_avg
        tech_independent_parameters_series.at['Loading Factor'] = loading_factor_avg
        tech_independent_parameters_series.at['Empty Run Share'] = empty_run_share_avg
        tech_independent_parameters_series.at['C_d'] = C_d_avg
        tech_independent_parameters_series.at['A_f'] = A_f_avg
        tech_independent_parameters_series.at['C_rr'] = C_rr_avg
    elif ep.getParameterType('PerformanceCharacteristics') == 'distributed':
        tech_independent_parameters_series.at['Weight'] = weight
        tech_independent_parameters_series.at['Range'] = range
        tech_independent_parameters_series.at['Payload Factor'] = payload_factor
        tech_independent_parameters_series.at['Loading Factor'] = loading_factor
        tech_independent_parameters_series.at['Empty Run Share'] = empty_run_share
        tech_independent_parameters_series.at['C_d'] = C_d
        tech_independent_parameters_series.at['A_f'] = A_f
        tech_independent_parameters_series.at['C_rr'] = C_rr

    return tech_independent_parameters_series

def selectTechnology(technologyRow):
    """
    This function takes in a row of the main USE_CASE_DF dataframe which contains all modelled technologies for
    a specific year, region and application. A new dataframe is then created for each technology in the parameter
    technologyRow where all use case characteristics are organized for all N_investors. For each technology, this
    function (selectTechnology()) then calls two other functions: setStochasticParameterInvestorMatrix() to
    determine the use case parameters that are independent of the technology (i.e. weight, range, C_d) and
    calculateUseCaseParameters() to determine the use case parameters that depend on the technology (i.e. power
    and energy) for each investor.
    """

    # Initialize the year, region and application
    year = technologyRow.name[0]
    region = technologyRow.name[1]
    application = technologyRow.name[2]

    print('USE CASE CHARACTERIZATION INIT : curr state : ['+ str(year) + ', ' + region + ', ' + application + ']')

    # Initialize the dataframe for use case parameters that do not depend on the technology for all
    tech_independent_parameters_list = ['Weight', 'Range', 'Payload Factor', 'Loading Factor', 'Empty Run Share', 'C_d', 'A_f', 'C_rr']
    tech_independent_parameters_df = pd.DataFrame(index=np.arange(NUM_INVESTORS), columns=tech_independent_parameters_list, dtype=np.float64)                   # NOTE: #investors x #returnedParameters    (dataframe)

    # Calculate the use case parameters that are independent of the technology for all investors (parameters may be distributed or constant)
    tech_independent_parameters_df = tech_independent_parameters_df.apply(setTechnologyIndependentParameters, axis=1,  args=[region, application])

    # Calculate the technology dependent parameters, vehicle power and energy values, for each each technology and investor
    for technology, element in technologyRow.items():
        technologyRow[technology] = calculateVehiclePowerAndEnergy(year, region, application, technology, tech_independent_parameters_df)

    return technologyRow

def getDriveProfileData(region, application, parameter):
    """
    This function takes in a region, application, and time or velocity parameter and returns the
    associated drive cycle data. Drive cycle selections are established in the ModelArchitectureData.xlsx
    file in the 'Drive Cycle Selection' sheet.
    """

    # Determine the drive cycle selection
    drive_cycle_selection = ep.getDriveCycleSelection(region, application)
    # Get the corresponding drive cycle data
    drive_cycle_data = ep.getDriveCycleData(drive_cycle_selection, parameter)
    # Get the application segment specific drive cycle modes
    drive_cycle_modes = ep.getDriveCycleModeData(drive_cycle_selection, application)


    return drive_cycle_data[drive_cycle_modes].values

def calculateVehiclePowerAndEnergy(year, region, application, technology, tech_independent_parameters_df):
    """
    This function splits the technology columns into single elements, i.e.
    an 'technologyElement' would be a single cell defined by a YEAR, REGION, APPL,
    and TECHNOLOGY. Within this element, use case parameters will be selected for
    nInvestors.

    NOTE: 'technologyElement' is a 1 x 1 element or cell

    return: matrix of use case parameters [nInvestors] x [weight, range, power, energy]

    """

    # Setup the datafram for #Investors x #ReturnedParameters that will be returned at the end of the function
    parameters = ['Weight', 'Range', 'Power', 'Energy', 'Payload Factor', 'Loading Factor', 'Empty Run Share']
    investor_all_use_case_parameters_df = pd.DataFrame(index=np.arange(NUM_INVESTORS), columns=parameters, dtype=np.float64)                   # NOTE: #investors x #returnedParameters

    # Equation Data
    weight = tech_independent_parameters_df['Weight'] # [kg]
    range = tech_independent_parameters_df['Range'] # [km/day]
    payload_factor = tech_independent_parameters_df['Payload Factor'] # [%]
    loading_factor = tech_independent_parameters_df['Loading Factor'] # [%]
    empty_run_share = tech_independent_parameters_df['Empty Run Share'] # [%]
    C_d = tech_independent_parameters_df['C_d'] # [unitless]
    A_f = tech_independent_parameters_df['A_f'] # [m2]
    C_rr = tech_independent_parameters_df['C_rr'] # [unitless]
    gravity = ep.getDynamicVehicleParametersData(region, application, 'Gravity') # [m/s2]
    rho_air = ep.getDynamicVehicleParametersData(region, application, 'Air Density') # [kg/m3]
    P_aux = ep.getTechnologyData(technology, application, 'Aux Power')   # [kW]
    depth_of_discharge = ep.getTechnologyData(technology, application, 'Depth of Discharge') # [%]

    # Get the Drive Cycle Data
    velocity = getDriveProfileData(region, application, 'Velocity')*(1000/(60*60))        # [m/s]
    time = np.arange(len(velocity))                                                          # [sec]

    # Calculate the acceleration
    acceleration = np.gradient(velocity,time)                                               # [m/s2]

    # Compute the propulsive force for each investor individually
    P_propulsion = np.zeros((NUM_INVESTORS,len(velocity)))
    for i in np.arange(0,NUM_INVESTORS):
        weight_i = weight[i]
        C_d_i = C_d[i]
        A_f_i = A_f[i]
        C_rr_i = C_rr[i]
        F_propulsion_i = 0.5*rho_air*C_d_i*A_f_i*np.square(velocity) + weight_i*gravity*C_rr_i + weight_i*acceleration     # [N]
        P_propulsion[i] = F_propulsion_i*velocity

    # Add the auxilary power to the propulsive power to get the total power required by the technology-specific drive-train
    P_total = np.amax(P_propulsion,1)/1000 + P_aux                             # [kW]

    # Calculate the technology dependent energy requirement
    eta_technology = ep.getTechnologyData(technology, application, 'TtW Efficiency')
    J_to_kWh = 0.000000277777778
    E_total = J_to_kWh*(1/eta_technology)*(1/depth_of_discharge)*(range*1000)*(np.trapz(np.heaviside(P_propulsion, 0)*P_propulsion + P_aux*1000, time)/np.trapz(velocity,time))  # [kWh]

    # Record the parameters in the dataframe and return
    investor_all_use_case_parameters_df.loc[:, 'Weight'] = weight
    investor_all_use_case_parameters_df.loc[:, 'Range'] = range
    investor_all_use_case_parameters_df.loc[:, 'Power'] = P_total
    investor_all_use_case_parameters_df.loc[:, 'Energy'] = E_total
    investor_all_use_case_parameters_df.loc[:, 'Payload Factor'] = payload_factor
    investor_all_use_case_parameters_df.loc[:, 'Loading Factor'] = loading_factor
    investor_all_use_case_parameters_df.loc[:, 'Empty Run Share'] = empty_run_share

    return investor_all_use_case_parameters_df
