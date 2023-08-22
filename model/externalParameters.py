# FILE: externalParameters.py
# PROJECT: Global System-Dynamics Freight Model
# MODULE DESCRIPTION: This is the parameter storage module where all data input parameters as well as other model
# constants are stored.


# Import statements
import pandas as pd
import numpy as np
from pert import PERT
import os
from pathlib import Path #added
from bisect import bisect_left
from datetime import datetime

################################################################################
##############################################################
# DIRECTORY INITIALIZATION:
##############################################################
# Working Directory Paths:
# (Change the path of the data here to fit your operating system.)
working_directory = os.getcwd()
parent_path = os.path.dirname(working_directory)
data_directory = parent_path+'\data'
final_output_directory = data_directory+'\_01_Final Output Files'
if not os.path.exists(final_output_directory + "\\" + '_01_Output Plots'):
    os.makedirs(final_output_directory + "\\" + '_01_Output Plots')
output_plots_directory = final_output_directory + '\_01_Output Plots'
use_case_characterization_output_directory = data_directory+'\_02_Use Case Characterization Output Files'
print('working_directory :', working_directory)
print('parent_path :', parent_path)
print('data_directory :', data_directory)
print('final_output_directory :', final_output_directory)
print('--------------------------------------------------------------')

# Working directory functions
def getWorkingDirectory():
    return working_directory
def getParentPath():
    return parent_path
def getDataDirectory():
    return data_directory
def getUseCaseCharacterizationOutputDirectory():
    return use_case_characterization_output_directory
def getFinalOutputDirectory():
    return final_output_directory
def getOutputPlotsDirectory():
    return output_plots_directory

################################################################################
##############################################################
# EXCEL FILE IMPORTS:
##############################################################
# File Reads:
xls_model_architecture_data = pd.ExcelFile(data_directory + '\_model_architecture_data.xlsx')
xls_tco_data = pd.ExcelFile(data_directory + '\_tco_data.xlsx')
xls_control_panel = pd.ExcelFile(data_directory + '\_control_panel.xlsx')

# Model Architechture Data:
years_df = pd.read_excel(xls_model_architecture_data, 'Years').dropna().sort_index()
regions_df = pd.read_excel(xls_model_architecture_data, 'Regions').dropna().sort_index(ignore_index=True)
applications_df = pd.read_excel(xls_model_architecture_data, 'Applications').dropna().sort_index(ignore_index=True)
technologies_df = pd.read_excel(xls_model_architecture_data, 'Technologies').dropna().sort_index(ignore_index=True)
application_data_df = pd.read_excel(xls_model_architecture_data, 'ApplicationData', index_col=[0,1]).sort_index()
technology_data_df = pd.read_excel(xls_model_architecture_data, 'TechnologyData', index_col=[0,1]).sort_index()
tech_boolean_df = pd.read_excel(xls_model_architecture_data,"TechBoolean",index_col=[0]).sort_index()
dynamic_vehicle_parameters_data_df = pd.read_excel(xls_model_architecture_data, 'DynamicVehicleParametersData', index_col=[0,1]).sort_index()
vehicle_sales_forecast_data_df = pd.read_excel(xls_model_architecture_data, 'VehicleSalesForecastData', index_col=[0,1]).sort_index()
switching_cost_markup_data_df = pd.read_excel(xls_model_architecture_data, 'SwitchingCostMarkupData', index_col=[0,1]).sort_index()
switching_cost_threshold_data_df = pd.read_excel(xls_model_architecture_data, 'SwitchingCostThresholdData', index_col=[0]).sort_index()
initial_market_shares_data_df = pd.read_excel(xls_model_architecture_data, 'InitialMarketSharesData', index_col=[0,1,2]).sort_index()
drive_cycle_selection_data_df = pd.read_excel(xls_model_architecture_data, 'DriveCycleSelectionData', index_col=[0,1]).sort_index()
drive_cycle_mode_selection_data_df = pd.read_excel(xls_model_architecture_data, 'DriveCycleModeSelectionData', index_col=[0,1]).sort_index()
cycle_1_df = pd.read_excel(xls_model_architecture_data, 'Cycle_1', index_col=[0])
cycle_2_df = pd.read_excel(xls_model_architecture_data, 'Cycle_2', index_col=[0])
cycle_3_df = pd.read_excel(xls_model_architecture_data, 'Cycle_3', index_col=[0])
cycle_4_df = pd.read_excel(xls_model_architecture_data, 'Cycle_4', index_col=[0])
cycle_5_df = pd.read_excel(xls_model_architecture_data, 'Cycle_5', index_col=[0])
cycle_6_df = pd.read_excel(xls_model_architecture_data, 'Cycle_6', index_col=[0])
fuel_consumption_data_df = pd.read_excel(xls_model_architecture_data, 'FuelConsumptionData', index_col=[0,1]).sort_index()
fuel_consumption_perc_diff_data_df = pd.read_excel(xls_model_architecture_data, 'FuelConsumptionPercDiffData', index_col=[0]).sort_index()
fuel_selection_data_df = pd.read_excel(xls_model_architecture_data, 'FuelSelectionData', index_col=[0,1]).sort_index()
wtt_emission_factors_data_df = pd.read_excel(xls_model_architecture_data, 'WTTEmissionsFactorsData', index_col=[0,1]).sort_index()
ttw_emission_factors_data_df = pd.read_excel(xls_model_architecture_data, 'TTWEmissionsFactorsData', index_col=[0,1]).sort_index()

# TCO Data:
monetary_parameters_data_df = pd.read_excel(xls_tco_data,'MonetaryParametersData',index_col=[0]).sort_index()
experience_component_data_df = pd.read_excel(xls_tco_data,'ExperienceCurveData',index_col=[0,1]).sort_index()
experience_curve_scenarios_df = pd.read_excel(xls_tco_data,'ExperienceCurveScenarios',index_col=[0]).sort_index()
chassis_data_df = pd.read_excel(xls_tco_data,'ChassisData',index_col=[0,1]).sort_index()
experience_component_boolean_df = pd.read_excel(xls_tco_data,'ExperienceComponentBoolean', index_col=[0]).sort_index()
top_up_down_df = pd.read_excel(xls_tco_data,'Top Up-Down', index_col=[0,1]).sort_index()
exogenous_market_capacity_data_df = pd.read_excel(xls_tco_data,'ExogenousMarketCapacity',index_col=[0,1]).sort_index()
exogenous_market_contributions_df = pd.read_excel(xls_tco_data,'ExogenousMarketContributions',index_col=[0]).sort_index()
capex_subsidies_df = pd.read_excel(xls_tco_data,'CAPEXSubsidies',index_col=[0,1,2]).sort_index()
tolls_df = pd.read_excel(xls_tco_data,'Tolls',index_col=[0,1,2]).sort_index()
fuel_costs_df = pd.read_excel(xls_tco_data,'FuelCosts',index_col=[0,1,2]).sort_index()
charging_infrastructure_selection_df = pd.read_excel(xls_tco_data,'ChargingInfrastructureSelection',index_col=[0,1]).sort_index()
charging_infrastructure_data_df = pd.read_excel(xls_tco_data,'ChargingInfrastructureData',index_col=[0,1]).sort_index()
o_and_m_df = pd.read_excel(xls_tco_data,'O&M',index_col=[0,1,2]).sort_index()
insurance_df = pd.read_excel(xls_tco_data,'Insurance',index_col=[0]).sort_index()
wages_df = pd.read_excel(xls_tco_data,'Wages',index_col=[0,1]).sort_index()
scrap_value_df = pd.read_excel(xls_tco_data,'ScrapValue',index_col=[0]).sort_index()
other_tco_parameters_df = pd.read_excel(xls_tco_data,'OtherTCOParameters',index_col=[0,1]).sort_index()

# Control Panel Inputs
cp_features_df = pd.read_excel(xls_control_panel, 'Features', index_col=[0])
cp_carbon_tax_intervention_selection_df = pd.read_excel(xls_control_panel, 'CarbonTaxInterventionSelection', index_col=[0])
cp_carbon_tax_intervention_data_df = pd.read_excel(xls_control_panel, 'CarbonTaxInterventionData', index_col=[0,1])
cp_tolls_intervention_selection_df = pd.read_excel(xls_control_panel, 'TollsInterventionSelection', index_col=[0,1,2])
cp_tolls_intervention_data_df = pd.read_excel(xls_control_panel, 'TollsInterventionData', index_col=[0,1,2,3])
cp_capex_subsidy_intervention_selection_df = pd.read_excel(xls_control_panel, 'CAPEXSubsInterventionSelection', index_col=[0,1,2])
cp_capex_subsidy_intervention_data_df = pd.read_excel(xls_control_panel, 'CAPEXSubsInterventionData', index_col=[0,1,2,3])
cp_private_intervention_selection_df = pd.read_excel(xls_control_panel, 'PrivateInterventionSelection', index_col=[0])
cp_private_intervention_data_df = pd.read_excel(xls_control_panel, 'PrivateInterventionData', index_col=[0,1,2])
cp_geopolitical_shock_selection_df = pd.read_excel(xls_control_panel, 'GeopoliticalShockSelection', index_col=[0])


# Make sure ICE-D technology is first in the list of technologies
# (This is done in order for the code to run the TCO of ICE-D technologies first as certain TCO parameters for other technologies depend on the TCO of the ICE-D technology.)
idx_ice_d = technologies_df[technologies_df['Technology'] == 'ICE-D'].index
value_first_row = technologies_df.iloc[0]
technologies_df.iloc[idx_ice_d] = value_first_row
technologies_df.loc[0] = 'ICE-D'


################################################################################
# FUNCTION DEFINITIONS (most are external, some internal to the module):
################################################################################

def initialize():
    """
    This function initializes all relevant global parameters used in the externalParameters.py
    module and calls other initialization functions for certain model features or interventions.
    """

    # Initialize the global variables
    global BASE_YEAR
    BASE_YEAR = getYears()['Years'][0]
    global MASTER_BOOL
    MASTER_BOOL = cp_features_df.loc[('MASTER Boolean'), 'Value']
    global FUEL_CONSUMPTION_MODE
    FUEL_CONSUMPTION_MODE = cp_features_df.loc[('Fuel Consumption'), 'Value']
    global EMPTY_RUN_BOOL
    EMPTY_RUN_BOOL = cp_features_df.loc[('Empty Running'), 'Value']
    global NUM_INVESTORS
    NUM_INVESTORS = cp_features_df.loc[('Monte Carlo Simulations'), 'Value']
    global USE_CASE_PARAMETERS_INIT_BOOL
    USE_CASE_PARAMETERS_INIT_BOOL = cp_features_df.loc[('Initialize Use Case Parameters Boolean'), 'Value']
    global CAPEX_PARAMETERS_TYPE
    CAPEX_PARAMETERS_TYPE = cp_features_df.loc[('CAPEX Parameters Type'), 'Value']
    global OPEX_PARAMETERS_TYPE
    OPEX_PARAMETERS_TYPE = cp_features_df.loc[('OPEX Parameters Type'), 'Value']
    global USE_CASE_PARAMETERS_TYPE
    USE_CASE_PARAMETERS_TYPE = cp_features_df.loc[('Use Case Parameters Type'), 'Value']

    # Initialize control panel variables: POLICY FEATURES
    initSwitchingCostFeature()
    initCarbonTaxIntervention()
    initTollIntervention()
    initCAPEXSubsidyIntervention()
    initPrivateIntervention()
    initGeopoliticalShock()

################################################################################
################################################################################
# Control Panel Functions: Model Features

def getMasterBoolean():
    global MASTER_BOOL
    if MASTER_BOOL == 'True':
        return True
    elif MASTER_BOOL == 'False':
        return False

def getFuelConsumptionMode():
    global FUEL_CONSUMPTION_MODE
    return FUEL_CONSUMPTION_MODE

def getEmptyRunBoolean():
    global EMPTY_RUN_BOOL
    if EMPTY_RUN_BOOL == 'True':
        return True
    elif EMPTY_RUN_BOOL == 'False':
        return False

def getNumInvestors():
    global NUM_INVESTORS
    return NUM_INVESTORS

def getUseCaseParametersInitBool():
    global USE_CASE_PARAMETERS_INIT_BOOL
    if USE_CASE_PARAMETERS_INIT_BOOL == 'True':
        return True
    elif USE_CASE_PARAMETERS_INIT_BOOL == 'False':
        return False

################################################################################
################################################################################
# Control Panel Functions: Interventions

def initSwitchingCostFeature():
    """
    This function initializes the 5 global variables for the Switching Cost feature: the switching cost boolean, initial market shares dataframe,
    switching cost markup dataframe, switching cost threshold dataframe, and the switching cost threshold boolean dataframe. Descriptions of each
    global variable are included below before declaration.
    """
    # Initialize the switching cost boolean (i.e. is the switching cost feature on or off)
    global SWITCHING_COST_FEATURE_BOOL
    SWITCHING_COST_FEATURE_BOOL = cp_features_df.loc[('Switching Cost'), :].values

    # Initialize the previous year's market shares (drive-technology specific)
    global INITIAL_MARKET_SHARES_DF
    INITIAL_MARKET_SHARES_DF = initial_market_shares_data_df.copy()
    INITIAL_MARKET_SHARES_DF = INITIAL_MARKET_SHARES_DF.reindex(columns=INITIAL_MARKET_SHARES_DF.columns.sort_values())

    # Initialize the switching cost dataframe that will potentially change each year depending on the change in drive-technology specific market shares
    global SWITCHING_COST_MARKUP_DF
    global SWITCHING_THRESHOLD_DF
    global SWITCHING_THRESHOLD_BOOL_DF
    SWITCHING_COST_MARKUP_DF = switching_cost_markup_data_df
    SWITCHING_THRESHOLD_DF = switching_cost_threshold_data_df[['Threshold']]
    SWITCHING_THRESHOLD_BOOL_DF = getEmptyDataFrame('technologies').loc[(2020, slice(None), slice(None)), :].reset_index().drop(labels='YEAR', axis=1).set_index(['REGION','APPLICATION'])
    SWITCHING_THRESHOLD_BOOL_DF.replace(0.0, False, inplace=True)

def initCarbonTaxIntervention():
    """
    This function initializes the 2 global variables for the Carbon Tax intervention: the intervention selection and
    the corresponding intervention data. Both global variables are initialized as dataframes.
    """
    # Initialize the carbon tax intervention selection global variable
    global CARBON_TAX_INTERVENTION_SELECTION_DF
    CARBON_TAX_INTERVENTION_SELECTION_DF = cp_carbon_tax_intervention_selection_df

    # Initialize the corresponding carbon tax intervention selection data for all modelled regions
    global CARBON_TAX_INTERVENTION_DATA_DF
    years_list = cp_carbon_tax_intervention_data_df.columns
    column_list = np.append(np.array(['Region']), years_list.to_numpy())
    CARBON_TAX_INTERVENTION_DATA_DF = pd.DataFrame(columns=column_list).set_index(['Region'])
    CARBON_TAX_INTERVENTION_DATA_DF.columns = CARBON_TAX_INTERVENTION_DATA_DF.columns.astype(int)
    for region in CARBON_TAX_INTERVENTION_SELECTION_DF.index:
        intervention_selection = CARBON_TAX_INTERVENTION_SELECTION_DF.loc[(region), 'Intervention Selection']
        CARBON_TAX_INTERVENTION_DATA_DF.at[(region), :] = cp_carbon_tax_intervention_data_df.loc[(region, intervention_selection), :].values

def getCarbonTaxInterventionSelection(region):
    """
    This function returns the carbon tax intervention selection.
    """
    global CARBON_TAX_INTERVENTION_SELECTION_DF
    return CARBON_TAX_INTERVENTION_SELECTION_DF.loc[(region), 'Intervention Selection']

def getCarbonTaxInterventionData(region, start_year, end_year):
    """
    This function returns the carbon tax intervention data that corresponds to the selected intervention for the
    passed number of years. The returned data is a series that gives the annual carbon tax value for the passed region and the
    passed number of years in USD/tonCO2eq.
    """
    global CARBON_TAX_INTERVENTION_DATA_DF

    # Check to see if the start year has no policy intervention. Even if in the later years a carbon tax intervention is introduced, the investor did not have prior knowledge of this intervention so it should therefore not be reflected in the TCO.
    carbon_tax_start_year = CARBON_TAX_INTERVENTION_DATA_DF.loc[(region), start_year]

    if carbon_tax_start_year == 0:
        carbon_tax_df = carbon_tax_start_year
    else:
        carbon_tax_df = pd.concat([CARBON_TAX_INTERVENTION_DATA_DF.loc[(region), start_year:end_year].squeeze()]*NUM_INVESTORS, axis=1, ignore_index=True).transpose()
        carbon_tax_df.columns = np.arange(end_year-start_year+1)

    return carbon_tax_df

def initTollIntervention():
    """
    This function initializes the 1 global variable for the Toll intervention: the intervention selection. The corresponding toll intervention data
    is inputed into the already initialized global variable tolls_df.
    """
    # Initialize the Tolls intervention selection global variable
    global TOLLS_INTERVENTION_SELECTION_DF
    TOLLS_INTERVENTION_SELECTION_DF = cp_tolls_intervention_selection_df

    for index in TOLLS_INTERVENTION_SELECTION_DF.index:
        region = index[0]
        application = index[1]
        technology = index[2]

        intervention_selection = TOLLS_INTERVENTION_SELECTION_DF.loc[(region, application, technology), 'Intervention Selection']

        # If the intervention selection is "(BAU Tolls)" then the default data initialized in the tolls_df is used
        if intervention_selection == '(BAU Tolls)':
            continue
        else:
            # Initialize the corresponding Tolls intervention data by overriding the tolls_df data from the TCO Data.xlsx file with the intervention data from the Control Panel.xlsx file.
            tolls_df.at[(region, application, technology), :] = cp_tolls_intervention_data_df.loc[(region, application, technology, intervention_selection), :].values

def initCAPEXSubsidyIntervention():
    """
    This function initializes the 1 global variable for the CAPEX Subsidy intervention: the intervention selection. The corresponding intervention data
    is inputed into the already initialized global variable capex_subsidies_df.
    """
    # Initialize the CAPEX Subsidy intervention selection global variable
    global CAPEX_SUBSIDY_INTERVENTION_SELECTION_DF
    CAPEX_SUBSIDY_INTERVENTION_SELECTION_DF = cp_capex_subsidy_intervention_selection_df

    for index in CAPEX_SUBSIDY_INTERVENTION_SELECTION_DF.index:
        region = index[0]
        application = index[1]
        technology = index[2]

        intervention_selection = CAPEX_SUBSIDY_INTERVENTION_SELECTION_DF.loc[(region, application, technology), 'Intervention Selection']

        # If the intervention selection is "(BAU CAPEX Subsidy)" then the default data initialized in the capex_subsidies_df is used
        if intervention_selection == '(BAU CAPEX Subsidy)':
            continue
        else:
            # Initialize the corresponding CAPEX Subsidy intervention data by overriding the capex_subsidies_df data from the TCO Data.xlsx file with the intervention data from the Control Panel.xlsx file.
            capex_subsidies_df.at[(region, application, technology), :] = cp_capex_subsidy_intervention_data_df.loc[(region, application, technology, intervention_selection), :].values

def getCAPEXSubsidyInterventionSelection(region, application, technology):
    """
    This function returns the CAPEX Subsidy intervention selection.
    """
    global CAPEX_SUBSIDY_INTERVENTION_SELECTION_DF
    return CAPEX_SUBSIDY_INTERVENTION_SELECTION_DF.loc[(region, getApplicationSegment(application, 'weight'), technology), 'Intervention Selection']

def initPrivateIntervention():
    """
    This functions initializes the 2 global variables for the Private intervention: the intervention selection and the corresponding
    intervention data. Both global variables are initialized as dataframes.
    """

    # Initialize the private intervention selection global variable
    global PRIVATE_INTERVENTION_SELECTION_DF
    PRIVATE_INTERVENTION_SELECTION_DF = cp_private_intervention_selection_df

    # Initialize the corresponding private intervention selection data for all modelled regions
    global PRIVATE_INTERVENTION_DATA_DF
    years_list = cp_private_intervention_data_df.columns
    column_list = np.append(np.array(['Region', 'Application']), years_list.to_numpy())
    PRIVATE_INTERVENTION_DATA_DF = pd.DataFrame(columns=column_list).set_index(['Region', 'Application'])
    PRIVATE_INTERVENTION_DATA_DF.columns = PRIVATE_INTERVENTION_DATA_DF.columns.astype(int)
    for region in PRIVATE_INTERVENTION_SELECTION_DF.index:
        intervention_selection = PRIVATE_INTERVENTION_SELECTION_DF.loc[(region), 'Intervention Selection']
        PRIVATE_INTERVENTION_DATA_DF.at[(region, 'LDV'), :] = cp_private_intervention_data_df.loc[(region, 'LDV', intervention_selection), :].values
        PRIVATE_INTERVENTION_DATA_DF.at[(region, 'MDV'), :] = cp_private_intervention_data_df.loc[(region, 'MDV', intervention_selection), :].values
        PRIVATE_INTERVENTION_DATA_DF.at[(region, 'HDV'), :] = cp_private_intervention_data_df.loc[(region, 'HDV', intervention_selection), :].values

def getPrivateInterventionSelection(region):
    """
    This function returns the private intervention selection.
    """
    global PRIVATE_INTERVENTION_SELECTION_DF
    return PRIVATE_INTERVENTION_SELECTION_DF.loc[(region), 'Intervention Selection']

def getPrivateInterventionData(year, region, application):
    """
    This function returns the private intervention data that corresponds to the selected private intervention. The returned data is a series that gives the
    private intervention data values for the passed year, region and application.
    """
    global PRIVATE_INTERVENTION_DATA_DF
    return PRIVATE_INTERVENTION_DATA_DF.loc[(region, application), year]

def initGeopoliticalShock():
    """
    This function initializes the 2 global variables for the Geopolitical Shock: the intervention selection and the corresponding
    intervention data. Both global variables are initialized as dataframes.
    """
    # Initialize the isolation feature selection
    global GEOPOLITICAL_SHOCK_SELECTION_DF
    GEOPOLITICAL_SHOCK_SELECTION_DF = cp_geopolitical_shock_selection_df

def getGeopoliticalShockSelection(region):
    """
    This function initializes the 2 global variables for the Geopolitical Shock: the intervention selection and the corresponding
    intervention data. Both global variables are initialized as dataframes.
    """
    # Initialize the isolation feature selection
    global GEOPOLITICAL_SHOCK_SELECTION_DF
    return GEOPOLITICAL_SHOCK_SELECTION_DF.loc[(region), 'Intervention Selection']

def getGeopoliticalShockMatrixBool(all_other_regions, n_columns):
    global GEOPOLITICAL_SHOCK_SELECTION_DF

    matrix_bool = pd.DataFrame(1, index=all_other_regions, columns=np.arange(n_columns))

    for region in all_other_regions:
        if getGeopoliticalShockSelection(region) == 'Geopolitical Shock':
            matrix_bool.loc[(region), :] = 0

    return matrix_bool

################################################################################
################################################################################
# Model Architecture Data Functions

def getYears():
    return years_df

def getYearsString():
    years_string = []
    for year in getYears().values:
        years_string.append(str(year[0]))

    return years_string

def getRegions():
    return regions_df

def getRegionsString():
    regions_string = []
    for region in getRegions().values:
        regions_string.append(region[0])

    return regions_string

def getApplications():
    return applications_df

def getApplicationsString():
    applications_string = []
    for application in getApplications().values:
        applications_string.append(application[0])

    return applications_string

def getTechnologies():
    return technologies_df

def getTechnologiesString():
    technologies_string = []
    for technology in getTechnologies().values:
        technologies_string.append(technology[0])

    return technologies_string

def getEmptyDataFrame(columns):
    """
    EMPTY DATA FRAME CREATION: establish a MultiIndexed DataFrame for storage of model outputs. A very specific numpy
    array must first be created, then converted to a Pandas DataFrame before a MultiIndex can be established. The
    MultiIndex allows for convenient parsing and locating of specific data subsets.
    The format for storing data will look as follows:

    YEAR    region  APPLICATION    BEV    FCV    HEV    ICE-D
    2018    US      Medium Haul    []     []     []     []
    2018    US      Long Haul      []     []     []     []
    2018    EU      Medium Haul    []     []     []     []
    2018    EU      Long Haul      []     []     []     []
    2018    China   Medium Haul    []     []     []     []
    2018    China   Long Haul      []     []     []     []
    2019    US      Medium Haul    []     []     []     []
    2019    US      Long Haul      []     []     []     []
     .      .           .          .      .      .      .
     .      .           .          .      .      .      .
     .      .           .          .      .      .      .

    Depending on the passed "columns" parameter, the column names will change.
    If the "columns" parameter is 'technologies', the returned storage dataFrame
    will look like the example above. If the "columns" parameter is 'investors'
    then the returend storage frame will look the same except the column names
    will be the investor number.

    NOTE: the MultiIndexed storage DataFrame will keep track of ADDED yearly capacity (NOT total cumulative capacity)

    :return: main_df (MultiIndexed DataFrame)
    """

     # FIRST: setup the year, region, and applicaiton dimensions
     # Create a column of years depending on # of Regions and # of applications
    year_interval = getRegions().size * getApplications().size
    year_array = np.array([], dtype=int)
    region_interval = getApplications().size
    region_array = np.array([])
    application_array = np.array([])
    for currYear in getYears().index:
         # Add to the years array
        curr_year_array = np.full(year_interval, getYears().loc[currYear])
        year_array = np.append(year_array, curr_year_array)

         # Loop through the number of Regions
        for currregion in getRegions().index:
             # Add to the region array
            curr_region_array = np.full(region_interval, getRegions().loc[currregion])
            region_array = np.append(region_array, curr_region_array)

             # Add to the application array
            application_array = np.append(application_array, getApplications())

    # Concatenate the previously created columns
    data = np.column_stack((year_array, region_array, application_array))
             # Construct a DataFrame from the larger 2D data
    df = pd.DataFrame(data, columns=['YEAR', 'REGION', 'APPLICATION'])


    # THEN: add in either the technologies or investors as additional columns
    if columns == 'technologies':
         # Now add in columns for each drive technology
        for currTechnology in getTechnologies().values:
             # NOTE: can also use df.insert() function which allows column insertion at a certain specified index
            df[currTechnology[0]] = np.zeros((df.shape[0], 1))

         # Now create the MultiIndex DataFrame
        df.set_index(['YEAR', 'REGION', 'APPLICATION'], inplace=True)
         # Sort the multiIndex (this sorts each level by alphanumeric order)
        df.sort_index(inplace=True)
         # Convert the DataFrame to dtype float64
        df = df.astype(object)

         # Assign df to the empty data frame variable
        df_empty = df

    elif columns == 'investors':

         # Now add in columns for each investor
        for currInvestor in range(1,NUM_INVESTORS+1):
             # NOTE: can also use df.insert() function which allows column insertion at a certain specified index
            df[currInvestor] = np.zeros((df.shape[0], 1))

         # Now create the MultiIndex DataFrame
        df.set_index(['YEAR', 'REGION', 'APPLICATION'], inplace=True)
         # Sort the multiIndex (this sorts each level by alphanumeric order)
        df.sort_index(inplace=True)
         # Convert the DataFrame to dtype float64
        df = df.astype(object)

        # Assign df to the empty data frame variable
        df_empty = df

    elif columns == 'learning components':

         # Now add in columns for each learning component
        for currLearningComponent in getLearningComponents().values:
             # NOTE: can also use df.insert() function which allows column insertion at a certain specified index
            df[currLearningComponent] = np.zeros((df.shape[0], 1))

         # Now create the MultiIndex DataFrame
        df.set_index(['YEAR', 'REGION', 'APPLICATION'], inplace=True)
         # Sort the multiIndex (this sorts each level by alphanumeric order)
        df.sort_index(inplace=True)
         # Convert the DataFrame to dtype float64
        df = df.astype(object)

        # Assign df to the empty data frame variable
        df_empty = df

    elif columns == 'statistics':
        df_empty = df

    return df_empty

def getMonetaryParameters(region, parameter):
    """
    This function takes in a region and a parameter (VAT, OEM Profit Margin)
    and returns a specific corresponding integer value.
    """
    return monetary_parameters_data_df.loc[(region), parameter]

def getVehicleSalesForecast(year, region, application):
    """
    This function returns the market forecast for the given year, region, and
    application. The returned value is an integer.
    """
    return vehicle_sales_forecast_data_df.loc[(region, application), year]

def getApplicationData(region, application, parameter):
    """
    This function takes in a region, application and parameter (weight, range,
    payload % of GVW, loading % of payload) and returns the [min, avg, max]
    as an integer.
    """

    return application_data_df.loc[(region, application), parameter]

def getTechnologyData(technology, application, parameter):
    """
    This function takes in a techology and applicaiton and returns the specified
    parameter as an integer.
    """

    return technology_data_df.loc[(technology, application), parameter]

def getCAPEXBoolean(technology):
    """
    This function takes in a technology and returns the associated
    CAPEX boolean values for specific vehicle components for the given technology.
    """

    return tech_boolean_df.loc[(technology),:]

def getSwitchingCostFeatureBoolean():
    return SWITCHING_COST_FEATURE_BOOL[0]

def getSwitchingCostData(parameter, start_year, end_year, region, application, technology):

    if parameter == 'Initial Market Share':
        return INITIAL_MARKET_SHARES_DF.loc[(region, application, technology), start_year:end_year]
    elif parameter == 'Threshold':
        return SWITCHING_THRESHOLD_DF.loc[(region), 'Threshold']
    elif parameter == 'Threshold - BOOL':
        return SWITCHING_THRESHOLD_BOOL_DF.loc[(region, application), technology]
    elif parameter == 'Switching Cost Markup':
        return SWITCHING_COST_MARKUP_DF.loc[(region, application), technology]

def setSwitchingThresholdBool(bool, region, application, technology):
    SWITCHING_THRESHOLD_BOOL_DF.loc[(region, application), technology] = bool

def getDynamicVehicleParametersData(region, application, parameter):
    """
    This function takes in a region, application and performance parameter
    and returns the specified value as an integer.
    """

    return dynamic_vehicle_parameters_data_df.loc[(region, application), parameter]

def getDriveCycleSelection(region, application):
    """
    This function takes in a region and an application (weight specific) and returns
    the associated drive cycle selection.
    """

    return drive_cycle_selection_data_df.loc[(region, getApplicationSegment(application, 'weight')), "Drive Cycle Selection"]

def getDriveCycleData(drive_cycle, parameter):
    """
    This function takes in one of two parameters: time and velocity. It then
    returns the entire array associated with the passed parameter for the specified
    vehicle drive cycle.
    """
    if drive_cycle == 1:
        return cycle_1_df.loc[:,parameter]
    if drive_cycle == 2:
        return cycle_2_df.loc[:,parameter]
    if drive_cycle == 3:
        return cycle_3_df.loc[:,parameter]
    if drive_cycle == 4:
        return cycle_4_df.loc[:,parameter]
    if drive_cycle == 5:
        return cycle_5_df.loc[:,parameter]
    if drive_cycle == 6:
        return cycle_6_df.loc[:,parameter]

def getDriveCycleModeData(drive_cycle, application):
    """
    This function takes in a drive cycle and an application (range) and returns the
    associated cycle "modes" that the model will run for the application-cycle pairing.
    """

    modes = drive_cycle_mode_selection_data_df.loc[(drive_cycle, getApplicationSegment(application, 'range')), :].values

    # Remove any elements in the modes array that has a dash (meaning there is one less mode for this specific drive cycle)
    modes = modes[modes!='-']

    return modes

def getFuelSelection(region, application, technology):
    weight_segment = getApplicationSegment(application, 'weight')

    return fuel_selection_data_df.loc[(region, weight_segment), technology]

def getWTTEmissionsFactorsData(region, fuel_type, year):

    return wtt_emission_factors_data_df.loc[(region, fuel_type), year]

def getTTWEmissionsFactorsData(application, technology, parameter):
    weight_segment = getApplicationSegment(application, 'weight')

    return ttw_emission_factors_data_df.loc[(weight_segment, technology), parameter]

################################################################################
################################################################################
# CAPEX Data Functions

def getLearningComponents():
    return experience_curve_scenarios_df.index

def getLearningComponentsString():
    learning_components_string = []
    for learning_component in getLearningComponents().values:
        learning_components_string.append(learning_component)

    return learning_components_string

def getExperienceComponentScenario(learning_component,scenario_type):
    """
    This function takes in a learning component and and returns the modelling scenario
    for the experience rate or the exogenous capacity additions, depending on
    the scenario_type.
    """
    return experience_curve_scenarios_df.loc[(learning_component), scenario_type]

def getExperienceComponentBaseCapacity(learning_component):
    """
    This function takes in a learning component and and returns the base Cummulative
    capacity additions as an integer.
    """

    if 'Integration Factor' in learning_component:
        application = getIntegrationFactorComponentApp(learning_component)
    else:
        application = 'LDV'         # Here we set the application abitrarily to LDV for learning components which do NOT differentiate the cummulative installed capacity by application segment

    return experience_component_data_df.loc[(learning_component, application), 'Base Year Cap']

def getExperienceComponentBoolean(learning_component, vehicle_component):
    """
    This function takes in a learning component and vehicle component type and returns
    the associatd boolean which would indicate whether or not the learning component matches
    with the vehicle component type.
    """
    return experience_component_boolean_df.loc[(learning_component), vehicle_component]

def getExperienceComponentData(component, application, parameter):
    """
    This function takes in a learning component and a data parameter and returns
    the associated experience curve data parameter.
    """

    return experience_component_data_df.loc[(component, application), parameter]

def getIntegrationFactorComponentApp(learning_component):
    """
    This function takes in an integration factor learning component and returns
    the associated application segment (LDV or MDV/HDV).
    """

    [component, technology, application] = learning_component.split(' - ')

    if application == 'MDV/HDV':
        application = 'MDV'

    return application

def technologyIntegrationFactorBool(technology):
    """
    This function takes in a technology and returns a bool (true/false) if the
    passed technology has an experience component integration factor.
    """

    # Get the list of integration factor experience components
    learning_components = getLearningComponentsString()
    integration_factor_components = [i for i in learning_components if 'Integration Factor' in i]

    # Then check to see if the passed technology has an integration factor
    if any(technology in string_i for string_i in integration_factor_components):
        return True
    else:
        return False

def getTopUpDownData(learning_component, region, year):
    """
    This function takes in a learning component, region, and year and returns the
    associated top up/down percentage of the "base market" cost for the passed
    component in the passed geogrpahy and given year.
    """

    return top_up_down_df.loc[(learning_component, region), year]

def getInitialCostData():
    """
    This function returns all cost values of the learning components for all
    regions and applications. Mid, high, and low values are returned in a
    DataFrame.
    """

    return experience_component_data_df.loc[(slice(None), slice(None), slice(None)), ('Mid Cost', 'High Cost', 'Low Cost')]

def getChassisData(region, application, parameter):
    """
    This function takes in a region and an application and returns
    the associated vehicle chassis cost.
    """

    # First get only the weight segment of the application as that is what the wage is dependent upon
    application = getApplicationSegment(application, 'weight')

    return chassis_data_df.loc[(region, application), parameter]

def getExogenousMarketCapacityAdditions(learning_component, scenario, year):
    """
    This function takes in a learning component, scenario and year and returns the
    associated newly added capacity.
    """
    if year == 'all':
        return exogenous_market_capacity_data_df.loc[(learning_component, scenario), :]
    else:
        return exogenous_market_capacity_data_df.loc[(learning_component, scenario), year]

def getExogenousMarketContributions(component, region):
    """
    This function takes in a learning component and region and returns the
    exogenous market capacity addition contribution % for this combination of
    region and component.
    """

    return exogenous_market_contributions_df.loc[(component), region]

def getCAPEXSubsidyData(year, region, application, technology):
    """
    This function takes in a region, application, technology and year and returns
    the associated CAPEX subsidy.
    """

    # First get only the weight segment of the application as that is what the wage is dependent upon
    application = getApplicationSegment(application, 'weight')

    return capex_subsidies_df.loc[(region, application, technology), year]

################################################################################
################################################################################
# OPEX Data Functions

def getTolls(start_year, end_year, region, application, technology):
    """
    This function takes in a year, region, application, and technology and
    returns the associated toll value as an integer.
    """

    # First get only the weight segment of the application as that is what the wage is dependent upon
    application = getApplicationSegment(application, 'weight')
    # Check to see if the start year has no policy intervention (even if the later years have a policy intervention, the TCO should be based on the investor's policy knowledge in the start year)
    toll_in_start_year = tolls_df.loc[(region, application, technology), start_year]

    if toll_in_start_year > 0:
        tolls_df_return = toll_in_start_year
    else:
        tolls_df_return = pd.concat([tolls_df.loc[(region, application, technology), start_year:end_year].squeeze()]*NUM_INVESTORS, axis=1, ignore_index=True).transpose()
        tolls_df_return.columns = np.arange(end_year-start_year+1)

    return tolls_df_return

def getOandM(year, region, application, technology):
    """
    This function takes in a year, region, application, and technology and
    returns the associated O&M value as an integer.
    """

    # First get only the weight segment of the application as that is what the wage is dependent upon
    application = getApplicationSegment(application, 'weight')

    return o_and_m_df.loc[(region, application, technology), year]

def getInsurance(region, parameter):
    """
    This function takes in a region and a parameter and
    returns the associated insurance percentage of CAPEX value as an integer.
    """

    return insurance_df.loc[(region), parameter]

def getWages(region, application, parameter):
    """
    This function takes in a region and an application
    """

    # First get only the weight segment of the application as that is what the wage is dependent upon
    application = getApplicationSegment(application, 'weight')

    return wages_df.loc[(region, application), parameter]

def getFuelCosts(region, fuel_type, stats_param, start_year, end_year):
    """
    This function ...
    """

    return fuel_costs_df.loc[(region, fuel_type, stats_param), start_year:end_year:1]

def getFuelConsumptionData(technology, parameter, year):
    """
    This function takes in a technology and returns the associated parameters for
    the fuel consumption fit in the passed year.
    (current fit: logarithmic
    (current parameters: a, b)
    """

    return fuel_consumption_data_df.loc[(technology, parameter), year]

def getFuelConsumptionPercDiffData(region):
    """
    This function takes in a region and returns the associated fuel consumption
    percentage difference value as compared to the baseline.
    """

    return fuel_consumption_perc_diff_data_df.loc[(region), 'Fuel Consumption % Difference']

def getChargingInfrastructureData(region, power_level, parameter):
    """
    This function returns data associated with the charging infrastructure for BEVs.
    The function takes in a region, a power level (i.e. 7kW, 150kW) and a parameter (i.e. O&M, Charging Infrastructure Cost)
    in order to return the appropriate data.
    """

    return charging_infrastructure_data_df.loc[(region, power_level), parameter]

def getChargingInfrastructureSelection(region, application):
    """
    This function takes in a region and an application and returns the associated
    power level selection.
    """
    # First get only the weight segment of the application as that is what the wage is dependent upon
    application = getApplicationSegment(application, 'weight')

    return charging_infrastructure_selection_df.loc[(region, application), 'Power Level']

def takeClosest(myList, myNumber):
    """
    Code taken from : https://stackoverflow.com/questions/12141150/from-list-of-integers-get-number-closest-to-a-given-value
    ...
    Assumes myList is sorted. Returns closest value to myNumber.

    If two numbers are equally close, return the smallest number.
    """
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return myList[0]
    if pos == len(myList):
        return myList[-1]
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
        return after
    else:
        return before

def getScrapValue(weight, total_km):
    """
    This function takes in a region, specific vehicle weight, and the total
    kilometers travelled in the vehicle lifetime. Then the function returns the
    corresponding scrappage value based on an exponential function with
    parameters a and b associated with a specific vehicle weight and total mileage.
    """

    # Available weights to use for the exponential:
    available_weights = [3500, 7500, 12000, 18000, 26000, 40000]

    # For all modelled weights, assign one of the above determined weights
    assigned_weights = np.zeros(weight.size)
    scrap_values = np.zeros(weight.size)
    i=0
    for weight_i in weight.iteritems():
        # Determine the closest assigned weight
        closest_assigned_weight = takeClosest(available_weights, weight_i[1])
        assigned_weights[i] = closest_assigned_weight
        a = scrap_value_df.loc[(closest_assigned_weight), 'a']
        b = scrap_value_df.loc[(closest_assigned_weight), 'b']
        scrap_values[i] = a*np.exp(b*total_km[i]/1000)
        i += 1

    return scrap_values

def getOtherTCOParameters(region, application, parameter):
    """
    This function takes in a region and an application and returns all of the 'Other TCO Parameters'
    which include lifetime, cost of capital, annual working days, and % driven on toll roads. These
    parameters are specified in the 'TCO Data.xlsx' file in the '_Other TCO Parameters' sheet.
    """

    # First get only the weight segment of the application as that is what the wage is dependent upon
    application = getApplicationSegment(application, 'weight')

    return other_tco_parameters_df.loc[(region, application), parameter]

################################################################################
################################################################################
# Functions for code structure and organization

def getApplicationSegment(application, dimension):
    """
    This function takes in the application string and returns either the weight
    or range segement depending on the passed dimension parameter.
    """
    [weight_segment, range_segment] = application.split('-')
    if dimension == 'weight':
        # Parse the application string
        return weight_segment
    elif dimension == 'range':
        return range_segment

def getRangeApplicationSegments(weight_segment):
    """
    This function takes in a weight segment (LDV, MDV, HDV) and returns all weight-range
    specific application segments associated with the weight segment.
    """
    weight_segement_app_list = [i for i in getApplicationsString() if weight_segment in i]

    return weight_segement_app_list

def getCAPEXRecordParameters():
    """
    This function returns the pre-set sub-parameters of the initial purchase cost
    that should be recorded and stored for each code run. Depending on how much
    information is needed to be tracked, the stored parameters can be adjusted
    here.
    """

    return ["Power Train", "Energy Storage", "Rest of Truck"]

def getOPEXRecordParameters():
    """
    This function returns the pre-set sub-parameters of the annual operating cost
    that should be recorded and stored for each code run. Depending on how much
    information is needed to be tracked, the stored parameters can be adjusted
    here.
    """

    return ["Insurance", "O & M", "Tolls", "Wages", "Fuel Costs", "Infrastructure Costs", "Carbon Costs"]

def isZEVTechnology(technology):
    if technology == 'BEV' or technology == 'FCEV' or technology == 'OHC' or technology == 'ICE-BD':
        return True
    else:
        return False

def getZEVList():
    list = []
    for technology in getTechnologiesString():
        if isZEVTechnology(technology):
            list.append(technology)
    return list

def getNonZEVList():
    list = []
    for technology in getTechnologiesString():
        if not isZEVTechnology(technology):
            list.append(technology)
    return list

def distributionFunction(stat_params, type, size):
    """
    This function takes in min, max and avg values for a distribution of a certain indicated type.
    Depending on the type, the according distribution is computed and a random selection from this
    distribution is returned.

    stat_param order:
        ~ Triangular ... [0]: min, [1]: most likely, [2]: max
        ~ PERT ......... [0]: min, [1]: most likley, [2]: max
        ~ Normal ....... [0]: mean, [1]: std
        ~ Beta ......... [0]: alpha, [1]: beta
        ~ Uniform ...... [0]: min, [1]: max
    """

    if type == 'Triangular':
        try:
            random_value = np.random.triangular(stat_params[0], stat_params[1], stat_params[2], size)
        except:
            random_value = stat_params[1]*np.ones(size)
    if type == 'PERT':
        try:
            #random_value = PERT(stat_params[0], stat_params[1], stat_params[2], size)
            pert = PERT(stat_params[0],stat_params[1],stat_params[2])
            random_value = pert.rvs(size)
        except:
            random_value = stat_params[1]*np.ones(size)
    if type == 'Normal':
        try:
            random_value = np.random.normal(stat_params[0], stat_params[1], size)
        except:
            random_value = stat_params[0]*np.ones(size)
    if type == 'Beta':
        try:
            random_value = np.random.beta(stat_params[0], stat_params[1], size)                       # NOTE: when passing parameters to distributionFunction for the BETA distribution, the max and min values are taken as the alpha and beta distribution parameters respectively
        except:
            random_value = stat_params[0]*np.ones(size)
    elif type == 'Uniform':
        try:
            random_value = np.random.uniform(stat_params[0], stat_params[1], size)
        except:
            random_value = stat_params[0]

    return random_value

def getParameterType(mode):
    """
    This function takes in the "mode" and returns whether or not to treat the
    mode parameters as DISTRIBUTED or CONSTANT.
    """
    # CONSTANT OR DISTRIBTUED
    if mode == 'capex':
        global CAPEX_PARAMETERS_TYPE
        return CAPEX_PARAMETERS_TYPE.lower()
    elif mode == 'opex':
        global OPEX_PARAMETERS_TYPE
        return OPEX_PARAMETERS_TYPE.lower()
    elif mode == 'PerformanceCharacteristics':
        global USE_CASE_PARAMETERS_TYPE
        return USE_CASE_PARAMETERS_TYPE.lower()

#
