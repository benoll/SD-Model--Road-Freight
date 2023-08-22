# FILE: intermediateOutputs.py
# PROJECT: Global System-Dynamics Freight Model
# MODULE DESCRIPTION: This is the intermediate storage output module. Following
# each iteration step, data will be summed/collected/managed
# and stored in this module.


# Import statements
import pandas as pd
import numpy as np
import externalParameters as ep

# Global constants
BASE_YEAR = ep.getYears()['Years'][0]

################################################################################
# EXTERNAL FUNCTIONS TO THE MODULE:
################################################################################

def initialize():
    """
    This function initializes all relevant global parameters used in the intermediateOutputs.py
    module.
    """

    # Initialize the global variables
    global TCO_DF
    global CAPEX_DF
    global OPEX_DF
    global TCO_PARAMETERS_DF
    global CAPEX_PARAMETERS_DF
    global OPEX_PARAMETERS_DF
    global CAPEX_PARAMETERS_PERCENTAGES_DF
    global MARKET_SHARES_DF
    global MAIN_OUTPUT_DF
    global SWITCHING_COST_MULTIPLIER_DF
    global SWITCHING_COST_DF

    # Initialize the empty storage frames for the global variables
    MARKET_SHARES_DF = ep.getEmptyDataFrame('technologies')
    MAIN_OUTPUT_DF = ep.getEmptyDataFrame('technologies')
    TCO_DF = ep.getEmptyDataFrame('technologies')
    CAPEX_DF = ep.getEmptyDataFrame('technologies')
    OPEX_DF = ep.getEmptyDataFrame('technologies')
    TCO_PARAMETERS_DF = ep.getEmptyDataFrame('technologies')
    CAPEX_PARAMETERS_DF = ep.getEmptyDataFrame('technologies')
    OPEX_PARAMETERS_DF = ep.getEmptyDataFrame('technologies')
    CAPEX_PARAMETERS_PERCENTAGES_DF = ep.getEmptyDataFrame('technologies')
    SWITCHING_COST_MULTIPLIER_DF = ep.getEmptyDataFrame('technologies')
    SWITCHING_COST_DF = ep.getEmptyDataFrame('technologies')

def recordTCO(year, region, application, technology, values_array):
    """
    This function records the calculated TCO values for a specific technology in a
    specific year, region and application.
    """
    TCO_DF.at[(year, region, application), technology] = values_array           # NOTE: order of values array : [0]=mean, [1]=min, [2]=max, [3]=std

def recordCAPEX(year, region, application, technology, values_array):
    """
    This function records the calculated CAPEX values for a specific technology in a
    specific year, region and application.
    """
    CAPEX_DF.at[(year, region, application), technology] = values_array           # NOTE: order of values array : [0]=mean, [1]=min, [2]=max, [3]=std

def recordOPEX(year, region, application, technology, values_array):
    """
    This function records the calculated OPEX values for a specific technology in a
    specific year, region and application.
    """
    OPEX_DF.at[(year, region, application), technology] = values_array           # NOTE: order of values array : [0]=mean, [1]=min, [2]=max, [3]=std

def recordTCOParameters(year, region, application, technology, values_array):
    """
    This function records the calculated TCO Parameter values for a specific technology in a
    specific year, region and application. TCO Parameters include: all CAPEX parameters,
    all OPEX parameters, CAPEX subsidy, switching cost, and scrappage value.
    """
    TCO_PARAMETERS_DF.at[(year, region, application), technology] = values_array           # NOTE: order of values array : [0]=mean, [1]=std

def recordCAPEXParameters(year, region, application, technology, values_array):
    """
    This function records the calculated CAPEX Parameter values for a specific technology in a
    specific year, region and application. CAPEX Parameters include: powertrain, energy storage,
    and rest of truck.
    """
    CAPEX_PARAMETERS_DF.at[(year, region, application), technology] = values_array           # NOTE: order of values array : [0]=mean, [1]=std

def recordOPEXParameters(year, region, application, technology, values_array):
    """
    This function records the calculated OPEX Parameter values for a specific technology in a
    specific year, region and application. CAPEX Parameters include: insurance, O & M, tolls,
    wages, fuel costs, infrastructure costs, and carbon costs.
    """
    OPEX_PARAMETERS_DF.at[(year, region, application), technology] = values_array           # NOTE: order of values array : [0]=mean, [1]=std

def recordMarketShares(year, region, application, value_array):
    """
    This function records the calculated OPEX values for all technologies in a
    specific year, region and application.
    """
    MARKET_SHARES_DF.at[(year, region, application), :] = value_array                   # NOTE: value_array = a percentage market share of each drive-technology
    MARKET_SHARES_DF.replace(np.nan, 0, inplace=True)

def recordMainOutput(year, year_df):
    """
    This function records the calculated main output (i.e. number of vehicles) for
    all technologies in all regions and application in a specific year.
    """
    MAIN_OUTPUT_DF.loc[(year, slice(None), slice(None)), :] = year_df

def recordSwitchingCostMultiplier(year, region, application, technology, value):
    """
    This function records the switching cost multiplier values for a specific technology in a
    specific year, region and application.
    """
    SWITCHING_COST_MULTIPLIER_DF.at[(year, region, application), technology] = value

def recordSwitchingCost(year, region, application, technology, value):
    """
    This function records the switching cost values (absolute) for a specific technology in a
    specific year, region and application.
    """
    SWITCHING_COST_DF.at[(year, region, application), technology] = value

def getMarketShares(start_year, end_year, region, application, technology):
    """
    This function returns computed market shares for a given range of years for a specific
    region, application and technology.
    """

    if '-' in application:
        return MARKET_SHARES_DF.loc[(slice(start_year,end_year), region, application),technology]
    else:
        application_list = ep.getRangeApplicationSegments(application)
        # Find the shares for an entire weight segment
        new_trucks_tech_i = MAIN_OUTPUT_DF.loc[(slice(start_year,end_year), region, application_list),technology].groupby(['YEAR', 'REGION']).sum().values
        new_trucks_total = ep.getVehicleSalesForecast(slice(start_year,end_year), region, application_list).sum().values

        # Divide to get the market shares for the segment
        application_segment_shares = new_trucks_tech_i/new_trucks_total

        return application_segment_shares

def storeICEDCAPEX(array):
    """
    This function stores the ICE-D CAPEX value for the current iteration.
    """
    global ICE_D_CAPEX_STORE_i
    ICE_D_CAPEX_STORE_i = array

def getICEDCAPEX():
    """
    This function returns the ICE-D CAPEX value for the current iteration.
    """
    global ICE_D_CAPEX_STORE_i
    return ICE_D_CAPEX_STORE_i

def storeTCODataframe():
    """
    This function stores the TCO mean and standard deviation dataframes.
    """
    TCO_DF_MEAN = ep.getEmptyDataFrame('technologies')
    TCO_DF_STD = ep.getEmptyDataFrame('technologies')
    TCO_DF_MEAN = TCO_DF_MEAN.apply(separateStatisticsArray,axis=1, args=[0, TCO_DF.copy()])        # NOTE: 0 implies the "mean" statistic
    TCO_DF_STD = TCO_DF_STD.apply(separateStatisticsArray,axis=1, args=[3, TCO_DF.copy()])        # NOTE: 3 implies the "std" statistic
    TCO_DF_MEAN.to_pickle(ep.getFinalOutputDirectory() + '\_TCO_DF_MEAN.pkl')
    TCO_DF_MEAN.to_excel(ep.getFinalOutputDirectory() + '\_tco_mean.xlsx')
    TCO_DF_STD.to_pickle(ep.getFinalOutputDirectory() + '\_TCO_DF_STD.pkl')
    TCO_DF_STD.to_excel(ep.getFinalOutputDirectory() + '\_tco_std.xlsx')

def storeCAPEXDataframe():
    """
    This function stores the CAPEX mean and standard deviation dataframes.
    """
    CAPEX_DF_MEAN = ep.getEmptyDataFrame('technologies')
    CAPEX_DF_STD = ep.getEmptyDataFrame('technologies')
    CAPEX_DF_MEAN = CAPEX_DF_MEAN.apply(separateStatisticsArray,axis=1, args=[0, CAPEX_DF.copy()])        # NOTE: 0 implies the "mean" statistic
    CAPEX_DF_STD = CAPEX_DF_STD.apply(separateStatisticsArray,axis=1, args=[3, CAPEX_DF.copy()])        # NOTE: 3 implies the "std" statistic
    CAPEX_DF_MEAN.to_pickle(ep.getFinalOutputDirectory() + '\_CAPEX_DF_MEAN.pkl')
    CAPEX_DF_MEAN.to_excel(ep.getFinalOutputDirectory() + '\_capex_mean.xlsx')
    CAPEX_DF_STD.to_pickle(ep.getFinalOutputDirectory() + '\_CAPEX_DF_STD.pkl')
    CAPEX_DF_STD.to_excel(ep.getFinalOutputDirectory() + '\_capex_std.xlsx')

def storeOPEXDataframe():
    """
    This function stores the OPEX mean and standard deviation dataframes.
    """
    OPEX_DF_MEAN = ep.getEmptyDataFrame('technologies')
    OPEX_DF_STD = ep.getEmptyDataFrame('technologies')
    OPEX_DF_MEAN = OPEX_DF_MEAN.apply(separateStatisticsArray,axis=1, args=[0, OPEX_DF.copy()])        # NOTE: 0 implies the "mean" statistic
    OPEX_DF_STD = OPEX_DF_STD.apply(separateStatisticsArray,axis=1, args=[3, OPEX_DF.copy()])        # NOTE: 3 implies the "std" statistic
    OPEX_DF_MEAN.to_pickle(ep.getFinalOutputDirectory() + '\_OPEX_DF_MEAN.pkl')
    OPEX_DF_MEAN.to_excel(ep.getFinalOutputDirectory() + '\_opex_mean.xlsx')
    OPEX_DF_STD.to_pickle(ep.getFinalOutputDirectory() + '\_OPEX_DF_STD.pkl')
    OPEX_DF_STD.to_excel(ep.getFinalOutputDirectory() + '\_opex_std.xlsx')

def storeTCOParametersDataframe():
    """
    This function stores the TCO parameters mean and standard deviation dataframes.
    """
    # Store the unorganized df
    TCO_PARAMETERS_DF.to_pickle(ep.getFinalOutputDirectory() + '\_TCO_PARAMETERS_DF.pkl')
    TCO_PARAMETERS_DF.to_excel(ep.getFinalOutputDirectory() + '\_tco_parameters.xlsx')
    # Then organize
    TCO_PARAMETERS_DF_MEAN = separateParametersArray(TCO_PARAMETERS_DF, 0)
    TCO_PARAMETERS_DF_STD = separateParametersArray(TCO_PARAMETERS_DF, 1)
    # And store the organized df's
    TCO_PARAMETERS_DF_MEAN.to_pickle(ep.getFinalOutputDirectory() + '\_TCO_PARAMETERS_DF_MEAN.pkl')
    TCO_PARAMETERS_DF_MEAN.to_excel(ep.getFinalOutputDirectory() + '\_tco_parameters_mean.xlsx')
    TCO_PARAMETERS_DF_STD.to_pickle(ep.getFinalOutputDirectory() + '\_TCO_PARAMETERS_DF_STD.pkl')
    TCO_PARAMETERS_DF_STD.to_excel(ep.getFinalOutputDirectory() + '\_tco_parameters_std.xlsx')

def storeCAPEXParametersDataframe():
    """
    This function stores the CAPEX parameters mean and standard deviation dataframes.
    """
    # Organize
    CAPEX_PARAMETERS_DF_MEAN = separateParametersArray(CAPEX_PARAMETERS_DF, 0)
    CAPEX_PARAMETERS_DF_STD = separateParametersArray(CAPEX_PARAMETERS_DF, 1)
    # Store
    CAPEX_PARAMETERS_DF_MEAN.to_pickle(ep.getFinalOutputDirectory() + '\_CAPEX_PARAMETERS_DF_MEAN.pkl')
    CAPEX_PARAMETERS_DF_MEAN.to_excel(ep.getFinalOutputDirectory() + '\_capex_parameters_mean.xlsx')
    CAPEX_PARAMETERS_DF_STD.to_pickle(ep.getFinalOutputDirectory() + '\_CAPEX_PARAMETERS_DF_STD.pkl')
    CAPEX_PARAMETERS_DF_STD.to_excel(ep.getFinalOutputDirectory() + '\_capex_parameters_std.xlsx')

def storeOPEXParametersDataframe():
    """
    This function stores the OPEX parameters mean and standard deviation dataframes.
    """
    # Organize
    OPEX_PARAMETERS_DF_MEAN = separateParametersArray(OPEX_PARAMETERS_DF, 0)
    OPEX_PARAMETERS_DF_STD = separateParametersArray(OPEX_PARAMETERS_DF, 1)
    # Store
    OPEX_PARAMETERS_DF_MEAN.to_pickle(ep.getFinalOutputDirectory() + '\_OPEX_PARAMETERS_DF_MEAN.pkl')
    OPEX_PARAMETERS_DF_MEAN.to_excel(ep.getFinalOutputDirectory() + '\_opex_parameters_mean.xlsx')
    OPEX_PARAMETERS_DF_STD.to_pickle(ep.getFinalOutputDirectory() + '\_OPEX_PARAMETERS_DF_STD.pkl')
    OPEX_PARAMETERS_DF_STD.to_excel(ep.getFinalOutputDirectory() + '\_opex_parameters_std.xlsx')

def storeMarketSharesDataframe():
    """
    This function stores the market shares dataframe.
    """
    MARKET_SHARES_DF.to_pickle(ep.getFinalOutputDirectory() + '\_MARKET_SHARES_DF.pkl')
    MARKET_SHARES_DF.to_excel(ep.getFinalOutputDirectory() + '\_market_shares.xlsx')

def storeSwitchingCostMultiplierDataframe():
    """
    This function stores the switching cost multiplier dataframe.
    """
    SWITCHING_COST_MULTIPLIER_DF.to_pickle(ep.getFinalOutputDirectory() + '\_SWITCHING_COST_MULTIPLIER_DF.pkl')
    SWITCHING_COST_MULTIPLIER_DF.to_excel(ep.getFinalOutputDirectory() + '\_switching_cost_multiplier.xlsx')

def storeSwitchingCostDataframe():
    """
    This function stores the switching cost (absolute) dataframe.
    """
    SWITCHING_COST_DF.to_pickle(ep.getFinalOutputDirectory() + '\_SWITCHING_COST_DF.pkl')
    SWITCHING_COST_DF.to_excel(ep.getFinalOutputDirectory() + '\_switching_cost.xlsx')

def printTCO(year, region, application, technology):
    """
    This function prints the TCO values for a given year, region, application and technology.
    """
    print('For the following segment: [', year, region, application, technology, '] the "TCO" values are:')
    print('Mean: ', np.around(TCO_DF.loc[(year, region, application), technology][0], decimals=2))
    print('Min: ', np.around(TCO_DF.loc[(year, region, application), technology][1], decimals=2))
    print('Max: ', np.around(TCO_DF.loc[(year, region, application), technology][2], decimals=2))
    print('Std: ', np.around(TCO_DF.loc[(year, region, application), technology][3], decimals=2))

def printCAPEX(year, region, application, technology):
    """
    This function prints the CAPEX values for a given year, region, application and technology.
    """
    print('For the following segment: [', year, region, application, technology, '] the "CAPEX" values are:')
    print('Mean: ', np.around(CAPEX_DF.loc[(year, region, application), technology][0], decimals=2))
    print('Min: ', np.around(CAPEX_DF.loc[(year, region, application), technology][1], decimals=2))
    print('Max: ', np.around(CAPEX_DF.loc[(year, region, application), technology][2], decimals=2))
    print('Std: ', np.around(CAPEX_DF.loc[(year, region, application), technology][3], decimals=2))

def printOPEX(year, region, application, technology):
    """
    This function prints the OPEX values for a given year, region, application and technology.
    """
    print('For the following segment: [', year, region, application, technology, '] the "OPEX" values are:')
    print('Mean: ', np.around(OPEX_DF.loc[(year, region, application), technology][0], decimals=2))
    print('Min: ', np.around(OPEX_DF.loc[(year, region, application), technology][1], decimals=2))
    print('Max: ', np.around(OPEX_DF.loc[(year, region, application), technology][2], decimals=2))
    print('Std: ', np.around(OPEX_DF.loc[(year, region, application), technology][3], decimals=2))

def printCAPEXParameters(year, region, application, technology):
    """
    This function prints the CAPEX parameter values for a given year, region, application and technology.
    """
    print('For the following segment: [', year, region, application, technology, '] the "CAPEX Parameter" values are:')
    print('-------------------------')
    print('Mean: ')
    print(np.around(CAPEX_PARAMETERS_DF.loc[(year, region, application), technology][0], decimals=2))
    print('-------------------------')
    print('Std: ')
    print(np.around(CAPEX_PARAMETERS_DF.loc[(year, region, application), technology][1], decimals=2))
    print('-------------------------')

def printOPEXParameters(year, region, application, technology):
    """
    This function prints the OPEX parameter values for a given year, region, application and technology.
    """
    print('For the following segment: [', year, region, application, technology, '] the "OPEX Parameter" values are:')
    print('-------------------------')
    print('Mean: ')
    print(np.around(OPEX_PARAMETERS_DF.loc[(year, region, application), technology][0], decimals=2))
    print('-------------------------')
    print('Std: ')
    print(np.around(OPEX_PARAMETERS_DF.loc[(year, region, application), technology][1], decimals=2))
    print('-------------------------')

"""
###########################################################
# NOTE:
#
# If you want to select a specific parameter from the OPEX or CAPEX parameters DF's, you can do so as follows:
#
#           CAPEX_PARAMETERS_DF.loc[(year, region, application), technology][1]['Power Train']
#           OPEX_PARAMETERS_DF.loc[(year, region, application), technology][0]['Tolls']
#
# Which would give the std of the Power Train and the mean of the tolls.
############################################################
"""

################################################################################
# INTERNAL FUNCTIONS TO THE MODULE:
################################################################################

def separateStatisticsArray(technology_row, parameter, df_ref):
    """
    This function takes in a dataframe of all technologies in a specific year, region,
    and application and separates out the statistics array (avg, high, low, std) that
    occupies the technology-specific frame value.

    Depending on the passed 'parameter' variable, the function will return the associated
    value as a single value in the dataframe.
    """

    # Initialize the year, region and application
    year = technology_row.name[0]
    region = technology_row.name[1]
    application = technology_row.name[2]


    for technology in ep.getTechnologiesString():
        separated_parameter = df_ref.loc[(year, region, application), technology][parameter]
        technology_row[technology] = separated_parameter

    return technology_row

def separateParametersArray(df_pass, statistic):
    """
    This function separates out the TCO parameters (components of the TCO) into
    the multiindex dataframe for easy storing and data management post code run.

    statistic: 0 indicates mean, 1 indicates std
    """

    # First get the indexs of the passed dataframe (which may vary depending on the code run)
    year_list = df_pass.index.levels[0]
    region_list = df_pass.index.levels[1]
    application_list = df_pass.index.levels[2]
    technology_list = df_pass.loc[(year_list[0],region_list[0],application_list[0])].index
    parameter_list = df_pass.loc[(year_list[0],region_list[0],application_list[0]), technology_list[0]][0].index

    # Then create a new multiindex dataframe with the additional parameters
    df_return = pd.DataFrame(data=1, index=df_pass.index, columns=parameter_list).stack()
    df_return.index.names = ['YEAR','REGION','APPLICATION','PARAMETER']
    df_return = df_return.to_frame()
    df_return[technology_list] = 0
    df_return.drop([0],axis=1, inplace=True)

    # Loop through each level and insert the proper values from the passed dataframe
    for technology in technology_list:
        for year in year_list:
            for region in region_list:
                for application in application_list:

                    df_return.loc[(year,region,application,slice(None)), technology] = df_pass.loc[(year,region,application), technology][statistic].values

    return df_return




#
