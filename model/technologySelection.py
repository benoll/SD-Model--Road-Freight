# FILE: technologySelection.py
# PROJECT: System-Dynamics Global Road Freight Model
# MODULE DESCRIPTION: This technology selection module will simulate investor decisions/
# selection of competing technologies via a Monte Carlo analysis in a given year, region,
# and application.


# Import statements
import pandas as pd
import numpy as np
import externalParameters as ep
import totalCostOfOwnership as tco
import intermediateOutputs as io

# Global constants
YEAR = None
REGION = None
APPLICATION = None

################################################################################
# EXTERNAL FUNCTIONS TO THE MODULE:
################################################################################

def investorSimulation(technology_series):
    """
    This function takes in a section of the main Pandas dataframe (initialized and
    stored in main.py), which is a series of all modelled technologies for a given
    year, region and application. The function then organizes a Monte Carlo simulation
    of N investors where a total cost of ownership (TCO) is calculated for each technology
    and for each investor. This process is outlined below in sequential steps.

    Returned at the end of the function are the number of vehicles of each technology type
    that were probabalistically 'selected' by investors in the given year, region and application.
    """
    # Initialize the year, region, and application within the module
    setState(technology_series.name)

    ##################################################
    # Simulate Monte Carlo Investor Selection Process:
    ##################################################

    # 1): Setup an empty dataframe to organize the number of investors
    num_investors = ep.getNumInvestors()
    investor_frame = pd.DataFrame(index=np.arange(1, num_investors+1), columns=technology_series.index, dtype=np.float64)                   # NOTE: #investors x #technologies    (dataframe)

    # 2): Calculate the total cost of ownership (TCO) for all competing technologies for each investor
    investor_frame = investor_frame.apply(tco.getCost, args=[YEAR, REGION, APPLICATION])            # (type: DataFrame) NOTE: we are calling the apply function along the "0" axis which means a technology specific series will be passed to the next module

    # 3): Determine the technology selection for each investor based on minimum cost (this code line finds the "index" (technology as a string) of the lowest cost technology)
    # INTERVENTION CHECK: Check first to see if a private intervention is imposed, specifically if the EV100 Fleet Committment corporate intervention is imposed
    if 'EV 100 Fleet Commitment' in ep.getPrivateInterventionSelection(REGION):
        # Find the % of vehicle sales committed to be ZEVs as per the EV100 member company corporate committments for the specific year, region and application
        zev_sales_perc_commitment = ep.getPrivateInterventionData(YEAR, REGION, ep.getApplicationSegment(APPLICATION, 'weight'))
        # Determine how many investors are mandated to purchase ZEVs
        num_investors_commited = num_investors*zev_sales_perc_commitment
        # Single out the corresponding % of investors and only let them choose from ZEV options
        investor_selections_zev = investor_frame[0:num_investors_commited.astype(int)][ep.getZEVList()].idxmin(axis=1)
        # Then let the rest choose from all available technology options
        investor_selections_other = investor_frame[num_investors_commited.astype(int):].idxmin(axis=1)
        # Finally, combine the two investor selection arrays
        investor_selections = pd.concat([investor_selections_zev, investor_selections_other])         # NOTE: #investors x 1   (series)
    else:
        # Investors can select from all competing technologies
        investor_selections = investor_frame.idxmin(axis=1)                                           # NOTE: #investors x 1   (series)

    # 4): Calculate the percentage market share of the selected technologies (the result of this line generates a series with technologies in the rows and the normalized technology selection share in the column)
    market_shares = investor_selections.value_counts(normalize=True)                                   # NOTE: #selected technologies x 1 (series)
    # Record the market share in the intermediate outputs module
    io.recordMarketShares(YEAR, REGION, APPLICATION, market_shares)

    # 5): Add to the technology_series storage the selected market shares
    for i in market_shares.index:
        technology_series.at[i] = market_shares.loc[i]

    # 6): Multiply by the vehicle sales forecast to get the added technology-specific vehicle increase for this specific year, region, application
    vehicle_sales_forecast = ep.getVehicleSalesForecast(YEAR, REGION, APPLICATION)
    technology_series = technology_series.multiply(vehicle_sales_forecast)

    return technology_series

##############################################################
# SET THE STATE OF THE MODULE:
##############################################################

def setState(state):
    """
    This function sets the state in the current module. The state encompasses all global variables listed below.
    """
    global YEAR
    YEAR = state[0]
    global REGION
    REGION = state[1]
    global APPLICATION
    APPLICATION = state[2]

def getState():
    """
    This function returns the current state, as in what year, region, application the module is iterating
    over. The function returns a python list with the state variables in the order listed below.
    """

    return [YEAR, REGION, APPLICATION]

def printState():
    """
    This function prints the current state, as in what year, region and application the module is in.
    """
    state_string = "Current State: [" + str(YEAR) + ", " + REGION + ", " + APPLICATION + "]"

    print(state_string)
