# FILE: totalCostOfOwnership.py
# PROJECT: Global System-Dynamics Freight Model
# MODULE DESCRIPTION: This is the Total Cost of Ownership (TCO) Module that computes the cost of a specific technology in a specific year, region,
# and application.

# Import statements
import pandas as pd
import numpy as np
import externalParameters as ep
import intermediateOutputs as io
import useCaseCharacterization as ucc
import capex as cpx
import opex as opx


# Global constants
NUM_INVESTORS = ep.getNumInvestors()
BASE_YEAR = ep.getYears()['Years'][0]

################################################################################
# EXTERNAL FUNCTIONS TO THE MODULE:
################################################################################

def getCost(investor_series, year, region, application):
    """
    Passed into this function is an "investor_series" which is a Pandas series with number of
    investors in the rows and the specific technology as the one column. In order to perform
    the Monte Carlo analysis, a TCO cost for a given technology in a given year, region,
    and application is then calculated for N number of investors and the investor_series with
    calculated cost values is returned.
    """

    # Initialize the year, region, application and technology within this module
    setState(year, region, application, investor_series.name)                   # NOTE: investorSeries.name gives the names of the technology column
    #printState()

    # Calculate the TCO and specific TCO parameters (i.e. capex)
    [tco, tco_parameters, capex, capex_parameters, opexSigma, opex_parameters] = calculateTCO()

    # Record the data in the intermediateOutputs.py module to use for plotting at a later point
    io.recordTCO(YEAR, REGION, APPLICATION, TECHNOLOGY, [tco.mean(), tco.min(), tco.max(), tco.std()])
    io.recordTCOParameters(YEAR, REGION, APPLICATION, TECHNOLOGY, [tco_parameters.mean(axis=1), tco_parameters.std(axis=1)])
    io.recordCAPEX(YEAR, REGION, APPLICATION, TECHNOLOGY, [capex.mean(), capex.min(), capex.max(), capex.std()])
    io.recordOPEX(YEAR, REGION, APPLICATION, TECHNOLOGY, [opexSigma.mean(), opexSigma.min(), opexSigma.max(), opexSigma.std()])
    io.recordCAPEXParameters(YEAR, REGION, APPLICATION, TECHNOLOGY, [capex_parameters.mean(axis=1), capex_parameters.std(axis=1)])
    io.recordOPEXParameters(YEAR, REGION, APPLICATION, TECHNOLOGY, [opex_parameters.mean(axis=1), opex_parameters.std(axis=1)])

    # Keep track of the ICE-D technology TCO in order to calculate the switching cost for other technologies that follow in this specific year, region and application.
    if TECHNOLOGY == 'ICE-D':
        io.storeICEDCAPEX(capex)

    return tco

################################################################################
# INTERNAL FUNCTIONS TO THE MODULE:
################################################################################

def calculateTCO():
    """
    This function calculates the total cost of ownership (TCO) for a given technology in a given year, region,
    and application. The function returns the TCO as well as other TCO parameters for recording purposes.
    """

    # VECTORS
    weights = ucc.getUseCaseParameters(REGION, APPLICATION, TECHNOLOGY, 'Weight')
    ranges = ucc.getUseCaseParameters(REGION, APPLICATION, TECHNOLOGY, 'Range')
    powers = ucc.getUseCaseParameters(REGION, APPLICATION, TECHNOLOGY, 'Power')
    energies = ucc.getUseCaseParameters(REGION, APPLICATION, TECHNOLOGY, 'Energy')
    payload_factors = ucc.getUseCaseParameters(REGION, APPLICATION, TECHNOLOGY, 'Payload Factor')
    loading_factors = ucc.getUseCaseParameters(REGION, APPLICATION, TECHNOLOGY, 'Loading Factor')
    empty_run_shares = ucc.getUseCaseParameters(REGION, APPLICATION, TECHNOLOGY, 'Empty Run Share')

    # SCALARS
    cost_of_capital = ep.getOtherTCOParameters(REGION, APPLICATION, 'Cost of Capital')
    lifetime = ep.getOtherTCOParameters(REGION, APPLICATION, 'Lifetime')
    crf = cost_of_capital*(1 + cost_of_capital)**lifetime/((1 + cost_of_capital)**lifetime - 1)       # [capital recovery factor]
    akt = ranges*ep.getOtherTCOParameters(REGION, APPLICATION, 'Annual Working Days')                 # [annual kilometers travelled]
    scrappage_value_perc = ep.getScrapValue(weights, akt*lifetime)

    # CAPEX
    [capex, capex_parameters] = capexCalculator(weights, powers, energies)
    capex_subsidy = capexSubsidyCalculator(capex)
    PVF = 1/((1+cost_of_capital)**lifetime)
    scrappage_value = (capex)*scrappage_value_perc*PVF

    # OPEX
    # Establish a matrix of discount terms (1/(1+i)^n) for every investor and for each lifetime year of the vehicle:  size:(NUM_INVESTORS x N_LIFETIME)
    discount_term = pd.concat([pd.Series([1/((1 + cost_of_capital)**n) for n in np.arange(1,lifetime+1)])]*NUM_INVESTORS, axis=1).T
    [opex, opex_parameters] = opexCalculator(YEAR, lifetime, capex, weights, ranges, akt, payload_factors, loading_factors, empty_run_shares)

    # Switching Cost Feature
    switching_cost = switchingCostCalculator(capex, lifetime)      # [USD]  (array: (NUM_INVESTORS x N_LIFETIME))

    # The sum of the operation costs discounted over the lifetime of the vehicle
    opexSigma = ((opex)*discount_term).sum(axis=1)
    # Discount the component parameters of the opex in order to record
    opex_parameters = opex_parameters.apply(lambda x: (x * discount_term).sum(axis=1))

    # Set the scaling factor for desired output unit (NOTE: can change this scale factor for desired ouput TCO unit)
    scale_factor = 1

    # Setup the dataframe of tco parameters needed for analysis and plotting
    tco_parameters = capex_parameters*(crf/akt)*scale_factor
    tco_parameters.loc['CAPEX Subsidy'] = -1*capex_subsidy*(crf/akt)*scale_factor
    tco_parameters.loc['Switching Cost'] = switching_cost*(crf/akt)*scale_factor
    tco_parameters.loc['Scrappage Value'] = -1*scrappage_value*(crf/akt)*scale_factor
    tco_parameters = pd.concat([tco_parameters, opex_parameters*(scale_factor/(lifetime*akt))])   # [USD/km] NOTE: check the scale factor above for correct unit here


    # TOTAL COST OF OWNERSHIP
    tco = ((((capex - capex_subsidy + switching_cost) - scrappage_value)*crf + opexSigma/lifetime)/akt)*scale_factor       # [USD/km] NOTE: check the scale factor above for correct unit here

    return tco, tco_parameters, capex, capex_parameters, opexSigma, opex_parameters

def capexSubsidyCalculator(capex_curr):
    """
    This function determines the CAPEX subsidy for a particular technology based on the
    selected CAPEX subsidy intervention in the Control Panel excel file.
    """

    # Get the CAPEX Subsidy intervention option
    capex_subsidy_intervention_selection = ep.getCAPEXSubsidyInterventionSelection(REGION, APPLICATION, TECHNOLOGY)

    # If the CAPEX Subsidy intervention option is a % differential subsidy, calculate the absolute capex_subsidy in [USD] as below
    if capex_subsidy_intervention_selection == '25% ZEV Differential Subsidy':
        # Get the % differential in cost between ICE-D technology and the ZEV technology
        capex_subsidy_perc_diff = ep.getCAPEXSubsidyData(YEAR, REGION, APPLICATION, TECHNOLOGY)
        # Get the ICE-D capex
        capex_iced = io.getICEDCAPEX()
        # Determine the difference between ICE-D and ZEV technology CAPEX
        capex_difference = capex_curr - capex_iced
        # Determine the subsidy
        capex_subsidy = capex_difference*capex_subsidy_perc_diff
        # Only calculate a subsidy if the ZEV technology is more expensive (i.e. if the capex_subsidy is negative, set it to zero)
        capex_subsidy[capex_subsidy < 0] = 0
        # Then return the subsidy
        return capex_subsidy
    else:
        # For all other CAPEX Subsidy intervention options that do not require additional calculation as the '25% ZEV Differential Subsidy' did above, then retun the value stored in the capex_subsidies_df dataframe
        capex_subsidy = ep.getCAPEXSubsidyData(YEAR, REGION, APPLICATION, TECHNOLOGY)

        return capex_subsidy

def switchingCostCalculator(capex, lifetime):
    """
    This function calculates the switching cost for a particular technology in a given year, region and application. The
    switching cost is based on a historical adoption factor and a qualitative adoption factor as outlined in the Supplementary
    Information of the paper associated with this model (DOI: ############).
    """

    # Check to see if the switching cost feature is turned on
    switching_cost_feature_ON_BOOL = ep.getSwitchingCostFeatureBoolean()

    if switching_cost_feature_ON_BOOL and TECHNOLOGY != 'ICE-D':

        # SET THE YEARSPAN
        year_span = lifetime

        ########################################################################
        # SET THE HISTORICAL MARKET SHARE WEIGHTINGS
        # Linearly decreasing weighting
        first_weight = 0.25
        n = year_span-1
        x = (first_weight*year_span - 1)*(2/(n*(n+1)))

        year_weighting_array = np.ones(int(year_span))
        for i in range(int(year_span)):
            year_weighting_array[i] = first_weight - x*(year_span-(i+1))

        ########################################################################
        # Depending on the model iteration year, 'historical' market shares may be taken purely from input data, partially from input data and partially from model output data, or entirely from model output data. The three if statements below differentiate these options.
        if YEAR == BASE_YEAR:
            # Take market share data only from years previous to base year
            pre_base_year_market_shares = ep.getSwitchingCostData('Initial Market Share', int(YEAR-year_span), BASE_YEAR-1, REGION, ep.getApplicationSegment(APPLICATION, 'weight'), TECHNOLOGY)
            # Then get the average
            previous_years_market_shares = (pre_base_year_market_shares*year_weighting_array).sum()/year_weighting_array.sum()

        elif YEAR > BASE_YEAR and YEAR-1-year_span < BASE_YEAR:
            # Take market share data from years previous to base year and from years within the modelling period
            post_base_year_market_shares = io.getMarketShares(BASE_YEAR, YEAR-1, REGION, ep.getApplicationSegment(APPLICATION, 'weight'), TECHNOLOGY)
            pre_base_year_market_shares = ep.getSwitchingCostData('Initial Market Share', int(YEAR-year_span), BASE_YEAR-1, REGION, ep.getApplicationSegment(APPLICATION, 'weight'), TECHNOLOGY)
            # Then concatenate the two arrays and get the average
            previous_years_market_shares = (np.concatenate([pre_base_year_market_shares.values, post_base_year_market_shares])*year_weighting_array).sum()/year_weighting_array.sum()

        else:
            # Take market share data only from within the modelling period
            post_base_year_market_shares = io.getMarketShares(YEAR-year_span, YEAR-1, REGION, ep.getApplicationSegment(APPLICATION, 'weight'), TECHNOLOGY)
            # Then take the average
            previous_years_market_shares = (post_base_year_market_shares*year_weighting_array).sum()/year_weighting_array.sum()

        threshold = ep.getSwitchingCostData('Threshold', 'None', 'None', REGION, ep.getApplicationSegment(APPLICATION, 'weight'), TECHNOLOGY)
        multiplier = ep.getSwitchingCostData('Switching Cost Markup', 'None', 'None', REGION, ep.getApplicationSegment(APPLICATION, 'weight'), TECHNOLOGY)

        # S-curve parameters (these parameters are assumed)
        start = 0
        stop = 25
        min = 0
        max = 1
        k = 0.3
        a = 10

        x = (previous_years_market_shares/threshold)*(stop-start)
        s_curve_factor = 1 - (min + (max-min)*np.power((1/(1+np.exp(-1*k*x))),a))

        # Get the ICE-D capex
        capex = io.getICEDCAPEX()

        # Calculate the switching cost
        switching_cost = multiplier*capex*s_curve_factor

        # Record the switching cost multiplier and the switching cost (absolute)
        io.recordSwitchingCostMultiplier(YEAR, REGION, APPLICATION, TECHNOLOGY, s_curve_factor)
        io.recordSwitchingCost(YEAR, REGION, APPLICATION, TECHNOLOGY, switching_cost.mean())

        # If the previous year market share average has reached the threshold (or within error of the threshold), set the threshold boolean to TRUE
        error = 0.01
        if abs(previous_years_market_shares - threshold) < error:
            ep.setSwitchingThresholdBool(True, REGION, APPLICATION, TECHNOLOGY)

        # Return the calculated switching cost
        return switching_cost
    else:
        return 0

def capexCalculator(weight, power, energy):
    """
    This function organizes the two CAPEX calculator functions: one for constant CAPEX parameters and the other for distributed
    CAPEX parameters. Both functions are called from the capex.py module where the capital expenditure for a given technology
    in a year, region, and application is determined.
    """

    # Determine whether the capex will be calculated with distributed or constant parameters
    type = ep.getParameterType('capex')

    if type == "constant":
        [capex, capex_parameters] = cpx.techCalculatorConstant(getState(), weight, power, energy)

    elif type == "distributed":
        [capex, capex_parameters] = cpx.techCalculatorDistributed(getState(), weight, power, energy)

    return [capex, capex_parameters]

def opexCalculator(year, lifetime, capex, weight, range, akt, payload_factor, loading_factor, empty_run_share):
    """
    This function organizes the two OPEX calculator functions: one for constant OPEX parameters and the other for distributed
    OPEX parameters. Both functions are called from the opex.py module where the operation expenditure for a given technology
    in a year, region, and application is determined.
    """
    # Determine whether the opex will be calculated with distributed or constant parameters
    type = ep.getParameterType('opex')

    # Get the current state of the module to pass to the opx module
    state = getState()
    state[0] = year

    if type == "constant":
        [opex, opex_parameters] = opx.regionCalculatorConstant(state, lifetime, capex, weight, range, akt, payload_factor, loading_factor, empty_run_share)

    elif type == 'distributed':
        [opex, opex_parameters] = opx.regionCalculatorDistributed(state, lifetime, capex, weight, range, akt, payload_factor, loading_factor, empty_run_share)

    return [opex, opex_parameters]

##############################################################
# SET THE STATE OF THE MODULE:
##############################################################

def setState(year, region, application, technology):
    """
    This function sets the state in the current module. The state encompasses all global variables listed below.
    """

    global YEAR
    YEAR = year
    global REGION
    REGION = region
    global APPLICATION
    APPLICATION = application
    global TECHNOLOGY
    TECHNOLOGY = technology

def getState():
    """
    This function returns the current state, as in what year, region, application and technology the investor
    is calculating the TCO for. The function returns a python list with the state variables in the order listed
    below.
    """

    return [YEAR, REGION, APPLICATION, TECHNOLOGY]

def printState():
    """
    This function prints the current state, as in what year, region, application, technology the module is in.
    """
    state_string = "Current State: [" + str(YEAR) + ", " + REGION + ", " + APPLICATION + ", " + TECHNOLOGY + "]"

    print(state_string)
