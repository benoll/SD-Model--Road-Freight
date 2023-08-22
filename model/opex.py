# FILE: opex.py
# PROJECT: Global System-Dynamics Freight Model
# MODULE DESCRIPTION: This module computes the operational expenditure (OPEX) as a sub-module
# of the main totalCostOfOwnership.py module.This module is called directly from
# the opexCalculator function and can calculate the OPEX with constant or distributed parameters.

# Import statements
import pandas as pd
import numpy as np
import externalParameters as ep
import experienceCurves as ec

# Global constants
NUM_INVESTORS = ep.getNumInvestors()
BASE_YEAR = ep.getYears()['Years'][0]

##############################################################
# INTERNAL FUNCTIONS TO THE MODULE:
##############################################################

def getLevelizedInfrastructureCost():
    """
    This function calculates the levelized cost of infrastructure to charge a battery electric at varying power levels. Power
    levels for BEVs in different applications and regions are pre-defined in the 'TCO Data.xlsx' file within the '_OPEX Infrastructure' sheet.
    This method follows the levelized cost of charging (LCOC) logic (DOI: https://doi.org/10.1038/s41467-022-32835-7) but deviates slightly from
    the method in the referenced study.
    """

    # Get the selected power level for this region and application
    power_level = ep.getChargingInfrastructureSelection(REGION, APPLICATION)                                                                    # [kW]

    # Get the charging infrastructure cost parameters. (Cost components include: charging equipment cost and installation cost).
    top_up_down = ep.getTopUpDownData('Infrastructure - Charging Equipment', REGION, YEAR)
    C_equipment = np.ones(NUM_INVESTORS)*ep.getChargingInfrastructureData(REGION, power_level, 'Charging Equipment Cost')*(1+top_up_down)       # [USD]
    top_up_down = ep.getTopUpDownData('Infrastructure - Installation', REGION, YEAR)
    C_installation = np.ones(NUM_INVESTORS)*ep.getChargingInfrastructureData(REGION, power_level, 'Installation Cost')*(1+top_up_down)          # [USD]

    # Get the remaining charging infrastructure data parameters. (Parameters include: operation and maintenance cost for the charging station, lifetime of the charging station, cost of capital,
    # daily station utilization rate, and number of yearly charging days)
    o_and_m_percent = ep.getChargingInfrastructureData(REGION, power_level, 'O&M')                                          # [%]
    lifetime = ep.getChargingInfrastructureData(REGION, power_level, 'Lifetime')                                            # [years]
    cost_of_capital = ep.getOtherTCOParameters(REGION, APPLICATION, 'Cost of Capital')                                      # [%]
    daily_utilization_rate = ep.getChargingInfrastructureData(REGION, power_level, 'Daily Utilization Rate')                # [% of day]
    charge_days = ep.getChargingInfrastructureData(REGION, power_level, 'Charge Days Per Year')                             # [days/year]

    # Calculate the charging infrastructure O&M cost
    C_o_and_m = pd.concat([pd.Series(C_equipment*o_and_m_percent)]*lifetime.astype(int), axis=1)

    # Calculate the total annual energy charged at a given station
    E_total = power_level*24*daily_utilization_rate*charge_days                                                             # [kWh]

    # Calculate the discount term
    discount_term = pd.concat([pd.Series([1/((1 + cost_of_capital)**n) for n in np.arange(1,lifetime+1)])]*NUM_INVESTORS, axis=1).T

    # Calculate the total levelized infrastructure cost
    levelized_infrastructure_cost = (pd.Series(C_equipment + C_installation) + (C_o_and_m*discount_term).sum(axis=1)) / (E_total*discount_term).sum(axis=1)           # [USD/kWh]

    return levelized_infrastructure_cost

################################################################################
# EXTERNAL FUNCTIONS TO THE MODULE:
################################################################################
##############################################################
# Calculate the OPEX for all technologies:
# (With constant parameters)
##############################################################

def regionCalculatorConstant(state, lifetime, capex_series, weight_series, range_series, akt_series, payload_factor_series, loading_factor_series, empty_run_share_series):
    """
    This function calculates the operational expenditure (OPEX) for a given technology in a given year, region and application with
    constant parameters.
    """

    # Set the state
    [YEAR, REGION, APPLICATION, TECHNOLOGY] = state
    setState(YEAR, REGION, APPLICATION, TECHNOLOGY)
    # Set the weight class for this specific application
    weight_class = ep.getApplicationSegment(APPLICATION, 'weight')

    # Initialize the number of yearly working days
    working_days = ep.getOtherTCOParameters(REGION, APPLICATION, 'Annual Working Days')     # [days/year]

    # Create a matrix for the capex, weight and range parameters
    capex = pd.concat([capex_series]*lifetime.astype(int), axis=1, ignore_index=True)                                      # [USD]
    weight = pd.concat([weight_series]*lifetime.astype(int), axis=1, ignore_index=True)                                    # [kg]
    range = pd.concat([range_series]*lifetime.astype(int), axis=1, ignore_index=True)                                      # [km/day]

    # Payload
    max_payload = weight*pd.concat([payload_factor_series]*lifetime.astype(int), axis=1, ignore_index=True)                # [kg]
    loading = max_payload*pd.concat([loading_factor_series]*lifetime.astype(int), axis=1, ignore_index=True)               # [kg]
    loading_with_empty_run_share = loading*(1-pd.concat([empty_run_share_series]*lifetime.astype(int), axis=1, ignore_index=True))    # [kg]
    if ep.getEmptyRunBoolean():
        on_road_weight = weight - (max_payload-loading_with_empty_run_share)                                               # [kg]
        net_payload = loading_with_empty_run_share                                                                         # [kg]
    else:
        on_road_weight = weight - (max_payload-loading)                                                                    # [kg]
        net_payload = loading                                                                                              # [kg]

    # Tolls
    tolls = range.values*working_days*ep.getTolls(YEAR, YEAR+lifetime-1, REGION, APPLICATION, TECHNOLOGY)*ep.getOtherTCOParameters(REGION, APPLICATION, '% Driven on Toll Roads')  # [USD]

    # O&M
    o_and_m = range.values*working_days*ep.getOandM(YEAR, REGION, APPLICATION, TECHNOLOGY)                                # [USD]

    # Insurance
    insurance = capex.values*ep.getInsurance(REGION, 'Mid Insurance')                                                     # [USD]

    # Wages
    wage_percent_increase = ep.getWages(REGION, APPLICATION, 'Yearly Wage % Increase')                                    # [%]
    mid_wage = ep.getWages(REGION, APPLICATION, 'Mid Wage')                                                               # [USD]
    start_wage = (mid_wage*((1+wage_percent_increase)**(YEAR-BASE_YEAR)))                                                 # [USD]
    wages = pd.concat([pd.Series([start_wage*((1+wage_percent_increase)**(n-1)) for n in np.arange(1,lifetime+1)])]*NUM_INVESTORS, axis=1).T        # [USD]

    # Fuel Costs
    if TECHNOLOGY == 'FCEV':
        # For the hydrogen fuel we use the 'most likely' parameter from the PERT distribution (NOTE: we do not use a distribution itself in the constant function)
        fuel_type = ep.getFuelSelection(REGION, APPLICATION, TECHNOLOGY)
        fuel_cost_array = pd.Series(ep.getFuelCosts(REGION, fuel_type, 'Most likely', YEAR, YEAR+lifetime-1).values)
        fuel_cost_matrix = pd.concat([fuel_cost_array]*NUM_INVESTORS, axis=1, ignore_index=True).T                              # [USD/fuel-specific-consumption-unit]
    elif TECHNOLOGY == 'ICE-D' and REGION == 'Brazil' and weight_class == 'LDV':
        # For all other fuels/electricity we use the 'mean' parameter from the normal distribution (NOTE: we do not use a distribution itself in the constant function).
        # A special case for LDV ICE-D vehicles in Brazil: the fuel split is assumed to be 50/50 diesel/biodiesel(hydrous ethanol).
        # First get the cost matrix for fuel type 1
        fuel_type_1 = ep.getFuelSelection(REGION, APPLICATION, TECHNOLOGY)
        fuel_cost_array_1 = pd.Series(ep.getFuelCosts(REGION, fuel_type_1, 'Mean', YEAR, YEAR+lifetime-1).values)
        fuel_cost_matrix_1 = pd.concat([fuel_cost_array_1]*NUM_INVESTORS, axis=1, ignore_index=True).T
        # Then get the cost matrix for fuel type 2
        fuel_type_2 = 'Biodiesel'
        fuel_cost_array_2 = pd.Series(ep.getFuelCosts(REGION, fuel_type_2, 'Mean', YEAR, YEAR+lifetime-1).values)
        fuel_cost_matrix_2 = pd.concat([fuel_cost_array_2]*NUM_INVESTORS, axis=1, ignore_index=True).T
        # Then average them (50/50 weight)
        fuel_cost_matrix = (fuel_cost_matrix_1 + fuel_cost_matrix_2)/2                                                          # [USD/fuel-specific-consumption-unit]
    else:
        # For all other fuels/electricity we use the 'mean' parameter from the normal distribution (NOTE: we do not use a distribution itself in the constant function)
        fuel_type = ep.getFuelSelection(REGION, APPLICATION, TECHNOLOGY)
        fuel_cost_array = pd.Series(ep.getFuelCosts(REGION, fuel_type, 'Mean', YEAR, YEAR+lifetime-1).values)
        fuel_cost_matrix = pd.concat([fuel_cost_array]*NUM_INVESTORS, axis=1, ignore_index=True).T                              # [USD/fuel-specific-consumption-unit]

    # Fuel Consumption
    if ep.getFuelConsumptionMode() == 'Static':
        param_c = ep.getFuelConsumptionData(TECHNOLOGY, 'c', BASE_YEAR)*(1+ep.getFuelConsumptionPercDiffData(REGION))
        param_d = ep.getFuelConsumptionData(TECHNOLOGY, 'd', BASE_YEAR)*(1+ep.getFuelConsumptionPercDiffData(REGION))
    else:
        param_c = ep.getFuelConsumptionData(TECHNOLOGY, 'c', YEAR)*(1+ep.getFuelConsumptionPercDiffData(REGION))
        param_d = ep.getFuelConsumptionData(TECHNOLOGY, 'd', YEAR)*(1+ep.getFuelConsumptionPercDiffData(REGION))

    consumption = param_c*np.log(on_road_weight) + param_d                                                                      # [fuel-specific-consumption-unit/km]

    # Infrastructure
    if TECHNOLOGY == 'BEV':
        levelized_infrastructure_cost = pd.concat([getLevelizedInfrastructureCost()]*lifetime.astype(int), axis=1)   # [USD/kWh]
    else:
        levelized_infrastructure_cost = np.full((NUM_INVESTORS,lifetime.astype(int)), 0)                             # [USD/kWh]

    # Annual Fuel Costs
    fuel_costs = fuel_cost_matrix*consumption*working_days*range                                                     # [USD]
    infrastructure_costs = levelized_infrastructure_cost*consumption*working_days*range                              # [USD]

    # Carbon Tax
    carbon_cost = 0     # initialize the cost to 0 for no intervention
    if ep.getCarbonTaxInterventionSelection(REGION) != 'No Carbon Tax':
        # First get the energy consumed per kilometer and total energy for the annual kilometers travelled
        fuel_type = ep.getFuelSelection(REGION, APPLICATION, TECHNOLOGY)
        energy_per_km = consumption                                                                                  # [fuel-specific-consumption-unit/km]
        total_energy = energy_per_km*pd.concat([akt_series]*lifetime.astype(int), axis=1, ignore_index=True)         # [fuel-specific-consumption-unit]
        # Then get the emissions factor and the total emissions
        MJ_to_kWh = 0.277778
        wtt_emission_factor = ep.getWTTEmissionsFactorsData(REGION, fuel_type, YEAR)/(1000*1000)                     # [t CO2eq/fuel-specific-consumption-unit]
        ttw_emission_factor = ep.getTTWEmissionsFactorsData(APPLICATION, TECHNOLOGY, 'TTW Emissions')/(1000*1000)    # [t CO2eq/fuel-specific-consumption-unit]
        total_emission_factor = wtt_emission_factor + ttw_emission_factor                                            # [t CO2eq/fuel-specific-consumption-unit]
        total_emissions = total_emission_factor*total_energy                                                         # [tCO2eq]
        # Finally, get the carbon tax and determine the carbon cost
        carbon_tax = ep.getCarbonTaxInterventionData(REGION, YEAR, YEAR+lifetime-1)                                  # [USD/tCO2eq]
        carbon_cost = carbon_tax*total_emissions                                                                     # [USD]

    # OPEX Calculation
    opex = tolls + fuel_costs + infrastructure_costs + o_and_m + insurance + wages + carbon_cost                     # [USD]

    # Establish dataframe of requested values for waterfallchart
    opex_parameters = pd.Series(data = [tolls, fuel_costs, infrastructure_costs, o_and_m, insurance, wages, carbon_cost], index = ep.getOPEXRecordParameters())

    return [opex, opex_parameters]

##############################################################
# Calculate the OPEX for all technologies:
# (With distributed parameters)
##############################################################

def regionCalculatorDistributed(state, lifetime, capex_series, weight_series, range_series, akt_series, payload_factor_series, loading_factor_series, empty_run_share_series):
    """
    This function calculates the operational expenditure (OPEX) for a given technology in a given year, region and application with
    distributed parameters.
    """

    # Set the state
    [YEAR, REGION, APPLICATION, TECHNOLOGY] = state
    setState(YEAR, REGION, APPLICATION, TECHNOLOGY)
    # Get the weight class for this specific application
    weight_class = ep.getApplicationSegment(APPLICATION, 'weight')

    # Initialize the number of yearly working days
    working_days = ep.getOtherTCOParameters(REGION, APPLICATION, 'Annual Working Days')     # [days/year]

    # Create a matrix for the capex, weight and range parameters
    capex = pd.concat([capex_series]*lifetime.astype(int), axis=1, ignore_index=True)                                      # [USD]
    weight = pd.concat([weight_series]*lifetime.astype(int), axis=1, ignore_index=True)                                    # [kg]
    range = pd.concat([range_series]*lifetime.astype(int), axis=1, ignore_index=True)                                      # [km/day]

    # Payload and on-road weight calculation
    max_payload = weight*pd.concat([payload_factor_series]*lifetime.astype(int), axis=1, ignore_index=True)                # [kg]
    loading = max_payload*pd.concat([loading_factor_series]*lifetime.astype(int), axis=1, ignore_index=True)               # [kg]
    loading_with_empty_run_share = loading*(1-pd.concat([empty_run_share_series]*lifetime.astype(int), axis=1, ignore_index=True))    # [kg]
    if ep.getEmptyRunBoolean():
        on_road_weight = weight - (max_payload-loading_with_empty_run_share)                                               # [kg]
        net_payload = loading_with_empty_run_share                                                                         # [kg]
    else:
        on_road_weight = weight - (max_payload-loading)                                                                    # [kg]
        net_payload = loading                                                                                              # [kg]

    # Tolls
    tolls = range.values*working_days*ep.getTolls(YEAR, YEAR+lifetime-1, REGION, APPLICATION, TECHNOLOGY)*ep.getOtherTCOParameters(REGION, APPLICATION, '% Driven on Toll Roads')  # [USD]

    # O&M
    o_and_m = range.values*working_days*ep.getOandM(YEAR, REGION, APPLICATION, TECHNOLOGY)                                # [USD]

    # Insurance
    mid = ep.getInsurance(REGION, 'Mid Insurance')
    high = ep.getInsurance(REGION, 'High Insurance')
    low = ep.getInsurance(REGION, 'Low Insurance')
    insurance = capex.values*pd.concat([pd.Series(ep.distributionFunction([low, mid, high], 'PERT', NUM_INVESTORS))]*lifetime.astype(int), axis=1)  # [USD]

    # Wages
    wage_percent_increase = ep.getWages(REGION, APPLICATION, 'Yearly Wage % Increase')                                    # [%]
    mid = ep.getWages(REGION, APPLICATION, 'Mid Wage')*(1+wage_percent_increase)**(YEAR-BASE_YEAR)                        # [USD]
    low = ep.getWages(REGION, APPLICATION, 'Low Wage')*(1+wage_percent_increase)**(YEAR-BASE_YEAR)                        # [USD]
    high = ep.getWages(REGION, APPLICATION, 'High Wage')*(1+wage_percent_increase)**(YEAR-BASE_YEAR)                      # [USD]
    start_wage = pd.Series(ep.distributionFunction([low, mid, high], 'PERT', [NUM_INVESTORS]))                            # [USD]
    wages = pd.concat([start_wage*((1+wage_percent_increase)**(n-1)) for n in np.arange(1,lifetime+1)], axis=1)           # [USD]

    # Fuel Costs
    if TECHNOLOGY == 'FCEV':
        # For the hydrogen fuel we use a PERT distribution
        fuel_type = ep.getFuelSelection(REGION, APPLICATION, TECHNOLOGY)
        fuel_cost_matrix = pd.DataFrame(index=np.arange(0, NUM_INVESTORS), columns=np.arange(0, lifetime.astype(int)))
        for index, y in np.ndenumerate(np.arange(YEAR,YEAR+lifetime)):
            most_likely = ep.getFuelCosts(REGION, fuel_type, 'Most likely', y, y).values
            high = ep.getFuelCosts(REGION, fuel_type, 'High', y, y).values
            low = ep.getFuelCosts(REGION, fuel_type, 'Low', y, y).values
            investor_column = ep.distributionFunction([low, most_likely, high], 'PERT', [NUM_INVESTORS, 1])
            fuel_cost_matrix.loc[:,index] = investor_column                                                               # [USD/fuel-specific-consumption-unit]
    elif TECHNOLOGY == 'ICE-D' and REGION == 'Brazil' and weight_class == 'LDV':
        # For all other fuels/electricity we use a normal distribution. A special case for LDV ICE-D vehicles in Brazil: the fuel split is assumed to be 50/50 diesel/biodiesel(hydrous ethanol).
        fuel_type_1 = ep.getFuelSelection(REGION, APPLICATION, TECHNOLOGY)
        # First get the cost matrix for fuel type 1
        fuel_cost_matrix_1 = pd.DataFrame(index=np.arange(0, NUM_INVESTORS), columns=np.arange(0, lifetime.astype(int)))
        for index, y in np.ndenumerate(np.arange(YEAR,YEAR+lifetime)):
            mean = ep.getFuelCosts(REGION, fuel_type_1, 'Mean', y, y).values
            std = ep.getFuelCosts(REGION, fuel_type_1, 'Std', y, y).values
            investor_column = ep.distributionFunction([mean, std], 'Normal', [NUM_INVESTORS, 1])
            fuel_cost_matrix_1.loc[:,index] = investor_column
        # Then get the cost matrix for fuel type 2
        fuel_type_2 = 'Biodiesel'
        fuel_cost_matrix_2 = pd.DataFrame(index=np.arange(0, NUM_INVESTORS), columns=np.arange(0, lifetime.astype(int)))
        for index, y in np.ndenumerate(np.arange(YEAR,YEAR+lifetime)):
            mean = ep.getFuelCosts(REGION, fuel_type_2, 'Mean', y, y).values
            std = ep.getFuelCosts(REGION, fuel_type_2, 'Std', y, y).values
            investor_column = ep.distributionFunction([mean, std], 'Normal', [NUM_INVESTORS, 1])
            fuel_cost_matrix_2.loc[:,index] = investor_column
        # Then average them (50/50 weight)
        fuel_cost_matrix = (fuel_cost_matrix_1 + fuel_cost_matrix_2)/2                                                   # [USD/fuel-specific-consumption-unit]
    else:
        # For all other fuels/electricity we use a normal distribution
        fuel_type = ep.getFuelSelection(REGION, APPLICATION, TECHNOLOGY)
        fuel_cost_matrix = pd.DataFrame(index=np.arange(0, NUM_INVESTORS), columns=np.arange(0, lifetime.astype(int)))
        for index, y in np.ndenumerate(np.arange(YEAR,YEAR+lifetime)):
            mean = ep.getFuelCosts(REGION, fuel_type, 'Mean', y, y).values
            std = ep.getFuelCosts(REGION, fuel_type, 'Std', y, y).values
            investor_column = ep.distributionFunction([mean, std], 'Normal', [NUM_INVESTORS, 1])
            fuel_cost_matrix.loc[:,index] = investor_column                                                             # [USD/fuel-specific-consumption-unit]

    # Fuel Consumption
    if ep.getFuelConsumptionMode() == 'Static':
        param_c_mid = ep.getFuelConsumptionData(TECHNOLOGY, 'c', BASE_YEAR)
        param_d_mid = ep.getFuelConsumptionData(TECHNOLOGY, 'd', BASE_YEAR)
    else:
        param_c_mid = ep.getFuelConsumptionData(TECHNOLOGY, 'c', YEAR)
        param_d_mid = ep.getFuelConsumptionData(TECHNOLOGY, 'd', YEAR)
    param_c = pd.concat([pd.Series(ep.distributionFunction([param_c_mid*0.9, param_c_mid, param_c_mid*1.1], 'PERT', NUM_INVESTORS))]*lifetime.astype(int), axis=1)*(1+ep.getFuelConsumptionPercDiffData(REGION))
    param_d = pd.concat([pd.Series(ep.distributionFunction([param_d_mid*0.9, param_d_mid, param_d_mid*1.1], 'PERT', NUM_INVESTORS))]*lifetime.astype(int), axis=1)*(1+ep.getFuelConsumptionPercDiffData(REGION))
    consumption = param_c*np.log(on_road_weight) + param_d                                                              # [fuel-specific-consumption-unit/km]

    # Infrastructure
    if TECHNOLOGY == 'BEV':
        levelized_infrastructure_cost = pd.concat([getLevelizedInfrastructureCost()]*lifetime.astype(int), axis=1)      # [USD/kWh]
    else:
        levelized_infrastructure_cost = np.full((NUM_INVESTORS,lifetime.astype(int)), 0)                                # [USD/kWh]

    # Annual Fuel Costs
    fuel_costs = fuel_cost_matrix*consumption*working_days*range                                                        # [USD]
    infrastructure_costs = levelized_infrastructure_cost*consumption*working_days*range                                 # [USD]

    # Carbon Tax
    carbon_cost = 0     # initialize the cost to 0 for no intervention
    if ep.getCarbonTaxInterventionSelection(REGION) != 'No Carbon Tax':
        # First get the energy consumed per kilometer and total energy for the annual kilometers travelled
        fuel_type = ep.getFuelSelection(REGION, APPLICATION, TECHNOLOGY)
        energy_per_km = consumption                                                                                     # [fuel-specific-consumption-unit/km]
        total_energy = energy_per_km*pd.concat([akt_series]*lifetime.astype(int), axis=1, ignore_index=True)            # [fuel-specific-consumption-unit]
        # Then get the emissions factor and the total emissions
        MJ_to_kWh = 0.277778
        wtt_emission_factor = ep.getWTTEmissionsFactorsData(REGION, fuel_type, YEAR)/(1000*1000)                        # [t CO2eq/fuel-specific-consumption-unit]
        ttw_emission_factor = ep.getTTWEmissionsFactorsData(APPLICATION, TECHNOLOGY, 'TTW Emissions')/(1000*1000)       # [t CO2eq/fuel-specific-consumption-unit]
        total_emission_factor = wtt_emission_factor + ttw_emission_factor                                               # [t CO2eq/fuel-specific-consumption-unit]
        total_emissions = total_emission_factor*total_energy                                                            # [tCO2eq]
        # Finally, get the carbon tax and determine the carbon cost
        carbon_tax = ep.getCarbonTaxInterventionData(REGION, YEAR, YEAR+lifetime-1)                                     # [USD/tCO2eq]
        carbon_cost = carbon_tax*total_emissions                                                                        # [USD]

    # OPEX Calculation
    opex = tolls + fuel_costs + infrastructure_costs + o_and_m + insurance + wages + carbon_cost                        # [USD]

    # Establish dataframe of requested values for waterfallchart
    opex_parameters = pd.Series(data = [insurance, o_and_m, tolls, wages, fuel_costs, infrastructure_costs, carbon_cost], index = ep.getOPEXRecordParameters())


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
    is calculating the OPEX for. The function returns a python list with the state variables in the order listed
    below.
    """

    return [YEAR, REGION, APPLICATION, TECHNOLOGY]

def printState():
    """
    This function prints the current state, as in what year, region, application, technology the module is in.
    """
    state_string = "Current State: [" + str(YEAR) + ", " + REGION + ", " + APPLICATION + ", " + TECHNOLOGY + "]"

    print(state_string)
