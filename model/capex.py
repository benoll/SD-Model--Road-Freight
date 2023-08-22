# FILE: capex.py
# PROJECT: Global System-Dynamics Freight Model
# MODULE DESCRIPTION: This modul computes the capital expenditure (CAPEX) as a sub-module
# of the main totalCostOfOwnership.py module. This module is called directly from
# capexCalculator and can calculate the capex with constant or distributed parameters.

# Import statements
import pandas as pd
import numpy as np
import externalParameters as ep
import experienceCurves as ec

# Global constants
NUM_INVESTORS = ep.getNumInvestors()

################################################################################
# EXTERNAL FUNCTIONS TO THE MODULE:
################################################################################

##############################################################
# Calculate the CAPEX for all technologies:
# (With constant parameters)
##############################################################

def techCalculatorConstant(state, weight, power, energy):
    """
    This function calculates the capital expenditure (CAPEX) for a given technology in a given year, region and application with
    constant parameters.
    """
    # Set the state
    [YEAR, REGION, APPLICATION, TECHNOLOGY] = state
    setState(YEAR, REGION, APPLICATION, TECHNOLOGY)
    # Set the weight class for this specific application
    weight_class = ep.getApplicationSegment(APPLICATION, 'weight')

    # Get the updated learning component costs from the experience curve module
    cost_series = ec.getLearningComponentCost(REGION, APPLICATION)
    # Get the technology-specific booleans for all CAPEX parameters in order to determine which vehicle components are included for the current technology
    cost_boolean = ep.getCAPEXBoolean(TECHNOLOGY)

    # Get the technology-specific technical parameters from the external parameters module. These parameters determine the ratio of energy storage (battery vs. tank) and power (electric motor vs. internal combustion engine) for a given technology.
    battery_energy_coeff = ep.getTechnologyData(TECHNOLOGY, APPLICATION, 'Battery Energy Coeff')
    tank_energy_coeff = ep.getTechnologyData(TECHNOLOGY, APPLICATION, 'Tank Energy Coeff')
    electric_motor_power_coeff = ep.getTechnologyData(TECHNOLOGY, APPLICATION, 'Electric Motor Power Coeff')
    ice_power_coeff = ep.getTechnologyData(TECHNOLOGY, APPLICATION, 'ICE Power Coeff')

    # Rest of Truck (Comprised of: chassis.)
    chassis_cost = cost_boolean.loc['Chassis']*np.ones(NUM_INVESTORS)*ep.getChassisData(REGION, APPLICATION, 'Mid Chassis')
    rest_of_truck_cost = chassis_cost


    # Powertrain (Comprised of: ice powertrain, electric drive system, fuel cell. Depending on the technology)
    top_up_down = ep.getTopUpDownData('ICE Powertrain', REGION, YEAR)
    ice_powertrain_cost = cost_boolean.loc['ICE Powertrain']*np.ones(NUM_INVESTORS)*cost_series.loc[(REGION,'ICE Powertrain', weight_class), 'Mid Cost']*(1+top_up_down)
    top_up_down = ep.getTopUpDownData('Electric Drive System', REGION, YEAR)
    electric_drive_system_cost = cost_boolean.loc['Electric Drive System']*np.ones(NUM_INVESTORS)*cost_series.loc[(REGION,'Electric Drive System', weight_class), 'Mid Cost']*(1+top_up_down)
    top_up_down = ep.getTopUpDownData('Fuel Cell System', REGION, YEAR)
    fuel_cell_cost = cost_boolean.loc['Fuel Cell System']*np.ones(NUM_INVESTORS)*cost_series.loc[(REGION,'Fuel Cell System', weight_class), 'Mid Cost']*(1+top_up_down)

    powertrain_cost = power*ice_powertrain_cost*ice_power_coeff + power*electric_drive_system_cost*electric_motor_power_coeff + power*fuel_cell_cost*electric_motor_power_coeff

    # Introduce a percentage increase for the natural gas ICE engine component of the powertrain as compared to diesel (but only for the MDV and HDV segments)

    if TECHNOLOGY == 'ICE-NG' and (weight_class == 'MDV' or weight_class == 'HDV'):
        natural_gas_engine_multiplier = 1.2                                     # For the MDV/HDV vehicles we increase the engine component of the powertrain cost by 20%
        engine_perc_of_powertrain = 0.85                                        # The % cost of the engine of the entire ICE powertrain (engine, exhaust, transmission)
        ice_powertrain_ng_cost = power*ice_powertrain_cost*engine_perc_of_powertrain*natural_gas_engine_multiplier + power*ice_powertrain_cost*(1-engine_perc_of_powertrain)
        powertrain_cost = ice_powertrain_ng_cost + power*electric_drive_system_cost*electric_motor_power_coeff + power*fuel_cell_cost*electric_motor_power_coeff


    # Energy Storage (Comprised of: battery, hydrogen tank, diesel tank, natural gas tank. Depending on the technology.)
    top_up_down = ep.getTopUpDownData('Li-ion Battery', REGION, YEAR)
    battery_cost = cost_boolean.loc['Li-ion Battery']*np.ones(NUM_INVESTORS)*cost_series.loc[(REGION,'Li-ion Battery', weight_class), 'Mid Cost']*(1+top_up_down)
    top_up_down = ep.getTopUpDownData('Hydrogen Tank', REGION, YEAR)
    tank_h2_cost = cost_boolean.loc['Hydrogen Tank']*np.ones(NUM_INVESTORS)*cost_series.loc[(REGION,'Hydrogen Tank', weight_class), 'Mid Cost']*(1+top_up_down)
    top_up_down = ep.getTopUpDownData('Diesel Tank', REGION, YEAR)
    tank_diesel_cost = cost_boolean.loc['Diesel Tank']*np.ones(NUM_INVESTORS)*cost_series.loc[(REGION,'Diesel Tank', weight_class), 'Mid Cost']*(1+top_up_down)
    top_up_down = ep.getTopUpDownData('Natural Gas Tank', REGION, YEAR)
    tank_ng_cost = cost_boolean.loc['Natural Gas Tank']*np.ones(NUM_INVESTORS)*cost_series.loc[(REGION,'Natural Gas Tank', weight_class), 'Mid Cost']*(1+top_up_down)

    energy_storage_cost = energy*tank_diesel_cost*tank_energy_coeff + energy*tank_ng_cost*tank_energy_coeff + energy*tank_h2_cost*tank_energy_coeff + energy*battery_cost*battery_energy_coeff

    # Profit Margin
    profit_margin = ep.getMonetaryParameters(REGION, 'Profit Margin')

    # Integration Factor (NOTE: this is only computed for non-incumbent technologies (i.e. all technologies except ICE-D))
    integration_factor = 0
    if TECHNOLOGY != 'ICE-D':
        if weight_class == 'LDV':
            integration_factor_title = 'Integration Factor - '+TECHNOLOGY+' - LDV'
        else:
            integration_factor_title = 'Integration Factor - '+TECHNOLOGY+' - MDV/HDV'
        # Determine the integration factor
        integration_factor = cost_boolean.loc[integration_factor_title]*np.ones(NUM_INVESTORS)*cost_series.loc[(REGION, integration_factor_title, weight_class), 'Mid Cost']

    # Calculate the total CAPEX as per the equation below
    capex = (powertrain_cost + energy_storage_cost + rest_of_truck_cost)*(1+integration_factor)/(1-profit_margin)

    # Establish a dataframe of parameters for storage
    capex_parameters = pd.DataFrame([powertrain_cost*(1+integration_factor)/(1-profit_margin), energy_storage_cost*(1+integration_factor)/(1-profit_margin), rest_of_truck_cost*(1+integration_factor)/(1-profit_margin)], index = ep.getCAPEXRecordParameters())

    return [capex, capex_parameters]

##############################################################
# Calculate the CAPEX for all technologies:
# (With distributed parameters)
##############################################################

def techCalculatorDistributed(state, weight, power, energy):
    """
    This function calculates the capital expenditure (CAPEX) for a given technology in a given year, region and application with
    constant parameters.
    """
    # Set the state
    [YEAR, REGION, APPLICATION, TECHNOLOGY] = state
    setState(YEAR, REGION, APPLICATION, TECHNOLOGY)
    # Set the weight class for this specific application
    weight_class = ep.getApplicationSegment(APPLICATION, 'weight')

    # Get the updated learning component costs from the experience curve module
    cost_series = ec.getLearningComponentCost(REGION, APPLICATION)
    # Get the technology-specific booleans for all CAPEX parameters in order to determine which vehicle components are included for the current technology
    cost_boolean = ep.getCAPEXBoolean(TECHNOLOGY)

    # Get the technology-specific technical parameters from the external parameters module. These parameters determine the ratio of energy storage (battery vs. tank) and power (electric motor vs. internal combustion engine) for a given technology.
    battery_energy_coeff = ep.getTechnologyData(TECHNOLOGY, APPLICATION, 'Battery Energy Coeff')
    tank_energy_coeff = ep.getTechnologyData(TECHNOLOGY, APPLICATION, 'Tank Energy Coeff')
    electric_motor_power_coeff = ep.getTechnologyData(TECHNOLOGY, APPLICATION, 'Electric Motor Power Coeff')
    ice_power_coeff = ep.getTechnologyData(TECHNOLOGY, APPLICATION, 'ICE Power Coeff')

    # Rest of Truck (Comprised of: chassis.)
    mid = ep.getChassisData(REGION, APPLICATION, 'Mid Chassis')
    high = ep.getChassisData(REGION, APPLICATION, 'High Chassis')
    low = ep.getChassisData(REGION, APPLICATION, 'Low Chassis')
    chassis_cost = cost_boolean.loc['Chassis']*ep.distributionFunction([low, mid, high], 'PERT', NUM_INVESTORS)
    rest_of_truck_cost = chassis_cost


    # Powertrain (Comprised of: ice powertrain, electric drive system, fuel cell. Depending on the technology)
    top_up_down = ep.getTopUpDownData('ICE Powertrain', REGION, YEAR)
    mid = cost_series.loc[(REGION,'ICE Powertrain', weight_class), 'Mid Cost']*(1+top_up_down)
    high = cost_series.loc[(REGION,'ICE Powertrain', weight_class), 'High Cost']*(1+top_up_down)
    low = cost_series.loc[(REGION,'ICE Powertrain', weight_class), 'Low Cost']*(1+top_up_down)
    ice_powertrain_cost = cost_boolean.loc['ICE Powertrain']*ep.distributionFunction([low, mid, high], 'PERT', NUM_INVESTORS)
    top_up_down = ep.getTopUpDownData('Electric Drive System', REGION, YEAR)
    mid = cost_series.loc[(REGION,'Electric Drive System', weight_class), 'Mid Cost']*(1+top_up_down)
    high = cost_series.loc[(REGION,'Electric Drive System',weight_class), 'High Cost']*(1+top_up_down)
    low = cost_series.loc[(REGION,'Electric Drive System', weight_class), 'Low Cost']*(1+top_up_down)
    electric_drive_system_cost = cost_boolean.loc['Electric Drive System']*ep.distributionFunction([low, mid, high], 'PERT', NUM_INVESTORS)
    top_up_down = ep.getTopUpDownData('Fuel Cell System', REGION, YEAR)
    mid = cost_series.loc[(REGION,'Fuel Cell System', weight_class), 'Mid Cost']*(1+top_up_down)
    high = cost_series.loc[(REGION,'Fuel Cell System', weight_class), 'High Cost']*(1+top_up_down)
    low = cost_series.loc[(REGION,'Fuel Cell System', weight_class), 'Low Cost']*(1+top_up_down)
    fuel_cell_cost = cost_boolean.loc['Fuel Cell System']*ep.distributionFunction([low, mid, high], 'PERT', NUM_INVESTORS)


    powertrain_cost = power*ice_powertrain_cost*ice_power_coeff + power*electric_drive_system_cost*electric_motor_power_coeff + power*fuel_cell_cost*electric_motor_power_coeff

    # Introduce a percentage increase for the natural gas ICE engine component of the powertrain as compared to diesel (but only for the MDV and HDV segments)
    weight_class = weight_class
    if TECHNOLOGY == 'ICE-NG' and (weight_class == 'MDV' or weight_class == 'HDV'):
        natural_gas_engine_multiplier = 1.2                                     # For the MDV/HDV vehicles we increase the engine component of the powertrain cost by 20%
        engine_perc_of_powertrain = 0.85                                        # The % cost of the engine of the entire ICE powertrain (engine, exhaust, transmission)
        ice_powertrain_ng_cost = power*ice_powertrain_cost*engine_perc_of_powertrain*natural_gas_engine_multiplier + power*ice_powertrain_cost*(1-engine_perc_of_powertrain)
        powertrain_cost = ice_powertrain_ng_cost + power*electric_drive_system_cost*electric_motor_power_coeff + power*fuel_cell_cost*electric_motor_power_coeff

    # Energy Storage (Comprised of: battery, hydrogen tank, diesel tank, natural gas tank. Depending on the technology.)
    top_up_down = ep.getTopUpDownData('Li-ion Battery', REGION, YEAR)
    mid = cost_series.loc[(REGION,'Li-ion Battery', weight_class), 'Mid Cost']*(1+top_up_down)
    high = cost_series.loc[(REGION,'Li-ion Battery', weight_class), 'High Cost']*(1+top_up_down)
    low = cost_series.loc[(REGION,'Li-ion Battery', weight_class), 'Low Cost']*(1+top_up_down)
    battery_cost = cost_boolean.loc['Li-ion Battery']*ep.distributionFunction([low, mid, high], 'PERT', NUM_INVESTORS)
    top_up_down = ep.getTopUpDownData('Hydrogen Tank', REGION, YEAR)
    mid = cost_series.loc[(REGION,'Hydrogen Tank', weight_class), 'Mid Cost']*(1+top_up_down)
    high = cost_series.loc[(REGION,'Hydrogen Tank', weight_class), 'High Cost']*(1+top_up_down)
    low = cost_series.loc[(REGION,'Hydrogen Tank', weight_class), 'Low Cost']*(1+top_up_down)
    tank_h2_cost = cost_boolean.loc['Hydrogen Tank']*ep.distributionFunction([low, mid, high], 'PERT', NUM_INVESTORS)
    top_up_down = ep.getTopUpDownData('Diesel Tank', REGION, YEAR)
    mid = cost_series.loc[(REGION,'Diesel Tank', weight_class), 'Mid Cost']*(1+top_up_down)
    high = cost_series.loc[(REGION,'Diesel Tank', weight_class), 'High Cost']*(1+top_up_down)
    low = cost_series.loc[(REGION,'Diesel Tank', weight_class), 'Low Cost']*(1+top_up_down)
    tank_diesel_cost = cost_boolean.loc['Diesel Tank']*ep.distributionFunction([low, mid, high], 'PERT', NUM_INVESTORS)
    top_up_down = ep.getTopUpDownData('Natural Gas Tank', REGION, YEAR)
    mid = cost_series.loc[(REGION,'Natural Gas Tank', weight_class), 'Mid Cost']*(1+top_up_down)
    high = cost_series.loc[(REGION,'Natural Gas Tank', weight_class), 'High Cost']*(1+top_up_down)
    low = cost_series.loc[(REGION,'Natural Gas Tank', weight_class), 'Low Cost']*(1+top_up_down)
    tank_ng_cost = cost_boolean.loc['Natural Gas Tank']*ep.distributionFunction([low, mid, high], 'PERT', NUM_INVESTORS)

    energy_storage_cost = energy*tank_diesel_cost*tank_energy_coeff + energy*tank_ng_cost*tank_energy_coeff + energy*tank_h2_cost*tank_energy_coeff + energy*battery_cost*battery_energy_coeff

    # Profit Margin
    profit_margin = ep.getMonetaryParameters(REGION, 'Profit Margin')

    # Integration Factor (NOTE: this is only computed for non-incumbent technologies (i.e. all technologies except ICE-D))
    integration_factor = 0
    if TECHNOLOGY != 'ICE-D':
        if weight_class == 'LDV':
            integration_factor_title = 'Integration Factor - '+TECHNOLOGY+' - LDV'
        else:
            integration_factor_title = 'Integration Factor - '+TECHNOLOGY+' - MDV/HDV'
        # Determine the integration factor
        mid = cost_series.loc[(REGION, integration_factor_title, weight_class), 'Mid Cost']
        high = cost_series.loc[(REGION, integration_factor_title, weight_class), 'High Cost']
        low = cost_series.loc[(REGION, integration_factor_title, weight_class), 'Low Cost']
        integration_factor = cost_boolean.loc[integration_factor_title]*ep.distributionFunction([low, mid, high], 'PERT', NUM_INVESTORS)

    # Calculate the total CAPEX as per the equation below
    capex = ((powertrain_cost + energy_storage_cost + rest_of_truck_cost)*(1+integration_factor)/(1-profit_margin))

    # Establish dataframe of requested values for storage
    capex_parameters = pd.DataFrame([powertrain_cost*(1+integration_factor)/(1-profit_margin), energy_storage_cost*(1+integration_factor)/(1-profit_margin), rest_of_truck_cost*(1+integration_factor)/(1-profit_margin)], index = ep.getCAPEXRecordParameters())

    return [capex, capex_parameters]

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
