# FILE: experienceCurves.py
# PROJECT: Global System-Dynamics Freight Model
# MODULE DESCRIPTION: This module organizes and updates the experience curve data by updating
# regularly the cost and annual cumulative capacities deployment for each experience component.


# Import statements
import pandas as pd
import numpy as np
import externalParameters as ep
import useCaseCharacterization as ucc

# Global constants
BASE_YEAR = ep.getYears()['Years'][0]
LAST_YEAR = ep.getYears()['Years'].iloc[-1]
KILO_TO_GIGA = 10**(-6)

################################################################################
# EXTERNAL FUNCTIONS TO THE MODULE:
################################################################################

def initialize():
    """
    This function initializes all relevant global parameters used in the experienceCurves.py module.
    """

    # Initialize the global variables for endogenous and exogenous capacity deployment additions as well as the base capacity deployment
    global ENDOG_CAP_ADD_DF
    global EXOG_CAP_ADD_DF
    global BASE_CAP_DF

    # Setup the empty storage frames
    ENDOG_CAP_ADD_DF = ep.getEmptyDataFrame('learning components')
    EXOG_CAP_ADD_DF = pd.DataFrame(index=[np.repeat(ep.getYears()['Years'].values, len(ep.getRegionsString())), ep.getRegionsString()*(ep.getYears().size)], columns=ep.getLearningComponentsString(), dtype=np.float64)
    EXOG_CAP_ADD_DF.index.set_names(["Year", "Region"], level=[0,1], inplace=True)
    BASE_CAP_DF = pd.DataFrame(index=['Base'], columns=ep.getLearningComponentsString(), dtype=np.float64)
    for learning_component in ep.getLearningComponents():
        # Initialize the base installed capacity (constant)
        BASE_CAP_DF[learning_component] = ep.getExperienceComponentBaseCapacity(learning_component)

    # Initialize the global variables for the current cummulative capacity, the current cost for all learning components as well as the cost range percentages dataframe.
    global CURR_CUM_CAP_DF
    global CURR_COST_DF
    global COST_RANGE_PERCENTAGES

    # Setup the dataframe for the current cumulative capacity (by region)
    CURR_CUM_CAP_DF = pd.DataFrame(index=ep.getRegionsString(), columns=ep.getLearningComponentsString(), dtype=np.float64)
    for region in ep.getRegionsString():
        for learning_component in ep.getLearningComponents():
            # Initialize the base year cumulative installed capacity
            CURR_CUM_CAP_DF.loc[(region), learning_component] = ep.getExperienceComponentBaseCapacity(learning_component)
    # Setup the dataframe for the current cost (by region)
    first_geog_bool = True
    for region in ep.getRegionsString():
        # Create the region specific dataframe
        size = ep.getInitialCostData().reset_index().shape[0]
        curr_cost_df_i = ep.getInitialCostData().reset_index()
        curr_cost_df_i.insert(0, 'Region', np.repeat(region, size))

        # Add to the global dataframe
        if first_geog_bool:
            CURR_COST_DF = curr_cost_df_i
            first_geog_bool = False
        else:
            CURR_COST_DF = pd.concat([CURR_COST_DF, curr_cost_df_i], ignore_index=True)

    # Set the index for the current cost dataframe
    CURR_COST_DF.set_index(['Region', 'Component', 'Application'], inplace=True)

    # Initialize the cost range percentages which remain the same each year
    COST_RANGE_PERCENTAGES = ep.getInitialCostData()
    mid_cost = COST_RANGE_PERCENTAGES.loc[(slice(None), slice(None),slice(None)), 'Mid Cost']
    high_cost = COST_RANGE_PERCENTAGES.loc[(slice(None), slice(None),slice(None)), 'High Cost']
    low_cost = COST_RANGE_PERCENTAGES.loc[(slice(None), slice(None),slice(None)), 'Low Cost']
    COST_RANGE_PERCENTAGES.loc[(slice(None), slice(None),slice(None)), 'Mid Cost'] = mid_cost/mid_cost
    COST_RANGE_PERCENTAGES.loc[(slice(None), slice(None),slice(None)), 'High Cost'] = high_cost/mid_cost
    COST_RANGE_PERCENTAGES.loc[(slice(None), slice(None),slice(None)), 'Low Cost'] = low_cost/mid_cost

    # Initialize the dynamic cost dataframe which tracks learning component costs from year to year
    global DYNM_COST_DF

    # Initialize and record the mid cost point for each learning component
    DYNM_COST_DF = CURR_COST_DF.copy()
    DYNM_COST_DF = DYNM_COST_DF.rename(columns={'Mid Cost': 2020})
    DYNM_COST_DF.drop('High Cost', axis=1, inplace=True)
    DYNM_COST_DF.drop('Low Cost', axis=1, inplace=True)
    remaining_years = ep.getYears()['Years'][1:]
    DYNM_COST_DF[remaining_years] = 0

    # Sort all dataframe indicies
    ENDOG_CAP_ADD_DF.sort_index(inplace=True)
    EXOG_CAP_ADD_DF.sort_index(inplace=True)
    BASE_CAP_DF.sort_index(inplace=True)
    CURR_CUM_CAP_DF.sort_index(inplace=True)
    CURR_COST_DF.sort_index(inplace=True)
    COST_RANGE_PERCENTAGES.sort_index(inplace=True)
    DYNM_COST_DF.sort_index(inplace=True)

def getLearningComponentCost(region, application):
    """
    This function takes in an region and application and returns the cost for all learning
    components based on the CURR_COST_DF dataframe that is updated each model run year.
    """
    if application != 'None':
        # First get only the weight segment of the application as that is what the wage is dependent upon
        application = ep.getApplicationSegment(application, 'weight')

    return CURR_COST_DF.loc[(region, slice(None), application), :]

def updateLearningComponentCost(year, year_vehicle_installments):
    """
    This function updates the learning component cost based on addition deployment (both exogenous and endogenous) from the previous
    model year. The new cost of a learning component depends on the experience rate and the total new capacity deployment, which
    then moves the cost down the experience curve according to the Wright's Law formula. This function takes you through the process
    in 6 steps to help guide the logic. Some steps include calls to other internal functions within this module.
    """

    # Step 1): Get the average power and energy for each
    power = ucc.getAvgPowerEnergyValues(slice(None), slice(None), 'Power')        # [kW]
    energy = ucc.getAvgPowerEnergyValues(slice(None), slice(None), 'Energy')      # [kWh]

    # Step 2): Get annual power and energy specific capacity installments by converting year_vehicle_installments to power and energy values
    power_cap_installments = power.reset_index(level=0, drop=True)*year_vehicle_installments.reset_index(level=0, drop=True)*KILO_TO_GIGA               # [GW]
    energy_cap_installments = energy.reset_index(level=0, drop=True)*year_vehicle_installments.reset_index(level=0, drop=True)*KILO_TO_GIGA               # [GWh]

    # Step 3): Determine the total endogenous capacity installments
    # Loop through each region-application-drive-technology pair and determine installments of learning components
    global ENDOG_CAP_ADD_DF
    year_i_endog_cap_add = ENDOG_CAP_ADD_DF.loc[(year, slice(None), slice(None)), :].apply(learningComponentsDeployment, axis=1, args=[power_cap_installments, energy_cap_installments, year_vehicle_installments.reset_index(level=0, drop=True)])
    ENDOG_CAP_ADD_DF.loc[(year, slice(None), slice(None)), :] = year_i_endog_cap_add

    # Step 4): Determine total exogenous capacity installments
    global EXOG_CAP_ADD_DF
    for region in ep.getRegionsString():
        for component, value in EXOG_CAP_ADD_DF.loc[(year, region), :].iteritems():
            # Get the learning component scenario
            scenario = ep.getExperienceComponentScenario(component, 'Exogenous Market Scenario')

            # Get and update the exogenous capacity
            EXOG_CAP_ADD_DF.loc[(year, region), component] = ep.getExogenousMarketCapacityAdditions(component, scenario, year)*ep.getExogenousMarketContributions(component, region)

    # Step 5): Update the cumulative annual capacity installments
    global CURR_CUM_CAP_DF
    endog_additions = ENDOG_CAP_ADD_DF.loc[(year, slice(None), slice(None)), :].groupby(['YEAR', 'REGION']).sum().sort_index()
    exog_additions = EXOG_CAP_ADD_DF.loc[(year, slice(None)),:].sort_index()

    # Loop through the regions in order to account for potential regional isolation scenarios from a geopolitical shock
    for region in ep.getRegionsString():

        ##########################################################
        # DOMESTIC CAPACITY ADDITIONS
        ##########################################################
        # Determine the domestic endogenous and exogenous capacity additions for the current iteration region
        domestic_endog_additions = endog_additions.loc[(year, region), :]
        domestic_exog_additions = exog_additions.loc[(year, region), :]

        ##########################################################
        # NON-DOMESTIC CAPACITY ADDITIONS
        ##########################################################
        # Setup an array of all other regions other than the current iteration region
        all_other_regions = ep.getRegionsString()
        all_other_regions.remove(region)

        # Check to see if the current region is isolated to determine non_domestic endogenous and exogenous capacity additions.
        if ep.getGeopoliticalShockSelection(region) == 'Geopolitical Shock':
            # If the current region is isolated, all non-domestic additions should be zero.
            non_domestic_endog_additions = endog_additions.loc[(year, all_other_regions), :]*0
            non_domestic_exog_additions = exog_additions.loc[(year, all_other_regions), :]*0
        else:
            # If the current region is NOT isolated, all non-domestic additions are either zero (if other regions are isolated), or actual values (if other regions are not isolated)
            # Get the region specific isolation boolean array
            geopolitical_shock_matrix_bool_all_other_regions = ep.getGeopoliticalShockMatrixBool(all_other_regions, len(endog_additions.columns))
            # Calculate the non-domestic endog and exog capacity additions as a function of whether or not other regions are isolated
            non_domestic_endog_additions = endog_additions.loc[(year, all_other_regions), :]*geopolitical_shock_matrix_bool_all_other_regions.values
            non_domestic_exog_additions = exog_additions.loc[(year, all_other_regions), :]*geopolitical_shock_matrix_bool_all_other_regions.values

        ##########################################################
        # TOTAL CAPACITY ADDITIONS
        ##########################################################
        # Add the total (endogenous + exogenous) capacity additions to the cummulative capacity dataframe for region i
        total_endog_additions = domestic_endog_additions.values + sum(non_domestic_endog_additions.values)
        total_exog_additions = domestic_exog_additions.values + sum(non_domestic_exog_additions.values)

        # Add the total capacity additions to the tracking to the current cumulative capacity dataframe
        CURR_CUM_CAP_DF.loc[(region), :] += total_exog_additions + total_endog_additions

    # Step 6): Update the learning component costs
    global CURR_COST_DF
    CURR_COST_DF = CURR_COST_DF.apply(moveCostAlongExperienceCurve, axis=1, args=[year])

def storeEndogenousCapacityAdded():
    """
    This function stores the endogenous annual capacity additions for all experience components in each region.
    """
    ENDOG_CAP_ADD_DF.to_pickle(ep.getFinalOutputDirectory() + '\_ENDOG_CAP_ADD_DF.pkl')
    ENDOG_CAP_ADD_DF.to_excel(ep.getFinalOutputDirectory() + '\_endog_cap_add.xlsx')

def storeExogenousCapacityAdded():
    """
    This function stores the exogenous annual capacity additions for all experience components in each region.
    """
    EXOG_CAP_ADD_DF.to_pickle(ep.getFinalOutputDirectory() + '\_EXOG_CAP_ADD_DF.pkl')
    EXOG_CAP_ADD_DF.to_excel(ep.getFinalOutputDirectory() + '\_exog_cap_add.xlsx')

def storeDynamicCostProgression():
    """
    This function stores the annual cost progression of each experience component in each region.
    """
    DYNM_COST_DF.to_pickle(ep.getFinalOutputDirectory() + '\_DYNM_COST_DF.pkl')
    DYNM_COST_DF.to_excel(ep.getFinalOutputDirectory() + '\_dynm_cost.xlsx')

################################################################################
# INTERNAL FUNCTIONS TO THE MODULE:
################################################################################

def learningComponentsDeployment(learning_components_deployment_df, power_cap_installments, energy_cap_installments, year_vehicle_installments):
    """
    This function calculates the endogenous deployment of each learning component in the given year and in each region and application.
    The calculation takes the yearly new vehicle additions per technology type and then converts this into new power (GW) and energy (GWh)
    additions per learning component.
    """

    # Initialize the state
    year = learning_components_deployment_df.name[0]
    region = learning_components_deployment_df.name[1]
    application = learning_components_deployment_df.name[2]

    # Loop through all learning components
    for component, value in learning_components_deployment_df.iteritems():
        # Loop through all technologies
        for technology in ep.getTechnologiesString():
            # Check to see if the component exists in the technology (if not, skip to the next technology iteration)
            tech_boolean = ep.getCAPEXBoolean(technology)
            if tech_boolean[component] == 1:
                # Calculate the capacity additions for the integration factor components separately (because the units are different)
                if 'Integration Factor' in component:
                    if ep.getApplicationSegment(application, 'weight') in component:
                        learning_components_deployment_df[component] += year_vehicle_installments.loc[(region, application), technology]        # NOTE: this is units installed per vehicle (not in GWh or GW installed)
                # Calculate the capacity additions for all other learning components
                else:
                    # Determine the energy ratio
                    percent_tank = ep.getTechnologyData(technology, application, 'Tank Energy Coeff')*ep.getExperienceComponentBoolean(component, 'Tank')
                    percent_battery = ep.getTechnologyData(technology, application, 'Battery Energy Coeff')*ep.getExperienceComponentBoolean(component, 'Battery')
                    # Determine the power ratios
                    percent_ice = ep.getTechnologyData(technology, application, 'ICE Power Coeff')*ep.getExperienceComponentBoolean(component, 'ICE')
                    percent_e_motor = ep.getTechnologyData(technology, application, 'Electric Motor Power Coeff')*ep.getExperienceComponentBoolean(component, 'E-Motor')
                    # Add the energy- and power-specific capacity additions to the storage dataframe
                    learning_components_deployment_df[component] += energy_cap_installments.loc[(region, application), technology]*(percent_tank + percent_battery) + power_cap_installments.loc[(region, application), technology]*(percent_ice + percent_e_motor)

    return learning_components_deployment_df

def moveCostAlongExperienceCurve(cost_array, curr_year):
    """
    This function updates the new cost for all experience components based on the new capacity additions. The new cost
    is calculated via Wright's Law which connects historic product prices to cumulative deployed capacities.
    """

    # Initialize the state
    region = cost_array.name[0]
    component = cost_array.name[1]
    application = cost_array.name[2]

    # Initialize the global variables
    global DYNM_COST_DF
    global COST_RANGE_PERCENTAGES
    global BASE_CAP_DF
    global CURR_CUM_CAP_DF

    # Get the base year cost (initial cost) and the base year capacity installments
    base_cost_mid = DYNM_COST_DF.loc[(region, component, application), BASE_YEAR]
    base_cap = BASE_CAP_DF.loc['Base',component]

    # Get the new cummulative installed capacity, that was updated in the learningComponentsDeployment() function above, for the given region and component
    new_cap = CURR_CUM_CAP_DF.loc[(region), component]

    # Get the learning parameter 'b'
    scenario = ep.getExperienceComponentScenario(component, 'Experience Rate Scenario')
    learning_parameter_scenario = 'Learning Parameter ' + scenario
    b = ep.getExperienceComponentData(component, application, learning_parameter_scenario)

    # Calculate the new experience component cost
    new_cost_mid = base_cost_mid*((new_cap/base_cap)**(-b))
    new_cost_high = new_cost_mid*COST_RANGE_PERCENTAGES.loc[(component, application), 'High Cost']
    new_cost_low = new_cost_mid*COST_RANGE_PERCENTAGES.loc[(component, application), 'Low Cost']

    # Add the new cost to the cost_array and return
    cost_array['Mid Cost'] = new_cost_mid
    cost_array['High Cost'] = new_cost_high
    cost_array['Low Cost'] = new_cost_low

    # Update the dynamic cost dataframe for storage
    if curr_year != LAST_YEAR:
        DYNM_COST_DF.at[(region, component, application), int(curr_year)+1] = cost_array['Mid Cost']

    return cost_array






























# end
