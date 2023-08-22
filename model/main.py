# FILE: main.py
# PROJECT: System-Dynamics Global Road Freight Model
# SCRIPT DESCRIPTION: This is the main module where the model should alwyas be
# run.

# Code Author: Bessie Noll
# Version Release: Version 1.0
# Version Release Date: ##/##/####
# Associated Publication: (DOI: )

"""
SYNTAX NOTES:

global variables:   GLOBAL_VARIABLES

function variables: function_variables

function definitions: functionDefinition

"""

# IMPORT PACKAGES
import numpy as np
import pandas as pd
import time
from tqdm import tqdm

start_time = time.time()

# Import and initialize the external parameters module
import externalParameters as ep
ep.initialize()
# Import and initialize the intermediate outputs module
import intermediateOutputs as io
io.initialize()
# Import and initialize the use case characterization module
import useCaseCharacterization as ucc
ucc.initialize()
# Import and initialize the experience curve module
import experienceCurves as ec
ec.initialize()
# Import the technology selection module
import technologySelection as ts

################################################################################
# Record the time to report how long it took to initialize the use case characterization parameters
ucc_time = time.time()
################################################################################

# Initialize the main data frame that stores the number of new trucks of a certain technology type added in each year, region, and application
df_main = ep.getEmptyDataFrame('technologies')

# Initialize the number of years the model will run for
years = ep.getYears()

with tqdm(total=years.shape[0]) as pbar:
    # Run the model by looping through all model years
    for y in years.itertuples(index=False):
        pbar.update(1)

        # Grab the YEAR subsection of the larger dataFrame
        year_frame = df_main.loc[(y[0], slice(None), slice(None)), :]

        # For each year, and in each region and application, simulate investor selection of available technologies
        year_frame = year_frame.apply(ts.investorSimulation, axis=1)

        # Save the year_frame to the main data frame
        df_main.loc[(y[0], slice(None), slice(None)), :] = year_frame

        # Record the year frame
        io.recordMainOutput(y[0], year_frame)

        # Update Experience Curves
        ec.updateLearningComponentCost(y[0], year_frame)

print('-------------------------------------------')
print(' ')
print('Finished the main code run.')
print(' ')
print('-------------------------------------------')
print(' ')

################################################################################
# Store things:

print('-------------------------------------------')
print(' ')
print('Storing data files ...')
print(' ')
print('-------------------------------------------')
print(' ')

# Store the main model output files (new vehicles sold, yearly market shares of new vehicles sold)
df_main.to_excel(ep.getFinalOutputDirectory() + '\_main_output.xlsx')
df_main.to_pickle(ep.getFinalOutputDirectory() + '\_MAIN_OUTPUT_DF.pkl')
io.storeMarketSharesDataframe()

# Store the TCO and related parameters dataframes
io.storeTCODataframe()
io.storeCAPEXDataframe()
io.storeOPEXDataframe()
io.storeCAPEXParametersDataframe()
io.storeOPEXParametersDataframe()
io.storeTCOParametersDataframe()

# Store the experience curve cost and annual capacity deployment dataframes
ec.storeEndogenousCapacityAdded()
ec.storeExogenousCapacityAdded()
ec.storeDynamicCostProgression()

# Store the switching cost dataframes
io.storeSwitchingCostMultiplierDataframe()
io.storeSwitchingCostDataframe()

###############################################################################
###############################################################################

print('---------------------------------------------------------------')
print('CODE RUN TIME:')
print('--- ',np.around((time.time() - start_time), decimals=3), 'seconds ---')
print('--- ',np.around((time.time() - start_time)/60, decimals=3), 'minutes ---')
print('(For ', ep.getNumInvestors(), ' investor(s).)')
print('   ')
print('UCC INITIALIZATION RUN TIME:')
print('--- ',np.around((ucc_time - start_time), decimals=3), 'seconds ---')
print('--- ',np.around((ucc_time - start_time)/60, decimals=3), 'minutes ---')
print('---------------------------------------------------------------')

###############################################################################
###############################################################################
"""
NOTE: If you would like the plot.py module to be called directly after running of
the main module, the code below can be uncommented. Otherwise, you can run the
plotting functions directly from the plot.py module by running the plot.py module
in the terminal.
"""
# # PLOT THINGS:
# import plots
# plots.runPlots()
#
# print('---------------------------------------------------------------')
# print('PLOT RUN TIME:')
# print('--- ',np.around((time.time() - start_time), decimals=3), 'seconds ---')
# print('--- ',np.around((time.time() - start_time)/60, decimals=3), 'minutes ---')
# print('(For ', ep.getNumInvestors(), ' investor(s).)')
# print('---------------------------------------------------------------')

######################################################################################################################
######################################################################################################################
