# FILE: plots.py
# PROJECT: Global System-Dynamics Freight Model
# MODULE DESCRIPTION: This is the plotting module that takes the output files
# from the main iteration scheme and plots accordingly.

# Import statements
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
import externalParameters as ep
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from scipy.stats import norm
from matplotlib.colors import to_rgba
import statsmodels.api as sm
import os
import time
import math

start_time = time.time()

# Initialize the external parameters module
ep.initialize()

################################################################################
################################################################################
"""
Initialize plotting variables (i.e. colors, opacities, units, etc.) for all plot
functions.
"""
# Initialize the base year
BASE_YEAR = ep.getYears()['Years'][0]

dpi=300

# Colors for the different technologies
ICE_D_c = "saddlebrown"
BEV_c = "tab:blue"
FCEV_c = "tab:green"
ICE_NG_c = "gold"
ZEV_c = "#00FFFF"

# Opacities for different technologies
ICE_D_op = 1.0
BEV_op = 1.0
ICE_NG_op = 1.0
FCEV_op = 1.0


# Python default colors list
prop_cycle = plt.rcParams['axes.prop_cycle']
python_default_colors = prop_cycle.by_key()['color']

# Organize the desired order of displayed technologies
tech_order = ep.getTechnologies().loc[:,'Technology']

# Global Constants
numRegions = ep.getRegions().size
numApplications = ep.getApplications().size
numTechnologies = ep.getTechnologies().size

# Plotting Constants
ticklabelpad = mpl.rcParams['xtick.major.pad']

# Set the region colors
def getRegionColor(region):

    if region == 'EU':
        return 'darkblue'
    if region == 'US':
        return 'blueviolet'
    if region == 'China':
        return 'tab:red'
    if region == 'India':
        return 'darkorange'
    if region == 'Brazil':
        return 'darkgreen'
    if region == 'Rest of World':
        return 'grey'

# Set the technology colors
def getTechnologyColor(technology):

    if technology == 'BEV':
        return BEV_c
    if technology == 'FCEV':
        return FCEV_c
    if technology == 'ICE-D':
        return ICE_D_c
    if technology == 'ICE-NG':
        return ICE_NG_c
    if technology == 'ZEV':
        return ZEV_c

# Return a list of technology colors
def getTechnologyColorList(technology_list):
    color_list = []
    for technology in technology_list:
        color_list.append(getTechnologyColor(technology))

    return color_list

# Set the technology opacities
def getTechnologyOpacity(technology):

    if technology == 'BEV':
        return BEV_op
    if technology == 'FCEV':
        return FCEV_op
    if technology == 'ICE-D':
        return ICE_D_op
    if technology == 'ICE-NG':
        return ICE_NG_op

# Return a list of technology opacities
def getTechnologyOpacityList(technology_list):
    opacity_list = []
    for technology in technology_list:
        opacity_list.append(getTechnologyOpacity(technology))

    return opacity_list

# Set the component units
def getComponentUnit(component, type):

    if type == 'Capacity':
        if component == 'Li-ion Battery':
            return '[GWh]'
        if component == 'Hydrogen Tank':
            return '[GWh]'
        if component == 'Diesel Tank':
            return '[GWh]'
        if component == 'Natural Gas Tank':
            return '[GWh]'
        if component == 'Electric Drive System':
            return '[GW]'
        if component == 'Fuel Cell System':
            return '[GW]'
        if component == 'ICE Powertrain':
            return '[GW]'
    if type == 'Price':
        if component == 'Li-ion Battery':
            return '[USD/kWh]'
        if component == 'Hydrogen Tank':
            return '[USD/kWh]'
        if component == 'Diesel Tank':
            return '[USD/kWh]'
        if component == 'Natural Gas Tank':
            return '[USD/kWh]'
        if component == 'Electric Drive System':
            return '[USD/kW]'
        if component == 'Fuel Cell System':
            return '[USD/kW]'
        if component == 'ICE Powertrain':
            return '[USD/kW]'

def getAreaChartTechnologyList():
    technology_list_from_model = ep.getTechnologiesString()

    # Initialize the desired technology list order
    desired_technology_list = ['FCEV', 'BEV', 'ICE-NG', 'ICE-D']

    return desired_technology_list

################################################################################
################################################################################
"""
Market Share Plots:
Plot types
    1): Stacked Area Charts
    2): Stacked Area Charts for the weight/range segments
    3): Stacked Area Charts for region-specific application aggregate
"""

def stackedAreaChartPlots(region, market_shares_df, filename_addition, output_plots_directory):

    # Setup the figure
    nrow=3
    ncol=3

    fig, axes = plt.subplots(nrows=nrow, ncols=ncol, sharey=True, sharex=True)

    # Initialize the years, applications and technologies
    years = np.transpose(ep.getYears().to_numpy())[0]
    application_list = ep.getApplicationsString()
    technology_list = getAreaChartTechnologyList()

    colors_list = getTechnologyColorList(technology_list)
    for i, ax in enumerate(axes.flatten()):
        # If we are ploting a specific region's market shares, then we enter the first if...
        if region != 'Global':
            # PLOT WITH 4 TECHNOLOGIES
            # Get the individual data columns for drive technologies in the specific application segment
            tech_array_1 = market_shares_df.loc[(slice(None), region, application_list[i]), technology_list[0]].values*100
            tech_array_2 = market_shares_df.loc[(slice(None), region, application_list[i]), technology_list[1]].values*100
            tech_array_3 = market_shares_df.loc[(slice(None), region, application_list[i]), technology_list[2]].values*100
            tech_array_4 = market_shares_df.loc[(slice(None), region, application_list[i]), technology_list[3]].values*100
            ax.stackplot(years, np.array(tech_array_1, dtype=float), np.array(tech_array_2, dtype=float), np.array(tech_array_3, dtype=float), np.array(tech_array_4, dtype=float),  colors=colors_list)
        else:
            # PLOT WITH 4 TECHNOLOGIES
            # Get the individual data columns for drive technologies in the specific application segment
            tech_array_1 = market_shares_df.loc[(slice(None), application_list[i]), technology_list[0]].values*100
            tech_array_2 = market_shares_df.loc[(slice(None), application_list[i]), technology_list[1]].values*100
            tech_array_3 = market_shares_df.loc[(slice(None), application_list[i]), technology_list[2]].values*100
            tech_array_4 = market_shares_df.loc[(slice(None), application_list[i]), technology_list[3]].values*100
            ax.stackplot(years, np.array(tech_array_1, dtype=float), np.array(tech_array_2, dtype=float), np.array(tech_array_3, dtype=float), np.array(tech_array_4, dtype=float), colors=colors_list)

        ax.set_title(application_list[i])
        # Only set the y-label on the left sub-plots
        if (i % 3) == 0:
            ax.set_ylabel('Market Share (%)')
        ax.margins(0, 0) # Set margins to avoid "whitespace"

        # Set x-axis tick label intervals
        ax.set_xticks([2020, 2025, 2030, 2035])#np.arange(years[0], years[-1], 4))

        # Set the y-axis bounds
        ax.set_ylim((0,100))

    # Set title and filename
    if filename_addition == 'None':
        fig.suptitle(region)
        filename = output_plots_directory + '\_market_shares_' + region + '.png'
    else:
        fig.suptitle(region + ' : ' + filename_addition)
        filename = output_plots_directory + '\_market_shares_' + region + '_' + filename_addition + '.png'

    # Setup the legend
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(technology_list, loc='lower center', ncol = len(technology_list))

    fig.subplots_adjust(left=0.1, bottom=0.095, right=0.968, top=0.91, wspace=0.2, hspace=0.26)
    fig.set_size_inches(12,8)

    # Save the figure
    plt.savefig(filename, dpi=dpi)

def stackedAreaChartSegmentPlots(region_selection, df_main, filename_addition, output_plots_directory):
    """
    ############################################################################
    ### NOTE: df_main is the actual number of trucks of each technology type, not the market shares
    ############################################################################
    """

    years = np.transpose(ep.getYears().to_numpy())[0]
    # Region list
    if region_selection == 'All':
        regions = pd.unique(df_main.index.get_level_values('REGION'))
    else:
        regions = region_selection
    applications = ep.getApplicationsString()
    # Application segment list
    application_segment_list = ['LDV', 'MDV', 'HDV']
    # Specific weight segment list
    ldv_segments = ['LDV-Urban', 'LDV-Regional', 'LDV-LongHaul']
    mdv_segments = ['MDV-Urban', 'MDV-Regional', 'MDV-LongHaul']
    hdv_segments = ['HDV-Urban', 'HDV-Regional', 'HDV-LongHaul']
    # Technologies
    technology_list = getAreaChartTechnologyList()
    # Establish the colors for the technologies
    colors_list = getTechnologyColorList(technology_list)
    opacity_list = 1


    # Reorganize the applications according to the weight segments (LDV, MDV, HDV)
    column_list = np.append(np.array(['YEAR', 'REGION', 'APPLICATION']), df_main.columns.to_numpy())
    df_plot = pd.DataFrame(columns=column_list).set_index(['YEAR', 'REGION', 'APPLICATION'])
    for year_i in years:
        for region_i in regions:
            for application_segment_i in application_segment_list:
                if application_segment_i == 'LDV':
                    segment_list = ldv_segments
                elif application_segment_i == 'MDV':
                    segment_list = mdv_segments
                elif application_segment_i == 'HDV':
                    segment_list = hdv_segments

                df_plot.loc[(year_i,region_i,application_segment_i), :] = df_main.loc[(year_i,region_i,segment_list), :].sum()

    # Initialize the figure and sub-figures depending on the plot type
    if region_selection == 'All':
        # Initialize the figure
        nrow=3
        ncol=len(regions)
        fig, axes = plt.subplots(nrows=nrow, ncols=ncol, sharey=True, sharex=True)

        fig_size_x = 16
        fig_size_y = 10
        border_left = 0.1
        border_right = 0.968
        wspace = 0.2

        # Setup the region and application arrays when plotting all regions on one plot
        region_array = np.append(np.append(regions, regions), regions)
        application_segment_array = np.append(np.append(np.repeat('LDV', len(regions)), np.repeat('MDV', len(regions))), np.repeat('HDV', len(regions)))

    elif len(region_selection) == 1:
        # Initialize the figure
        nrow=3
        ncol=1
        fig, axes = plt.subplots(nrows=nrow, ncols=ncol, sharey=True, sharex=True)

        fig_size_x = 5
        fig_size_y = 10
        border_left = 0.140
        border_right = 0.913
        wspace = 0.2

        # Setup the region and application arrays when plotting all regions on one plot
        region_array = np.repeat(regions, 3)
        application_segment_array = application_segment_list

    elif len(region_selection) > 1:
        # Initialize the figure
        nrow=3
        ncol=len(region_selection)
        fig, axes = plt.subplots(nrows=nrow, ncols=ncol, sharey=True, sharex=True)

        fig_size_x = (16/6)*len(region_selection)
        fig_size_y = 10
        border_left = 0.145
        border_right = 0.958
        wspace = 0.255

        # Setup the region and application arrays when plotting all regions on one plot
        region_array = np.append(np.append(regions, regions), regions)
        application_segment_array = np.append(np.append(np.repeat('LDV', len(region_selection)), np.repeat('MDV', len(region_selection))), np.repeat('HDV', len(region_selection)))

    # Plot the data
    for i, ax in enumerate(axes.flatten()):
        # Sum to get the total vehicles in this region and segment
        total_vehicles = df_plot.loc[(slice(None), region_array[i], application_segment_array[i]), :].sum(axis=1).values

        # PLOT WITH 4 TECHNOLOGIES
        # Get the individual data columns for drive technologies in the specific application segment
        tech_array_1 = (df_plot.loc[(slice(None), region_array[i], application_segment_array[i]), technology_list[0]].values/total_vehicles)*100
        tech_array_2 = (df_plot.loc[(slice(None), region_array[i], application_segment_array[i]), technology_list[1]].values/total_vehicles)*100
        tech_array_3 = (df_plot.loc[(slice(None), region_array[i], application_segment_array[i]), technology_list[2]].values/total_vehicles)*100
        tech_array_4 = (df_plot.loc[(slice(None), region_array[i], application_segment_array[i]), technology_list[3]].values/total_vehicles)*100
        ax.stackplot(years, np.array(tech_array_1, dtype=float), np.array(tech_array_2, dtype=float), np.array(tech_array_3, dtype=float), np.array(tech_array_4, dtype=float),  colors=colors_list, alpha=opacity_list)

        if application_segment_array[i] == application_segment_list[0]:
            ax.set_title(region_array[i], fontsize=14)
        # Only set the y-label on the left sub-plots
        if region_array[i] == regions[0]:
            ax.set_ylabel(application_segment_array[i] + '\n \n Market Share', fontsize=14)
            # Set the y-axis units
            ax.annotate('[%]', xy=(0,1), xytext=(10, ticklabelpad+15), ha='right', va='top',xycoords='axes fraction', textcoords='offset points')


    # SET THE AXES PARAMETERS
    # Set the margins
    ax.margins(0, 0) # Set margins to avoid "whitespace"
    # Set x-axis tick label intervals
    ax.set_xticks([2020, 2025, 2030, 2035])#np.arange(years[0], years[-1], 4))
    # Set the y-axis bounds
    ax.set_ylim((0,100))

    # Set the title
    fig.suptitle(filename_addition)

    # Set the legend
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(technology_list, loc='lower center', ncol = len(technology_list))

    # Set the figure size
    fig.subplots_adjust(left=border_left, bottom=0.095, right=border_right, top=0.91, wspace=wspace, hspace=0.26)
    fig.set_size_inches(fig_size_x,fig_size_y)

    # Save the figure
    filename = output_plots_directory + '\_shares_ws_' + filename_addition + '.png'
    plt.savefig(filename, dpi=dpi)

def stackedAreaChartPlotsSingle(region, market_shares_df, filename_addition, output_plots_directory):

    # Setup the figure
    nrow=1
    ncol=1

    fig, ax = plt.subplots(nrows=nrow, ncols=ncol, sharey=True, sharex=True)

    # Initialize the years and the technologies
    years = np.transpose(ep.getYears().to_numpy())[0]
    technology_list = getAreaChartTechnologyList()

    colors_list = getTechnologyColorList(technology_list)

    # If we are ploting a specific region's market shares, then we enter the first if...
    if region != 'Global':
        # PLOT WITH 4 TECHNOLOGIES
        # Get the individual data columns for drive technologies in the specific application segment
        tech_array_1 = market_shares_df.loc[(slice(None), region), technology_list[0]].values*100
        tech_array_2 = market_shares_df.loc[(slice(None), region), technology_list[1]].values*100
        tech_array_3 = market_shares_df.loc[(slice(None), region), technology_list[2]].values*100
        tech_array_4 = market_shares_df.loc[(slice(None), region), technology_list[3]].values*100
        ax.stackplot(years, np.array(tech_array_1, dtype=float), np.array(tech_array_2, dtype=float), np.array(tech_array_3, dtype=float), np.array(tech_array_4, dtype=float),  colors=colors_list)
    else:
        # PLOT WITH 4 TECHNOLOGIES
        # Get the individual data columns for drive technologies in the specific application segment
        tech_array_1 = market_shares_df.loc[(slice(None)), technology_list[0]].values*100
        tech_array_2 = market_shares_df.loc[(slice(None)), technology_list[1]].values*100
        tech_array_3 = market_shares_df.loc[(slice(None)), technology_list[2]].values*100
        tech_array_4 = market_shares_df.loc[(slice(None)), technology_list[3]].values*100
        ax.stackplot(years, np.array(tech_array_1, dtype=float), np.array(tech_array_2, dtype=float), np.array(tech_array_3, dtype=float), np.array(tech_array_4, dtype=float), colors=colors_list)

    # Set the axis labels amd margins
    ax.set_title('Application Segment Aggregate', loc='left')
    ax.set_ylabel('Market Share (%)')
    ax.margins(0, 0) # Set margins to avoid "whitespace"

    # Set x-axis tick label intervals
    ax.set_xticks([2020, 2025, 2030, 2035])#np.arange(years[0], years[-1], 4))

    # Set the y-axis bounds
    ax.set_ylim((0,100))

    # Set figure title and filename
    if filename_addition == 'None':
        fig.suptitle(region)
        filename = output_plots_directory + '\_market_shares_' + region + '.png'
    else:
        fig.suptitle(region + ' : ' + filename_addition)
        filename = output_plots_directory + '\_market_shares_' + region + '_' + filename_addition + '.png'

    # Set the legend
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(technology_list, loc='lower center', ncol = len(technology_list))

    # Set the figure size
    fig.subplots_adjust(left=0.1, bottom=0.095, right=0.968, top=0.91, wspace=0.2, hspace=0.26)
    fig.set_size_inches(12,10)

    # Save the figure
    plt.savefig(filename, dpi=dpi)

################################################################################
"""
TCO Plots:
Plot types
    1): Region-specific TCO stacked bar plots for each applicaiton segment (all technologies included in each sub-plot)
    2): Dynamic TCO plots for each application segment (all technologies included in each sub-plot)
"""

def tcoStackedBarPlot(year, region):

    # Setup the figure
    nrow=3
    ncol=3

    fig, axes = plt.subplots(nrows=nrow, ncols=ncol, sharex=True)

    # Initialize the application and technologies
    application = ep.getApplicationsString()
    tech_order = ep.getTechnologies().loc[:,'Technology']

    # Initialize the tick parameter label rotation
    rotation = 40

    # Initialize other minor plot details
    bar_width = 0.5
    cap_thick = 1.5
    cap_size = 2.5
    epsilon = .015
    line_width = 1
    opacity = 0.7

    for i, ax in enumerate(axes.flatten()):

        # Setup a new dataframe for restructuring
        tco_parameters_all = pd.DataFrame(columns=['Technology','CAPEX (minus subsidy and SV)', 'Insurance', 'O&M', 'Tolls', 'Wages', 'Fuel Costs', 'Infrastructure Costs', 'Carbon Costs', 'Switching Costs'])
        for technology in ep.getTechnologiesString():
            # Set the scale factor
            scale_factor = 1000
            mean = TCO_PARAMETERS_DF_MEAN.loc[(year, region, application[i], slice(None)), technology]
            capex = mean.loc[(year, region, application[i], 'Power Train')] + mean.loc[(year, region, application[i], 'Energy Storage')] + mean.loc[(year, region, application[i], 'Rest of Truck')] \
                 - mean.loc[(year, region, application[i], 'CAPEX Subsidy')] - mean.loc[(year, region, application[i], 'Scrappage Value')]
            insurance = mean.loc[(year, region, application[i], 'Insurance')]
            o_and_m = mean.loc[(year, region, application[i], 'O & M')]
            tolls = mean.loc[(year, region, application[i], 'Tolls')]
            wages = mean.loc[(year, region, application[i], 'Wages')]
            fuel_costs = mean.loc[(year, region, application[i], 'Fuel Costs')]
            infrastructure_costs = mean.loc[(year, region, application[i], 'Infrastructure Costs')]
            carbon_costs = mean.loc[(year, region, application[i], 'Carbon Costs')]
            switching_costs = mean.loc[(year, region, application[i], 'Switching Cost')]
            dict = {'Technology':[technology],
                     'CAPEX (minus subsidy and SV)':[capex],
                     'Insurance':[insurance],
                     'O&M':[o_and_m],
                     'Tolls':[tolls],
                     'Wages':[wages],
                     'Fuel Costs':[fuel_costs],
                     'Infrastructure Costs':[infrastructure_costs],
                     'Carbon Costs':[carbon_costs],
                     'Switching Costs':[switching_costs]}
            df_dict = pd.DataFrame(dict)
            tco_parameters_all = pd.concat([tco_parameters_all, df_dict], ignore_index=True)


        tco_parameters_all.set_index(['Technology'], inplace=True)

        # Plot the dataframe
        plot_df = tco_parameters_all.loc[(slice(None)), :]
        plot_df.plot(kind='bar', stacked=True, ax=ax, legend=False)

        # Set the subplot title
        ax.set_title(application[i], size=10, loc='left')
        # Set the x-labels
        ax.set_xlabel(' ', labelpad=30)
        ax.tick_params(axis='x', rotation=rotation)
        # Set the y-labels
        if ep.getApplicationSegment(application[i], 'range') == 'Urban' :
            ax.set_ylabel('USD/km', labelpad=15)

        # Set the legend
        if application[i] == 'HDV-Regional':
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.6), ncol = 8)

    # Set the figure title
    fig.suptitle(region + ' : ' + str(year) + ' : Stacked TCO')

    # Set the figure size
    fig.subplots_adjust(left=0.065, bottom=0.14, right=0.94, top=0.88, wspace=0.225, hspace=0.4)
    fig.set_size_inches(14,10)

    # Save the figure
    filename = ep.getOutputPlotsDirectory() + '\_' + region + '_' + str(year) + '_tco_stacked.png'
    plt.savefig(filename, dpi=dpi)

def tcoDynamicLinePlot(region):

    # Setup the figure
    nrow=3
    ncol=3

    fig, axes = plt.subplots(nrows=nrow, ncols=ncol, sharex=True)#, gridspec_kw={'width_ratios': widths, 'height_ratios':heights})

    # Initialize the applications and technologies
    application = ep.getApplicationsString()
    tech_order = ep.getTechnologies().loc[:,'Technology']

    # Initialize the tick parameter rotation and the standard deviation for error bars
    rotation = 40
    num_std = 1

    # Initialize other minor plot details
    bar_width = 0.5
    cap_thick = 1.5
    cap_size = 2.5
    epsilon = .015
    line_width_low = 0.05
    opacity = 0.7

    for i, ax in enumerate(axes.flatten()):

        # Setup a new dataframe for restructuring
        plot_df_mean = pd.DataFrame(index=ep.getYearsString(), columns=ep.getTechnologiesString())
        plot_df_high = pd.DataFrame(index=ep.getYearsString(), columns=ep.getTechnologiesString())
        plot_df_low = pd.DataFrame(index=ep.getYearsString(), columns=ep.getTechnologiesString())

        # Get the mean TCO for all years
        tco_mean = TCO_DF_MEAN.loc[(slice(None), region, application[i]), :]
        tco_std = TCO_DF_STD.loc[(slice(None), region, application[i]), :]

        plot_df_mean.loc[(slice(None)), :] = tco_mean.values
        plot_df_high.loc[(slice(None)), :] = tco_mean.values + num_std*tco_std.values
        plot_df_low.loc[(slice(None)), :] = tco_mean.values - num_std*tco_std.values

        for technology in ep.getTechnologiesString():
            # Plot the dataframe
            plot_df_mean.loc[(slice(None)), technology].plot(kind='line', ax=ax, legend=False, label=technology, color=getTechnologyColor(technology))
            plot_df_high.loc[(slice(None)), technology].plot(kind='line', ax=ax, legend=False, label='_Hidden', linewidth=line_width_low, color=getTechnologyColor(technology))
            plot_df_low.loc[(slice(None)), technology].plot(kind='line', ax=ax, legend=False, label='_Hidden', linewidth=line_width_low, color=getTechnologyColor(technology))
            try:
                ax.fill_between(plot_df_mean.index, plot_df_high.loc[(slice(None)), technology], plot_df_low.loc[(slice(None)), technology], alpha=0.1, color=getTechnologyColor(technology))
            except:
                continue

        # Set the subplot title
        ax.set_title(application[i], size=10, loc='left')
        # Set the x-labels
        ax.set_xlabel(' ', labelpad=30)
        # Set x-axis tick label intervals
        ax.set_xticks(plot_df_mean.index)
        ax.tick_params(axis='x', rotation=rotation)
        # Set the y-labels
        if ep.getApplicationSegment(application[i], 'range') == 'Urban' :
            ax.set_ylabel('USD/km', labelpad=15)

        # Set the figure legend
        if application[i] == 'HDV-Regional':
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.6), ncol = 6)

    # Set the figure title
    fig.suptitle(region + ' : TCO Dynamic Lines (' + str(num_std) + ' std error)')

    # Set the figure size
    fig.subplots_adjust(left=0.065, bottom=0.14, right=0.94, top=0.88, wspace=0.145, hspace=0.2)
    fig.set_size_inches(14,10)

    # Save the figure
    filename = ep.getOutputPlotsDirectory() + '\_' + region + '_tco_dynamic_lines.png'
    plt.savefig(filename, dpi=dpi)

################################################################################
"""
CAPEX Plots:
Plot types
    1): CAPEX bar plots
    2): CAPEX dynamic line plots
"""

def CAPEXPlot(year, region):

    # Setup the figure
    nrow=3
    ncol=3

    fig, axes = plt.subplots(nrows=nrow, ncols=ncol, sharey=True)

    # Initialize the application and technology order
    application = ep.getApplicationsString()
    tech_order = ep.getTechnologies().loc[:,'Technology']

    rotation = 0

    colors = getTechnologyColorList(tech_order)

    hatch1 = '//'
    hatch2 = 'OO'

    # plot details
    bar_width = 0.5
    cap_thick = 1.5
    cap_size = 2.5
    epsilon = .015
    line_width = 1
    opacity = 0.7

    for i, ax in enumerate(axes.flatten()):
        # Setup the dataframe first for proper plotting
        capex_df_mean = pd.DataFrame(columns=['Mean', 'Min', 'Max', 'Std'])
        capex_df_std = pd.DataFrame(columns=['Mean', 'Min', 'Max', 'Std'])
        capex_parameters_df_mean = pd.DataFrame(columns=ep.getCAPEXRecordParameters())
        capex_parameters_df_std = pd.DataFrame(columns=ep.getCAPEXRecordParameters())
        for technology in ep.getTechnologiesString():
            capex_df_mean.loc[technology] = CAPEX_DF_MEAN.loc[(year, region, application[i]), technology]
            capex_df_std.loc[technology] = CAPEX_DF_STD.loc[(year, region, application[i]), technology]
            capex_parameters_df_mean.loc[technology] = CAPEX_PARAMETERS_DF_MEAN.loc[(year, region, application[i], slice(None)), technology].values
            capex_parameters_df_std.loc[technology] = CAPEX_PARAMETERS_DF_STD.loc[(year, region, application[i], slice(None)), technology].values

        # Scale the dataframes
        scale_factor = 1000
        capex_df_mean = capex_df_mean/scale_factor
        capex_df_std = capex_df_std/scale_factor
        capex_parameters_df_mean = capex_parameters_df_mean/scale_factor
        capex_parameters_df_std = capex_parameters_df_std/scale_factor

        rest_of_truck = capex_parameters_df_mean["Rest of Truck"].values
        powertrain = capex_parameters_df_mean["Power Train"].values
        energy_storage = capex_parameters_df_mean["Energy Storage"].values

        # make bar plots
        rest_of_truck_bar = ax.barh(tech_order, rest_of_truck, bar_width,
                                  color=colors,
                                  label='Rest of Truck')
                                  #ax=ax1)
        powertrain_bar = ax.barh(tech_order, powertrain, bar_width-epsilon,
                                  left=rest_of_truck,
                                  alpha=opacity,
                                  color='white',
                                  edgecolor=colors,
                                  linewidth=line_width,
                                  hatch=hatch1,
                                  label='Powertrain')
                                  #ax=ax1)
        energy_storage_bar = ax.barh(tech_order, energy_storage, bar_width-epsilon,
                                   left=rest_of_truck+powertrain,
                                   alpha=opacity,
                                   color='white',
                                   edgecolor=colors,
                                   linewidth=line_width,
                                   hatch=hatch2,
                                   label='Energy Storage')
                                   #ax=ax1)

        ax.set_title(application[i], size=10, loc='left')
        ax.tick_params(rotation=rotation, labelsize=7)
        ax.errorbar(capex_df_mean['Mean'], capex_df_mean['Mean'].index, xerr=capex_df_std['Std']*2, fmt='o', color='Black', elinewidth=line_width, capthick=cap_thick, errorevery=1, alpha=1, ms=4, capsize=cap_size)
        ax.annotate('[kUSD]', xy=(1,0), xytext=(5, -ticklabelpad), ha='left', va='top',xycoords='axes fraction', textcoords='offset points')
        if application[i] == 'HDV-Regional':
            leg_a = mpatches.Patch(facecolor='dimgrey', label='Rest of Truck')
            leg_b = mpatches.Patch(alpha=opacity, facecolor='white', edgecolor='dimgrey', linewidth=line_width, hatch=hatch1, label='Powertrain')
            leg_c = mpatches.Patch(alpha=opacity, facecolor='white', edgecolor='dimgrey', linewidth=line_width, hatch=hatch2, label='Energy Storage')
            ax.legend(handles=[leg_a,leg_b,leg_c], loc='lower center', bbox_to_anchor=(0.5, -0.5), ncol=3)

    # Set the figure title
    fig.suptitle(region + ' CAPEX : ' + str(year))

    # Set the figure size
    fig.subplots_adjust(left=0.065, bottom=0.14, right=0.94, top=0.88, wspace=0.225, hspace=0.4)
    fig.set_size_inches(14,10)

    # Save the figure
    filename = ep.getOutputPlotsDirectory() + '\_' + region + '_CAPEX_' + str(year) + '.png'
    plt.savefig(filename, dpi=dpi)

def CAPEXDynamicLinePlot(region):

    # Setup the figure
    nrow=3
    ncol=3

    fig, axes = plt.subplots(nrows=nrow, ncols=ncol, sharex=True)

    # Initialize the application and technology order
    application = ep.getApplicationsString()
    tech_order = ep.getTechnologies().loc[:,'Technology']

    # Initialize the tick parameter lable rotation and the number of standard deviations for error plotting
    rotation = 40
    num_std = 1

    # Initialize other minor plot details
    bar_width = 0.5
    cap_thick = 1.5
    cap_size = 2.5
    epsilon = .015
    line_width_low = 0.05
    opacity = 0.7

    for i, ax in enumerate(axes.flatten()):

        # Setup a new dataframe for restructuring
        plot_df_mean = pd.DataFrame(index=ep.getYearsString(), columns=ep.getTechnologiesString())
        plot_df_high = pd.DataFrame(index=ep.getYearsString(), columns=ep.getTechnologiesString())
        plot_df_low = pd.DataFrame(index=ep.getYearsString(), columns=ep.getTechnologiesString())

        # Get the mean TCO for all years
        capex_mean = CAPEX_DF_MEAN.loc[(slice(None), region, application[i]), :]/1000
        capex_std = CAPEX_DF_STD.loc[(slice(None), region, application[i]), :]/1000

        # Organize the plotting dataframe
        plot_df_mean.loc[(slice(None)), :] = capex_mean.values
        plot_df_high.loc[(slice(None)), :] = capex_mean.values + num_std*capex_std.values
        plot_df_low.loc[(slice(None)), :] = capex_mean.values - num_std*capex_std.values

        for technology in ep.getTechnologiesString():
            # Plot the dataframe
            plot_df_mean.loc[(slice(None)), technology].plot(kind='line', ax=ax, legend=False, label=technology, color=getTechnologyColor(technology))
            plot_df_high.loc[(slice(None)), technology].plot(kind='line', ax=ax, legend=False, label='_Hidden', linewidth=line_width_low, color=getTechnologyColor(technology))
            plot_df_low.loc[(slice(None)), technology].plot(kind='line', ax=ax, legend=False, label='_Hidden', linewidth=line_width_low, color=getTechnologyColor(technology))
            try:
                ax.fill_between(plot_df_mean.index, plot_df_high.loc[(slice(None)), technology], plot_df_low.loc[(slice(None)), technology], alpha=0.1, color=getTechnologyColor(technology))
            except:
                continue

        # Set the title
        ax.set_title(application[i], size=10, loc='left')
        # Set the x-labels
        ax.set_xlabel(' ', labelpad=30)
        # Set x-axis tick label intervals
        #ax.set_xticks([2020, 2025, 2030, 2035])
        ax.set_xticks(plot_df_mean.index)
        ax.tick_params(axis='x', rotation=rotation)
        # Set the y-labels
        if ep.getApplicationSegment(application[i], 'range') == 'Urban' :
            ax.set_ylabel('k USD', labelpad=15)

        # Set the legend
        if application[i] == 'HDV-Regional':
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.6), ncol = 6)

    # Set figure title
    fig.suptitle(region + ' : CAPEX Dynamic Lines (' + str(num_std) + ' std error)')

    # Set figure size
    fig.subplots_adjust(left=0.065, bottom=0.14, right=0.94, top=0.88, wspace=0.145, hspace=0.2)
    fig.set_size_inches(14,10)

    # Save the figure
    filename = ep.getOutputPlotsDirectory() + '\_' + region + '_capex_dynamic_lines.png'
    plt.savefig(filename, dpi=dpi)

################################################################################
"""
Use Case Parameters Plots:
Plot types
    1): Weight
    2): Range
    3): Power
    4): Energy
"""

def useCaseParametersPlots(year, parameter):

    parameter_unit = ''
    use_case_investors = ep.getNumInvestors()

    # Retrieve the appropriate pickle file
    if parameter == 'Weight':
        DF = pd.read_pickle(ep.getUseCaseCharacterizationOutputDirectory() + '\_WEIGHT_' + str(use_case_investors) + '_INVESTORS_DF.pkl').loc[(year, slice(None), slice(None), slice(None)), :]
        parameter_unit = '[kg]'
    elif parameter == 'Range':
        DF = pd.read_pickle(ep.getUseCaseCharacterizationOutputDirectory() + '\_RANGE_' + str(use_case_investors) + '_INVESTORS_DF.pkl').loc[(year, slice(None), slice(None), slice(None)), :]
        parameter_unit = '[km/day]'
    elif parameter == 'Power':
        DF = pd.read_pickle(ep.getUseCaseCharacterizationOutputDirectory() + '\_POWER_' + str(use_case_investors) + '_INVESTORS_DF.pkl').loc[(year, slice(None), slice(None), slice(None)), :]
        parameter_unit = '[kW]'
    elif parameter == 'Energy':
        DF = pd.read_pickle(ep.getUseCaseCharacterizationOutputDirectory() + '\_ENERGY_' + str(use_case_investors) + '_INVESTORS_DF.pkl').loc[(year, slice(None), slice(None), slice(None)), :]
        parameter_unit = '[kWh]'

    # Loop through each region and plot
    for region in ep.getRegionsString():

        # Setup the figure
        nrow=3
        ncol=3

        x_pad = 10
        y_pad = 15

        fig, axes = plt.subplots(nrows=nrow, ncols=ncol, sharey=True, sharex=False)

        # Initialize the years, applications and technologies
        years = np.transpose(ep.getYears().to_numpy())[0]
        application_list = ep.getApplicationsString()
        technology_list = ep.getTechnologiesString()

        # Set the colors
        ICE_D = "tab:orange"
        BEV = "tab:green"
        HET = "tab:blue"
        FCEV = "tab:red"
        ICE_NG = "tab:purple"
        ICE_BD = 'tab:brown'
        OHC = 'tab:cyan'
        colors = [ICE_BD, ICE_NG, FCEV, HET, BEV, ICE_D].reverse()


        for i, ax_i in enumerate(axes.flatten()):

            # Get the array of investor parameters for plotting
            plot_array = DF.loc[(year, region, application_list[i], 'BEV'), :]
            plot_array.hist(color='k', alpha=0.5, ax=ax_i, bins=30)

            # # Make the plot
            # ax.stackplot(years, np.array(tech_array_1, dtype=float), np.array(tech_array_2, dtype=float), np.array(tech_array_3, dtype=float), np.array(tech_array_4, dtype=float), np.array(tech_array_5, dtype=float), np.array(tech_array_6, dtype=float), np.array(tech_array_7, dtype=float), colors=colors)
            ax_i.set_title(application_list[i])
            # Only set the y-label on the left sub-plots
            if (i % 3) == 0:
                ax_i.set_ylabel('Frequency', labelpad=y_pad)
            # Set the x-axis label on the bottom row
            last_i = nrow*ncol
            if (i==last_i-1 or i==last_i-2 or i==last_i-3):
                ax_i.set_xlabel(parameter + ' ' + parameter_unit, labelpad=x_pad)
            ax_i.margins(0, 0) # Set margins to avoid "whitespace"

            # Set x-axis tick label intervals
            #ax.set_xticks([2020, 2025, 2030, 2035])#np.arange(years[0], years[-1], 4))

        # Set Title
        fig.suptitle(region + ' : ' + str(year) + ' : ' + parameter.upper() + ' Histogram Plots')
        # fig.subplots_adjust(hspace=0.2, wspace=0.065, bottom=0.1)
        #handles, labels = ax_i.get_legend_handles_labels()
        #axis.legend(reversed(handles), reversed(labels), loc='lower center', ncol = len(technology_list))
        #fig.legend(technology_list, loc='lower center', ncol = len(technology_list))

        fig.subplots_adjust(left=0.1, bottom=0.095, right=0.968, top=0.91, wspace=0.2, hspace=0.26)
        fig.set_size_inches(12,10)
        # Save the figure
        filename = ep.getOutputPlotsDirectory() + '\_' + parameter.lower() + '_' + region + '_' + str(use_case_investors) + '_investors.png'
        plt.savefig(filename, dpi=dpi)

################################################################################
"""
Experience Component Capacity Additions Plots:
Plot types
    1): Total annual capacity additions for each experience component
    2): Endogenous annual additions for each experience component, stacked by application segment

"""

def experienceComponentTotalCapacityAdditionPlot(component):

    # Get the exogenous scenarios
    exog_mid = ep.getExogenousMarketCapacityAdditions(component, 'Mid', 'all')
    exog_high = ep.getExogenousMarketCapacityAdditions(component, 'High', 'all')
    exog_low = ep.getExogenousMarketCapacityAdditions(component, 'Low', 'all')

    # Get the exogenous market capacity additions
    EXOG_CAP_ADD_DF = pd.read_pickle(ep.getFinalOutputDirectory() + '\_EXOG_CAP_ADD_DF.pkl').loc[:,component]
    EXOG_CAP_ADD_DF = EXOG_CAP_ADD_DF.groupby(level='Year', axis=0).sum()
    # Get the endogenous market capacity additions and then sum them
    ENDOG_CAP_ADD_DF = pd.read_pickle(ep.getFinalOutputDirectory() + '\_ENDOG_CAP_ADD_DF.pkl').loc[(slice(None), slice(None), slice(None)), component]
    ENDOG_CAP_ADD_DF = ENDOG_CAP_ADD_DF.groupby(level='YEAR', axis=0).sum()

    # Stack the two in a new dataframe
    df_plot = pd.DataFrame(columns={'EXOG', 'ENDOG', 'EXOG Scenario - High', 'EXOG Scenario - Mid',  'EXOG Scenario - Low'})
    df_plot = pd.DataFrame(columns={'EXOG', 'ENDOG'})
    df_plot['EXOG Scenario - Mid'] = exog_mid
    df_plot['EXOG Scenario - High'] = exog_high
    df_plot['EXOG Scenario - Low'] = exog_low
    df_plot['EXOG'] = EXOG_CAP_ADD_DF
    df_plot['ENDOG'] = ENDOG_CAP_ADD_DF
    df_plot['EXOG'] = df_plot['EXOG'].replace(np.nan,0)
    df_plot['ENDOG'] = df_plot['ENDOG'].replace(np.nan,0)
    df_plot = df_plot.reset_index()
    df_plot['index'] = df_plot['index'].astype(str)

    # Set the colors
    bar_bottom = 'darkgrey'
    bar_top = 'darkorange'
    line_high = 'lightskyblue'
    line_mid = 'lightskyblue'
    line_low = 'lightskyblue'

    # Plot the bars
    ax = df_plot[['EXOG', 'ENDOG']].plot(kind='bar', stacked=True, color=[bar_bottom, bar_top], legend=True)
    # Plot the lines
    df_plot[['EXOG Scenario - Mid']].plot(kind='line', ax=ax, color=line_mid, linestyle='-',legend=True)
    # NOTE: if high or low exogenous capacity addition scenario data is included in the model, then the below two plot lines can be used
    #df_plot[['EXOG Scenario - High']].plot(kind='line', ax=ax, color=line_high, linestyle='--', legend=True)
    #df_plot[['EXOG Scenario - Low']].plot(kind='line', ax=ax, color=line_low, linestyle='-.',legend=True)

    # Set the axis, labels, title, etc.
    ax.set_xticklabels(df_plot['index'], rotation=40)
    plt.title(component + ' : Annual Capacity Additions')
    plt.ylabel('Installed Capacity ' + getComponentUnit(component, 'Capacity'))
    plt.xlabel(' ')

    # Set the legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='center left', bbox_to_anchor=(1.0, 0.5))

    # Set the plot size
    plt.subplots_adjust(left=0.13, bottom=0.110, right=0.665, top=0.880, wspace=0.18, hspace=0.4)

    # Save the figure
    component_name = component.lower().replace('/', '_')
    filename = ep.getOutputPlotsDirectory() + '\_' + component_name + '_annual_capacity_additions.png'
    plt.savefig(filename, dpi=dpi)

def experienceComponentEndogenousCapacityAdditionsPlot(component):

    # Get the endogenous market capacity additions and then sum them
    ENDOG_CAP_ADD_DF = pd.read_pickle(ep.getFinalOutputDirectory() + '\_ENDOG_CAP_ADD_DF.pkl').loc[(slice(None), slice(None), slice(None)), component]
    ENDOG_CAP_ADD_DF = ENDOG_CAP_ADD_DF.groupby(['YEAR', 'APPLICATION'], axis=0).sum()
    # unstack the dataframe for plotting
    df_plot = ENDOG_CAP_ADD_DF.unstack(level='APPLICATION')

    # Plot the bars
    ax = df_plot.plot(kind='bar', stacked=True, legend=True)

    # Set the axis, labels, title, etc.
    ax.set_xticklabels(df_plot.index, rotation=40)
    plt.title(component + ' : Endogenous Capacity Additions')
    plt.ylabel('Installed Capacity ' + getComponentUnit(component, 'Capacity'))
    plt.xlabel(' ')

    # Set the legends
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='center left', bbox_to_anchor=(1.0, 0.5))

    # Set the figure size
    plt.subplots_adjust(left=0.13, bottom=0.110, right=0.665, top=0.880, wspace=0.18, hspace=0.4)

    # Save the figure
    component_name = component.lower().replace('/', '_')
    filename = ep.getOutputPlotsDirectory() + '\_' + component_name + '_endogenous_capacity_additions.png'
    plt.savefig(filename, dpi=dpi)

################################################################################
"""
Experience Curve Plots:
Plot types
    1): Experience curve component plot with actual cost progression

"""

def experienceCurvePlot(component, isolated_regions, non_isolation_regions):

    # Initialize the figure
    fig = plt.figure()
    ax = plt.axes()

    # Plot cost curves for each region
    for region in ep.getRegionsString():

        if region != 'US' and region != 'Rest of World':
            # Set the applicaiton how you would like
            application = 'HDV'

            # Get experience curve parameters and scenarios
            base_year_cap = ep.getExperienceComponentBaseCapacity(component)
            base_year_cost = ep.getExperienceComponentData(component, application, 'Mid Cost')
            b_mid = ep.getExperienceComponentData(component, application, 'Learning Parameter Mid')
            b_high = ep.getExperienceComponentData(component, application, 'Learning Parameter High')
            b_low = ep.getExperienceComponentData(component, application, 'Learning Parameter Low')

            # Get the exogenous market capacity additions
            if region in isolated_regions:
                # If the region is an isolated jurisdiction...
                EXOG_CAP_ADD_DF = pd.read_pickle(ep.getFinalOutputDirectory() + '\_EXOG_CAP_ADD_DF.pkl').loc[(slice(None), region),component]
                # Get the endogenous market capacity additions and then sum them
                ENDOG_CAP_ADD_DF = pd.read_pickle(ep.getFinalOutputDirectory() + '\_ENDOG_CAP_ADD_DF.pkl').loc[(slice(None), region, slice(None)), component]
                ENDOG_CAP_ADD_DF = ENDOG_CAP_ADD_DF.groupby(level=['YEAR', 'REGION'], axis=0).sum()

            else:
                # If the region is a non-isolated jurisdiction...
                EXOG_CAP_ADD_DF = pd.read_pickle(ep.getFinalOutputDirectory() + '\_EXOG_CAP_ADD_DF.pkl').loc[(slice(None), non_isolation_regions),component]
                EXOG_CAP_ADD_DF = EXOG_CAP_ADD_DF.groupby(level=['Year'], axis=0).sum()
                # Get the endogenous market capacity additions and then sum them
                ENDOG_CAP_ADD_DF = pd.read_pickle(ep.getFinalOutputDirectory() + '\_ENDOG_CAP_ADD_DF.pkl').loc[(slice(None), non_isolation_regions, slice(None)), component]
                ENDOG_CAP_ADD_DF = ENDOG_CAP_ADD_DF.groupby(level=['YEAR'], axis=0).sum()

            # Sum the EXOG + ENDOG
            TOTAL_CAP_ADD_DF = EXOG_CAP_ADD_DF + ENDOG_CAP_ADD_DF

            # Compute the cumulative capacity addition
            TOTAL_CUM_CAP_DF = pd.DataFrame(columns=['YEAR', 'CUM CAP'], dtype='object')
            for year in ep.getYears().values:
                if(year == BASE_YEAR):
                    cum_cap = base_year_cap
                    dict = {'YEAR':[year[0]],
                             'CUM CAP':[cum_cap]}
                    df_dict = pd.DataFrame(dict, dtype='object')
                    TOTAL_CUM_CAP_DF = pd.concat([TOTAL_CUM_CAP_DF, df_dict], ignore_index=True)
                else:
                    prev_cap = TOTAL_CUM_CAP_DF.copy().set_index('YEAR').loc[(year[0]-1), 'CUM CAP']
                    cum_cap = prev_cap + TOTAL_CAP_ADD_DF[year-1].values[0]
                    dict = {'YEAR':[year[0]],
                             'CUM CAP':[cum_cap]}
                    df_dict = pd.DataFrame(dict, dtype='object')
                    TOTAL_CUM_CAP_DF = pd.concat([TOTAL_CUM_CAP_DF, df_dict], ignore_index=True)

            # Get the final capactiy for plot x-axis limit reference
            final_cap = TOTAL_CUM_CAP_DF['CUM CAP'].iloc[-1]

            # Get the region specific top up/down
            if 'Integration Factor' in component:
                top_up_down = 1
            else:
                top_up_down = 1+ep.getTopUpDownData(component, region, BASE_YEAR)

            # Get the yearly cost progression
            x_points_cum_cap = TOTAL_CUM_CAP_DF['CUM CAP'].iloc[:].values
            y_points_cost = pd.read_pickle(ep.getFinalOutputDirectory() + '\_DYNM_COST_DF.pkl').loc[(region, component, application), :].values*top_up_down

            # Experience curve arrays
            n=1000
            step = (final_cap-base_year_cap)/n
            x_cum_cap = np.arange(base_year_cap,final_cap,step)
            y_cost_curve_mid = (base_year_cost*(x_cum_cap/base_year_cap)**(-b_mid))*top_up_down
            y_cost_curve_high = (base_year_cost*(x_cum_cap/base_year_cap)**(-b_high))*top_up_down
            y_cost_curve_low = (base_year_cost*(x_cum_cap/base_year_cap)**(-b_low))*top_up_down

            # Plot experience curves
            ax.plot(x_cum_cap, y_cost_curve_mid, color=getRegionColor(region), linestyle='-', label=(region+' ER: '+ str(round(((1-2**(-b_mid))*100))) + '%'))
            ax.plot(x_cum_cap, y_cost_curve_high, color=getRegionColor(region), linestyle='--', label=(region+' ER: '+ str(round(((1-2**(-b_high))*100))) + '%'))
            ax.plot(x_cum_cap, y_cost_curve_low, color=getRegionColor(region), linestyle='-.', label=(region+' ER: '+ str(round(((1-2**(-b_low))*100))) + '%'))
            # Fill between the experience curves
            ax.fill_between(x_cum_cap, y_cost_curve_high, y_cost_curve_low, alpha=0.1, color=getRegionColor(region))

            # Plot the actual cost progression points
            ax.scatter(x_points_cum_cap, y_points_cost, color=getRegionColor(region))
            if region == 'EU':
                x_offset = 0
                y_offset = 0
                for i, txt in enumerate(ep.getYears().values):
                    ax.annotate(txt[0], (x_points_cum_cap[i]+x_offset, y_points_cost[i]+y_offset), color=getRegionColor(region))

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1.0, 0.5))

        # Insert horizontal figure lines
        ax.yaxis.grid(color='gray', linestyle='dashed')

        x_pad = 15
        plt.xlabel('Total Cummulative Installed Capacity [GWh]', labelpad=x_pad)
        if 'Integration Factor' in component:
            plt.ylabel('Integration Factor [%]', labelpad=x_pad)
        else:
            plt.ylabel('Component Cost ' + getComponentUnit(component, 'Price'), labelpad=x_pad)

        # Set the figure title
        fig.suptitle('Experience Curve : ' + component + ' (Exogenous Market Scenario: '+ ep.getExperienceComponentScenario(component, 'Exogenous Market Scenario') + ')')

        # Set the figure size
        fig.subplots_adjust(left=0.1, bottom=0.095, right=0.838, top=0.91, wspace=0.2, hspace=0.26)
        fig.set_size_inches(12,10)

        # Save the figure
        component_name = component.lower().replace('/', '_')
        component_name = component_name.replace(' ', '_')
        filename = ep.getOutputPlotsDirectory() + '\_' + component_name + '_experience_curve_plot.png'
        plt.savefig(filename, dpi=dpi)

################################################################################
"""
Global Vehicle Sales Plots:
Plot types
    1): Vehicle sales projections subplotted into three weight segments (LDV, MDV, HDV) and differentiated by region
"""

def globalVehicleSalesPlot():

    region_colors = np.array([])

    vehicle_scale_factor = 1000000

    # Organize the vehicle sales data for the LDV segment
    ldv_applications = ['LDV-Urban', 'LDV-Regional', 'LDV-LongHaul']
    mdv_applications = ['MDV-Urban', 'MDV-Regional', 'MDV-LongHaul']
    hdv_applications = ['HDV-Urban', 'HDV-Regional', 'HDV-LongHaul']
    ldv_vehicle_sales = pd.DataFrame()
    mdv_vehicle_sales = pd.DataFrame()
    hdv_vehicle_sales = pd.DataFrame()
    # Get the data for each region
    for region in ep.getRegionsString():
        # LDV
        vehicle_sales_ldv = ep.getVehicleSalesForecast(slice(None), region, ldv_applications).sum(axis=0)
        ldv_vehicle_sales[region] = (vehicle_sales_ldv/vehicle_scale_factor).values
        # MDV
        vehicle_sales_mdv = ep.getVehicleSalesForecast(slice(None), region, mdv_applications).sum(axis=0)
        mdv_vehicle_sales[region] = (vehicle_sales_mdv/vehicle_scale_factor).values
        # HDV
        vehicle_sales_hdv = ep.getVehicleSalesForecast(slice(None), region, hdv_applications).sum(axis=0)
        hdv_vehicle_sales[region] = (vehicle_sales_hdv/vehicle_scale_factor).values

        # Collect the region colors
        region_colors = np.append(region_colors, getRegionColor(region))


    # Plot the 3 segments as subplots
    # Define number of rows and columns for subplot
    nrow=1
    ncol=3
    fig, axes = plt.subplots(nrows=nrow, ncols=ncol, sharey=False, sharex=True)
    ax1 = axes[0]
    ax2 = axes[1]
    ax3 = axes[2]

    # Plot the LDV segment
    ldv_vehicle_sales.plot(kind='bar', stacked=True, ax=ax1, color=region_colors, alpha=0.75, legend=False)
    years = np.arange(2020,2035+1,1)
    ax1.set_xticklabels(years, rotation=40)
    x_pad = 15
    ax1.set_ylabel('Million Vehicles', labelpad=x_pad)
    ax1.set_title('Light-Duty Vehicles', loc='left')
    # Plot the MDV segment
    mdv_vehicle_sales.plot(kind='bar', stacked=True, ax=ax2, color=region_colors, alpha=0.75, legend=False)
    ax2.set_xticklabels(years, rotation=40)
    ax2.set_ylabel('Million Vehicles', labelpad=x_pad)
    ax2.set_title('Medium-Duty Vehicles', loc='left')
    # Plot the HDV segment
    hdv_vehicle_sales.plot(kind='bar', stacked=True, ax=ax3, color=region_colors, alpha=0.75, legend=True)
    ax3.set_xticklabels(years, rotation=40)
    ax3.set_ylabel('Million Vehicles', labelpad=x_pad)
    ax3.set_title('Heavy-Duty Vehicles', loc='left')
    handles, labels = ax3.get_legend_handles_labels()
    ax3.legend(handles, labels, loc='center left', bbox_to_anchor=(1.0, 0.5))

    # Set the figure subtitle
    fig.suptitle('Projected Global Vehicle Sales by Segment')

    # Set the figure size
    fig.subplots_adjust(left=0.06, bottom=0.095, right=0.9, top=0.91, wspace=0.2, hspace=0.26)
    fig.set_size_inches(18,10)

    # Save the figure
    filename = ep.getOutputPlotsDirectory() + '\_projected_global_vehicle_sales.png'
    plt.savefig(filename, dpi=dpi)

################################################################################
"""
Fuel Cost Plots:
Plot types
    1): Dynamic fuel costs plot for a specific fuel type
"""

def fuelCostPlot(fuel_type):

    fuel_unit = ''
    # Set the fuel type unit
    if fuel_type == 'Diesel' or fuel_type == 'Bio Diesel':
        fuel_unit = 'USD/L'
    elif fuel_type == 'Natural Gas (LNG)' or fuel_type == 'Natural Gas (CNG)':
        fuel_unit = 'USD/kg-NG'
    elif fuel_type == 'Electricity':
        fuel_unit = 'USD/kWh'
    elif fuel_type == 'Hydrogen (renewable)':
        fuel_unit = 'USD/kg-H2'

    std_factor = 0.5

    # Colors
    ref_color = "deeppink"
    high_color = "dodgerblue"
    low_color = "g"
    proj_color = "orange"

    # Get the fuel cost arrays for all regions
    # MIDDLE Cost Arrays
    ref_df = pd.DataFrame()
    region_colors = np.array([])
    for region in ep.getRegionsString():
        if fuel_type == 'Hydrogen (renewable)':
            stats_param = 'Most likely'
        else:
            stats_param = 'Mean'
        ref_df[region] = ep.getFuelCosts(region, fuel_type, stats_param, 2020, 2045)
        # Collect the region colors
        region_colors = np.append(region_colors, getRegionColor(region))

    # UP Cost Arrays
    ref_df_up = ref_df.copy()
    for region in ep.getRegionsString():
        if fuel_type == 'Hydrogen (renewable)':
            stats_param = 'High'
        else:
            stats_param = 'Std'

        ref_df_up[region] = ref_df_up[region] + std_factor*ep.getFuelCosts(region, fuel_type, stats_param, 2020, 2045)

    # LOW Cost Arrays
    ref_df_down = ref_df.copy()
    for region in ep.getRegionsString():
        if fuel_type == 'Hydrogen (renewable)':
            stats_param = 'Low'
        else:
            stats_param = 'Std'

        ref_df_down[region] = ref_df_down[region] - std_factor*ep.getFuelCosts(region, fuel_type, stats_param, 2020, 2045)

    # Setup the figure and plot
    nrow=1
    ncol=1
    fig, ax = plt.subplots(nrows=nrow, ncols=ncol)

    for region in ep.getRegionsString():

        standard_line_width = 1
        bigger_line_width = 2
        line_width = bigger_line_width

        ref_label = region + ' Ref'

        # Plot the mean fuel cost line
        ref_df[region].plot(kind='line', ax=ax, color=getRegionColor(region), linewidth=line_width, linestyle='-', label=ref_label)

        # Fill between the standard deviation of fuel cost projections
        ax.fill_between(ref_df.index, ref_df_up[region], ref_df_down[region], alpha=0.1, color=getRegionColor(region))

    # Set the ylabel
    ax.set_ylabel(fuel_unit, labelpad=15)

    # Set the legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1.0, 0.5))

    # Set the figure subtitle
    fig.suptitle('Fuel Projections : ' + fuel_type)

    # Set the figure size
    fig.subplots_adjust(left=0.05, bottom=0.095, right=0.85, top=0.91, wspace=0.25, hspace=0.26)
    fig.set_size_inches(16,10)

    # Put gridlines on for the hydrogen fuel plot
    if fuel_type == 'Hydrogen (renewable)':
        ax.grid(axis='y')

    # Save the figure
    filename = ep.getOutputPlotsDirectory() + '\_' + fuel_type + '_projected_fuel_costs.png'
    plt.savefig(filename, dpi=dpi)

################################################################################
"""
Switching Cost Plots:
Plot types
    1): Switching cost plot (function works for both the switching cost multiplier and the absolute switching cost)
"""

def switchingCostPlot(df, region, filename, plot_title, y_unit_label):
    # Set the figure
    nrow=3
    ncol=3

    fig, axes = plt.subplots(nrows=nrow, ncols=ncol, sharex=True)

    # Initialize the years, applications and technologies
    years = np.transpose(ep.getYears().to_numpy())[0]
    application = ep.getApplicationsString()
    tech_order = ep.getTechnologies().loc[:,'Technology']

    # Initialize the tick label parameters rotation
    rotation = 40

    for i, ax in enumerate(axes.flatten()):

        plot_df = df.loc[(slice(None), region, application[i]), :]

        for technology in ep.getTechnologiesString():
            values = plot_df[technology].values
            ax.plot(years, values, label=technology, color=getTechnologyColor(technology))

        # Set the title
        ax.set_title(application[i], size=10, loc='center')
        # Set the x-labels
        ax.set_xlabel(' ', labelpad=30)
        # Set x-axis tick label intervals
        ax.set_xticks(years)
        ax.tick_params(axis='x', rotation=rotation)
        # Set the y-axis units
        ax.annotate(y_unit_label, xy=(0,1), xytext=(20, ticklabelpad+12), ha='right', va='top',xycoords='axes fraction', textcoords='offset points')

        # Set the legend
        if application[i] == 'HDV-Regional':
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.6), ncol = 6)

    # Set the figure title
    fig.suptitle(region + ' : ' + plot_title)

    # Set the figure size
    fig.subplots_adjust(left=0.065, bottom=0.14, right=0.94, top=0.88, wspace=0.24, hspace=0.24)
    fig.set_size_inches(14,10)

    # Save the figure
    plt.savefig(filename, dpi=dpi)

################################################################################
"""
Vehicle Drive Cycle Plots:
Plot types
    1): Drive Cycle Plot
"""

def driveCyclePlot(drive_cycle_names):

    num_drive_cycles = len(drive_cycle_names)

    for i in np.arange(num_drive_cycles):

        nrow=1
        ncol=1

        fig, ax = plt.subplots(nrows=nrow, ncols=ncol, sharey=False, sharex=False)

        velocity = ep.getDriveCycleData(i+1, 'Velocity').values
        time = ep.getDriveCycleData(i+1, 'Time').values

        ax.plot(time, velocity)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Velocity (km/h)')
        ax.set_title(drive_cycle_names[i])
        ax.set_ylim([0,140])

        filename = drive_cycle_names[i]
        plt.savefig(ep.getOutputPlotsDirectory() + '\_' + filename, dpi=dpi)

################################################################################
################################################################################
# PLOT CALLS:
################################################################################
################################################################################

def runPlots():
    """
    Run specific plots of your choosing by calling (turning on/off) their respective functions:
    """
    ####################################################
    # Set whether or not you want to run each plot (True means the plot is run; False means the plot is not run).
    market_share_plots_BOOL = True
    market_share_weight_segments_plots_BOOL = True
    market_share_plots_single_BOOL = True
    tco_plots_BOOL = True
    capex_plots_BOOL = True
    use_case_parameters_plots_BOOL = True
    capacity_additions_plots_BOOL = True
    experience_curve_plots_BOOL = False
    global_vehicle_sales_plots_BOOL = True
    fuel_cost_plots_BOOL = True
    switching_cost_multiplier_plots_BOOL = True
    switching_cost_plots_BOOL = True
    drive_cycle_plots_BOOL = True
    ####################################################

    global MARKET_SHARES_DF
    global MAIN_OUTPUT_DF
    global TCO_DF_MEAN
    global TCO_DF_STD
    global TCO_PARAMETERS_DF_MEAN
    global TCO_PARAMETERS_DF_STD
    global CAPEX_DF_MEAN
    global CAPEX_DF_STD
    global CAPEX_PARAMETERS_DF_MEAN
    global CAPEX_PARAMETERS_DF_STD
    global SWITCHING_COST_MULTIPLIER_DF
    global SWITCHING_COST_DF

    # Read in the pickle files from the main output
    MAIN_OUTPUT_DF = pd.read_pickle(ep.getFinalOutputDirectory() + '\_MAIN_OUTPUT_DF.pkl')
    MARKET_SHARES_DF = pd.read_pickle(ep.getFinalOutputDirectory() + '\_MARKET_SHARES_DF.pkl')
    TCO_DF_MEAN = pd.read_pickle(ep.getFinalOutputDirectory() + '\_TCO_DF_MEAN.pkl')
    TCO_DF_STD = pd.read_pickle(ep.getFinalOutputDirectory() + '\_TCO_DF_STD.pkl')
    TCO_PARAMETERS_DF_MEAN = pd.read_pickle(ep.getFinalOutputDirectory() + '\_TCO_PARAMETERS_DF_MEAN.pkl')
    TCO_PARAMETERS_DF_STD = pd.read_pickle(ep.getFinalOutputDirectory() + '\_TCO_PARAMETERS_DF_STD.pkl')
    CAPEX_DF_MEAN = pd.read_pickle(ep.getFinalOutputDirectory() + '\_CAPEX_DF_MEAN.pkl')
    CAPEX_DF_STD = pd.read_pickle(ep.getFinalOutputDirectory() + '\_CAPEX_DF_STD.pkl')
    CAPEX_PARAMETERS_DF_MEAN = pd.read_pickle(ep.getFinalOutputDirectory() + '\_CAPEX_PARAMETERS_DF_MEAN.pkl')
    CAPEX_PARAMETERS_DF_STD = pd.read_pickle(ep.getFinalOutputDirectory() + '\_CAPEX_PARAMETERS_DF_STD.pkl')
    SWITCHING_COST_MULTIPLIER_DF = pd.read_pickle(ep.getFinalOutputDirectory() + '\_SWITCHING_COST_MULTIPLIER_DF.pkl')
    SWITCHING_COST_DF = pd.read_pickle(ep.getFinalOutputDirectory() + '\_SWITCHING_COST_DF.pkl')

    if market_share_plots_BOOL:
        # Plot the market share plots
        print('---------------------------------------')
        print('')
        print('Plotting the market share plots ...')
        print('')
        print('---------------------------------------')

        ###################################################################
        # SET THE LOWESS SMOOTHING BOOL
        lowess_smoothing_BOOL = True
        ###################################################################

        if lowess_smoothing_BOOL:
            ###################################################################
            # The lowess smoothing method
            years = ep.getYears()['Years'].values
            for region in ep.getRegionsString():
                for application in ep.getApplicationsString():
                    for technology in ep.getTechnologiesString():
                        Y = MARKET_SHARES_DF.loc[(slice(None), region, application),technology].values
                        Z = MAIN_OUTPUT_DF.loc[(slice(None), region, application),technology].values
                        X = years
                        smoothing_factor = 0.25
                        y_lowess = sm.nonparametric.lowess(Y, X, frac = smoothing_factor)  # 30 % lowess smoothing (must be between [0,1]; higher frac means higher smoothing)
                        z_lowess = sm.nonparametric.lowess(Z, X, frac = smoothing_factor)  # 30 % lowess smoothing (must be between [0,1]; higher frac means higher smoothing)
                        MARKET_SHARES_DF.loc[(slice(None), region, application),technology] = y_lowess[:,1]
                        MAIN_OUTPUT_DF.loc[(slice(None), region, application),technology] = z_lowess[:,1]
            ###################################################################

        for region in ep.getRegionsString():
            # Make a market share plot for each application segment in each geogrpahy
            stackedAreaChartPlots(region, MARKET_SHARES_DF, 'None', ep.getOutputPlotsDirectory())

        # Then plot all regions together as a 'Global' region
        global_market_shares_df = MAIN_OUTPUT_DF.groupby(['YEAR', 'APPLICATION']).sum()
        global_market_shares_df = global_market_shares_df.div(global_market_shares_df.sum(axis=1), axis=0)
        stackedAreaChartPlots('Global', global_market_shares_df, 'None', ep.getOutputPlotsDirectory())

    if market_share_weight_segments_plots_BOOL:
        # First plot the BASE market share plots so we can compare
        print('---------------------------------------')
        print('')
        print('Plotting the market share WEIGHT SEGMENTS plots ...')
        print('')
        print('---------------------------------------')

        ###################################################################
        # SET THE LOWESS SMOOTHING BOOL
        lowess_smoothing_BOOL = True
        ###################################################################

        if lowess_smoothing_BOOL:
            ###################################################################
            # The lowess smoothing method
            years = ep.getYears()['Years'].values
            for region in ep.getRegionsString():
                for application in ep.getApplicationsString():
                    for technology in ep.getTechnologiesString():
                        Y = MAIN_OUTPUT_DF.loc[(slice(None), region, application),technology].values
                        X = years
                        y_lowess = sm.nonparametric.lowess(Y, X, frac = 0.4)  # 30 % lowess smoothing (must be between [0,1]; higher frac means higher smoothing)
                        MAIN_OUTPUT_DF.loc[(slice(None), region, application),technology] = y_lowess[:,1]
            ###################################################################

        ###################################################################
        # Reorganize the dataframe for the global plot
        column_list = np.append(np.array(['YEAR', 'REGION', 'APPLICATION']), MAIN_OUTPUT_DF.columns.to_numpy())
        main_output_global = pd.DataFrame(columns=column_list).set_index(['YEAR', 'REGION', 'APPLICATION'])
        years = ep.getYears()['Years'].values
        grouped_main_output = MAIN_OUTPUT_DF.groupby(['YEAR', 'APPLICATION']).sum()
        for year in years:
            for application in ep.getApplicationsString():
                main_output_global.loc[(year, 'Global', application), :] = grouped_main_output.loc[(year, application), :]

        # Now append the global frame to the main frame
        main_output_df = pd.concat([MAIN_OUTPUT_DF.copy(), main_output_global])
        ###################################################################

        ###################################################################
        # SET THE PLOT TYPES
        all_regions_BOOL = True
        each_region_BOOL = False
        select_regions_BOOL = False

        ###################################################################
        if all_regions_BOOL:
            # PLOT ALL REGIONS TOGETHER (not including 'Global'): Make a market share plot for each application segment in each region
            filename = 'All'
            region = 'All'
            stackedAreaChartSegmentPlots(region, MAIN_OUTPUT_DF, filename, ep.getOutputPlotsDirectory())

        if each_region_BOOL:
        # PLOT EACH REGION INDIVIDUALLY: Make a market share plot for each application segment in each region
            for region in np.append(ep.getRegionsString(), 'Global'):
                filename = region
                # Make a market share plot for each application segment in each geogrpahy
                stackedAreaChartSegmentPlots([region], main_output_df, filename, ep.getOutputPlotsDirectory())

        if select_regions_BOOL:
            # PLOT SELECT REGIONS TOGETHER: Make a market share plot for each application segment in each selected region
            # (NOTE: you can change/select individually the included regions as well as the order of the regions)
            filename = 'Select_Regions'
            regions = ['Global', 'China', 'EU', 'US', 'India', 'Brazil', 'Rest of World']
            stackedAreaChartSegmentPlots(regions, main_output_df, filename, ep.getOutputPlotsDirectory())

    if market_share_plots_single_BOOL:
        # Plot the market share plots
        print('---------------------------------------')
        print('')
        print('Plotting the market share plots SINGLE...')
        print('')
        print('---------------------------------------')

        ###################################################################
        # SET THE LOWESS SMOOTHING BOOL
        lowess_smoothing_BOOL = True
        ###################################################################

        if lowess_smoothing_BOOL:
            ###################################################################
            # The lowess smoothing method
            years = ep.getYears()['Years'].values
            for region in ep.getRegionsString():
                for application in ep.getApplicationsString():
                    for technology in ep.getTechnologiesString():
                        Y = MAIN_OUTPUT_DF.loc[(slice(None), region, application),technology].values
                        X = years
                        y_lowess = sm.nonparametric.lowess(Y, X, frac = 0.25)  # 30 % lowess smoothing (must be between [0,1]; higher frac means higher smoothing)
                        MAIN_OUTPUT_DF.loc[(slice(None), region, application),technology] = y_lowess[:,1]
            ###################################################################

        # First, aggregate the application segments to create a single market share plot for a given region
        market_shares_df = MAIN_OUTPUT_DF.groupby(['YEAR', 'REGION']).sum()
        market_shares_df = market_shares_df.div(market_shares_df.sum(axis=1), axis=0)

        for region in ep.getRegionsString():
            # Make a market share plot for each application segment in each geogrpahy
            stackedAreaChartPlotsSingle(region, market_shares_df, 'single', ep.getOutputPlotsDirectory())

        # Then plot all regions together
        global_market_shares_df = MAIN_OUTPUT_DF.groupby(['YEAR']).sum()
        global_market_shares_df = global_market_shares_df.div(global_market_shares_df.sum(axis=1), axis=0)
        stackedAreaChartPlotsSingle('Global', global_market_shares_df, 'single', ep.getOutputPlotsDirectory())

    if tco_plots_BOOL:
        # Plot TCO stacked bar plots
        print('---------------------------------------')
        print('')
        print('Plotting the TCO stacked bar plots ...')
        print('')
        print('---------------------------------------')
        years = [2020, 2035]
        regions = ep.getRegionsString()
        for year in years:
            for region in regions:
                tcoStackedBarPlot(year, region)

        # # Plot TCO dynamic line plots
        # print('---------------------------------------')
        # print('')
        # print('Plotting the TCO dynamic line plots ...')
        # print('')
        # print('---------------------------------------')
        # regions = ep.getRegionsString()
        # for region in regions:
        #     tcoDynamicLinePlot(region)

    if capex_plots_BOOL:
        # # Plot CAPEX bar plots
        # print('---------------------------------------')
        # print('')
        # print('Plotting the CAPEX bar plots ...')
        # print('')
        # print('---------------------------------------')
        # # HERE: select the specific years you would like to plot for
        # years = [2020]
        # for year in years:
        #     for region in ep.getRegionsString():
        #         CAPEXPlot(year, region)

        # Plot CAPEX line plots
        print('---------------------------------------')
        print('')
        print('Plotting the CAPEX dynamic line plots ...')
        print('')
        print('---------------------------------------')
        for region in ep.getRegionsString():
            CAPEXDynamicLinePlot(region)

    if use_case_parameters_plots_BOOL:
        # Plot use case parameters plots (histograms)
        print('-------------------------------------------')
        print('')
        print('Plotting the use case parameters plots ...')
        print('')
        print('-------------------------------------------')
        parameters_list = ['Weight', 'Range', 'Power', 'Energy']
        for param in parameters_list:
            useCaseParametersPlots(BASE_YEAR, param)

    if capacity_additions_plots_BOOL:
        # Plot the total annual capacity additions
        print('-----------------------------------------')
        print('')
        print('Plotting the total annual capacity addition plots ...')
        print('')
        print('---------------------------------------')
        components_list = ['Li-ion Battery', 'Fuel Cell System', 'Diesel Tank', 'Natural Gas Tank', 'Hydrogen Tank', 'Electric Drive System', 'ICE Powertrain']
        for component in components_list:
            experienceComponentTotalCapacityAdditionPlot(component)

        # Plot the endogeouns annual capacity additions
        print('------------------------------------------------------------------')
        print('')
        print('Plotting the endogenous capacity addition (by segment) plots ...')
        print('')
        print('------------------------------------------------------------------')
        components_list = ['Li-ion Battery', 'Fuel Cell System', 'Diesel Tank', 'Natural Gas Tank', 'Hydrogen Tank', 'Electric Drive System', 'ICE Powertrain']
        for component in components_list:
            experienceComponentEndogenousCapacityAdditionsPlot(component)

    if experience_curve_plots_BOOL:
        # Plot the experience curves
        print('---------------------------------------')
        print('')
        print('Plotting the experience curve plots ...')
        print('')
        print('---------------------------------------')
        components_list = ['Li-ion Battery', 'Diesel Tank', 'Hydrogen Tank', 'Natural Gas Tank', 'Electric Drive System', 'Fuel Cell System', 'ICE Powertrain']
        isolated_regions = []
        non_isolation_regions = ['China','EU', 'US', 'India', 'Brazil', 'Rest of World']
        for component in components_list:
            experienceCurvePlot(component, isolated_regions, non_isolation_regions)

    if global_vehicle_sales_plots_BOOL:
        # Plot the global vehicle sales plot  (three weight segments separated)
        print('--------------------------------------------')
        print('')
        print('Plotting the global vehicle sales plots ...')
        print('')
        print('--------------------------------------------')

        globalVehicleSalesPlot()

    if fuel_cost_plots_BOOL:
        # Plot the fuel cost plots (for all regions and all fuels)
        print('---------------------------------------')
        print('')
        print('Plotting the fuel cost plots ...')
        print('')
        print('---------------------------------------')
        fuel_types = ['Diesel', 'Natural Gas (LNG)', 'Natural Gas (CNG)', 'Electricity', 'Hydrogen (renewable)']
        for fuel_type in fuel_types:
            fuelCostPlot(fuel_type)

    if switching_cost_multiplier_plots_BOOL:
        # Plot the switching cost multiplier plots (for all regions)
        print('---------------------------------------')
        print('')
        print('Plotting the switching cost multiplier plots ...')
        print('')
        print('---------------------------------------')
        regions = ep.getRegionsString()
        for region in regions:
            filename = ep.getOutputPlotsDirectory() + '\_' + region + '_switching_cost_multiplier.png'
            switchingCostPlot(SWITCHING_COST_MULTIPLIER_DF, region, filename, 'Switching Cost Multiplier', '[norm %]')

    if switching_cost_plots_BOOL:
        # Plot the switching cost plots (for all regions)
        print('---------------------------------------')
        print('')
        print('Plotting the switching cost plots ...')
        print('')
        print('---------------------------------------')
        regions = ep.getRegionsString()
        for region in regions:
            filename = ep.getOutputPlotsDirectory() + '\_' + region + '_switching_cost.png'
            switchingCostPlot(SWITCHING_COST_DF, region, filename, 'Switching Cost', '[USD]')

    if drive_cycle_plots_BOOL:
        # Plot the drive cycle plots
        print('---------------------------------------')
        print('')
        print('Plotting the drive cycle plots...')
        print('')
        print('---------------------------------------')

        drive_cycle_names = ['World Light Duty Test Cycle (WLTC) Class 2', 'World Light Duty Test Cycle (WLTC) Class 3a', 'World Harmonized Vehicle Cycle (WHVC)',
                                'World Harmonized Vehicle Cycle - India (WHVC-India)', 'China Heavy Duty Commercial Vehicle Test Cycle - Light Trucks (CHTC-LT)',
                                'China Heavy Duty Commercial Vehicle Test Cycle - Heavy Trucks (CHTC-HT)']

        driveCyclePlot(drive_cycle_names)

    ################################################################################

    print('---------------------------------------------------------------')
    print('PLOT RUN TIME:')
    print('--- ',np.around((time.time() - start_time), decimals=3), 'seconds ---')
    print('--- ',np.around((time.time() - start_time)/60, decimals=3), 'minutes ---')
    print('(For ', ep.getNumInvestors(), ' investor(s).)')
    print('---------------------------------------------------------------')

    ###############################################################################


################################################################################
# MAIN PLOTS RUN CODE:
# NOTE: calling the function runPlots() below will run with the currently stored data files in '_01_Final Output Files'
runPlots()
################################################################################
