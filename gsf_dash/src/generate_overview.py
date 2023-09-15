from datetime import datetime
import pandas as pd
import warnings
import math
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse, Rectangle
from . import generate_multipage_pdf

class PondsOverviewPlots:
    def __init__(self, select_date, scorecard_dataframe, processing_dataframe, epa_data_dict, active_dict, save_output=True):
        self.select_date = pd.to_datetime(select_date).normalize() # Normalize select_date to remove potential time data and prevent possible key errors when selecting date range from data
        self.scorecard_dataframe = scorecard_dataframe
        self.processing_dataframe = processing_dataframe
        self.active_dict = active_dict
        self.save_output = save_output
        self.epa_data_dict = epa_data_dict
        self.potential_harvests_dict = None # initialize potential_harvests_dict, will be populated by self.plot_scorecard()
        self.output_filenames = [] # initialize a list to collect output filenames
        [self.output_filenames.append(x) for x in (self.plot_scorecard(), self.plot_potential_harvests(), self.plot_epa())]

    # function to plot outlined legend boxes, using a dict of labels/data, and a separate dict of x_alignments (see examples of dict structure in calls to this function)
    def plot_legend(self, fig, ax, legend_data, x_align, y_spacing, y_align=0.9):
        y_align = y_align # start y-coord for plotting legend
        max_coords = [1,1,0,0] #  initialize for tracking the max coordinates of legend items for printing a box around all the legend items (x0, y0, x1, y1)
        data_rows_x_bound = [1,0]
        data_rows_y_bounds = [] 
        indicator_plots = []
        for row in legend_data:
            row_y_bound = [1,0]
            row_fill_color = 'white'
            if 'fill_color' in row:
                row_fill_color = row['fill_color']

            for idx, (label, properties) in enumerate(row['labels'].items()):
                underline = False
                text_weight = 'normal'
                text_color = 'black'

                if 'weight' in properties:
                    if properties['weight'] == 'underline': # check if 'underline' option since it's not an option in matplotlib and needs to be added after text with an annotation using text bounding box coords
                        underline = True
                    elif properties['weight'] == 'bold': 
                        text_weight = 'bold'

                if 'color' in properties:
                    text_color = properties['color']

                if 'indicator_fill' in properties: # plot indicators instead of normal text
                    # Get transform info to properly plot circles, using Ellipse. Otherwise they are deformed 
                    # ref: https://stackoverflow.com/questions/9230389/why-is-matplotlib-plotting-my-circles-as-ovals
                    ax_x0, ax_y0 = ax.transAxes.transform((0, 0)) # lower left in pixels
                    ax_x1, ax_y1 = ax.transAxes.transform((1, 1)) # upper right in pixes
                    ax_dx = ax_x1 - ax_x0
                    ax_dy = ax_y1 - ax_y0
                    ax_maxd = max(ax_dx, ax_dy)
                    circle_width = .07 * ax_maxd / ax_dx
                    circle_height = .07 * ax_maxd / ax_dy

                    # plot indicator
                    t = ax.add_patch(Ellipse((x_align[properties['align']][0], y_align), circle_width, circle_height, color=properties['indicator_fill'], fill=properties['indicator_fill'], clip_on=False))
                    ax.text(x_align[properties['align']][0], y_align, label, color='white', ha='center', va='center', fontweight='bold', fontsize='x-small', linespacing=0.7, clip_on=False)

                else:
                    t = ax.text(x_align[properties['align']][0], y_align, label, ha = x_align[properties['align']][1], va = 'center', weight=text_weight, color=text_color)

                # get bounding box of plotted items
                bb = t.get_window_extent(renderer=fig.canvas.get_renderer()).transformed(ax.transAxes.inverted())

                if underline:
                    ax.annotate('', xy=(bb.x0-0.01,bb.y0), xytext=(bb.x1+0.01,bb.y0), xycoords="axes fraction", arrowprops=dict(arrowstyle="-", color='k'))

                # update the max coord boundaries for the rows that are assigned a color
                if row_fill_color != 'white':  
                    #print(label, bb)
                    if 'excl_color' not in properties: # add option of excluding a row item from being colored
                        if bb.x0 < data_rows_x_bound[0]:
                            data_rows_x_bound[0] = bb.x0
                        if bb.x1 > data_rows_x_bound[1]:
                            data_rows_x_bound[1] = bb.x1
                        if bb.y0 < row_y_bound[0]:
                            row_y_bound[0] = bb.y0
                        if bb.y1 > row_y_bound[1]:
                            row_y_bound[1] = bb.y1

                # update max_coords for bounding box around legend and for x-bounds of rows that will be filled with color
                if bb.x0 < max_coords[0]:
                    max_coords[0] = bb.x0
                if bb.y0 < max_coords[1]:
                    max_coords[1] = bb.y0
                if bb.x1 > max_coords[2]:
                    max_coords[2] = bb.x1
                if bb.y1 > max_coords[3]:
                    max_coords[3] = bb.y1

            y_align -= y_spacing # decrease y coord for each line for spacing       

            if row_fill_color != 'white':
                data_rows_y_bounds.append((row_y_bound,row_fill_color))

        #print('rows x-bound:', data_rows_x_bound)    
        #print('row y-bounds:', data_rows_y_bounds)
        
        # plot colored rectangle for corresponding row
        row_box_padding = 0.01
        for row_y in data_rows_y_bounds:
            ax.add_patch(Rectangle((data_rows_x_bound[0]-row_box_padding,row_y[0][0]-row_box_padding),data_rows_x_bound[1]-data_rows_x_bound[0]+row_box_padding*2,row_y[0][1]-row_y[0][0]+row_box_padding*2,linewidth=0,edgecolor=None,facecolor=row_y[1], clip_on=False))
        
        # plot color filled box around legend text
        box_padding = 0.03
        box_xy = (max_coords[0]-box_padding, max_coords[1]-box_padding)
        box_width = max_coords[1]-max_coords[0]
        ax.add_patch(Rectangle((max_coords[0]-box_padding,max_coords[1]-box_padding),max_coords[2]-max_coords[0]+box_padding*2,max_coords[3]-max_coords[1]+box_padding*2,linewidth=1,edgecolor='black',facecolor='none',clip_on=False))
    
    def plot_scorecard(self, target_to_density=0.4, target_topoff_depth=13, harvest_density=0.5, plot_title='Pond Health Overview'): 
        print('Plotting ponds overview...', flush=True)
    
        select_date = self.select_date
        
        # Initialize variables for aggregations
        total_mass_all = 0 # total mass for entire farm (regardless of density)
        total_harvestable_mass = 0 # all available mass with afdw > target_to_density
        potential_total_harvest_mass = 0 # all harvestable mass with resulting afdw > target_to_density but only ponds with current afdw > harvest_density
        potential_total_harvest_gals = 0 # potential harvest_mass in terms of volume in gallons
        potential_harvests = {'columns': [], 'data': {}, 'aggregates': {}} # dict to store calculated harvest depths for ponds with density >= harvest_density
        num_active_ponds = 0 # number of active ponds (defined by the plot_ponds().check_active() function)
        num_active_ponds_sm = 0 # number of active 1.1 acre ponds
        num_active_ponds_lg = 0 # number of active 2.2 acre ponds
        
        any_noncurrent_flag = False # flag to indicate when noncurrent AFDW or Depth data is being used, to add a note on plot explaining the notation 
        
        # helper function to query data from previous days, if it is not available for current day
        def current_or_prev_query(df, col_name, num_prev_days, return_noncurrent_flag=False):
            '''
            col_name: the name of the df column to query
            num_prev_days: the maximum number of days prior to look for data
            return_noncurrent_flag: whether to return a string ("current" or "noncurrent") which indicates whether the value is from the current or a prior day
            '''
            for day_num in range(0,num_prev_days+1):
                data = df[col_name].shift(day_num).fillna(0).loc[select_date] # use fillna(0) to force missing data as 0
                if data > 0:
                    if return_noncurrent_flag and day_num == 0:
                        return data, 'current' # return second value as 'current' to indicate data is current 
                    elif return_noncurrent_flag and day_num > 0:
                        return data, 'noncurrent' # return second value as 'non-current' to indicate data is not current (not from the current report day)
                    else:
                        return data
            if return_noncurrent_flag:
                return 0, 'n/a'
            else:
                return 0 # return 0 if no data found (though 'data' variable will be 0 anyway due to fillna(0) method used) 

        # helper function to query single data points from each pond dataframe
#         def data_query(pond_data, pond_name, query_name):
#             PLACEHOLDER TO IMPLEMENT WHEN GENERATING PLOT OF CURRENT VERSUS PREV DAY CHANGE
#
#
        # helper function to convert density & depth to mass (in kilograms)
        def afdw_depth_to_mass(afdw, depth, pond_name):
            depth_to_liters = 35000 * 3.78541 # save conversion factor for depth (in inches) to liters
            # for the 6 and 8 columns, double the depth to liters conversion because they are twice the size
            if pond_name[-2:] == '06' or pond_name[-2:] == '08': 
                depth_to_liters *= 2
            # calculate and return total mass (kg)
            return (afdw * depth_to_liters * depth) / 1000
        
        def calculate_growth(pond_data_df, select_date, num_days, remove_outliers, data_count_threshold=2, weighted_stats_for_outliers=True):
            '''
            pond_data_df: pandas dataframe
                - dataframe for individual pond
            select_date: datetime.date
                - current date (required for growth relative to this date)
            remove_outliers: bool
                - whether to remove outliers from the data before calculating growth (which are then zeroed and forward-filled)
                - outliers are calculated using the 10 most-recent non-zero data points
                - when this is set then the current day is excluded from calculating mean and std. dev (since it's more likely to be an outlier itself)
            num_days: int
                - number of days to calculate growth for
            data_count_threshold: int 
                - absolute threshold for days that must have data within the num_days growth period, otherwise growth will be 'n/a'
            weighted_stats_for_outliers: bool
                - whether to use weighted statistics for finding outliers, applies 80% weight evenly to the most-recent 5 days, then 20% to the remainder
            outlier_stddev_thresh: int
                - the standard deviation threshold for determining outliers (when greater or less than the threshold * standard deviation)
                - using a threshold of 2 standard deviations as default for more aggressive outlier detection (statistical standard is usally 3 std. dev.)
            pond_test: str
                - FOR TESTING: string corresponds to pond_name to display df output between steps
            '''
            
            # the standard deviation threshold for determining outliers (when greater or less than the threshold * standard deviation)
            # using a threshold of 2.25 standard deviations as default for more aggressive outlier detection (statistical standard is usally 3 std. dev.)
            outlier_stddev_thresh = 2.25 
            
            pond_test=None # set to pond_name string for printing output at intermediate calculation steps
            
            # Select last 20 days of data for columns 'AFDW', 'Depth', and 'Split Innoculum' (higher num in case of missing data, etc)
            pond_growth_df = pond_data_df.loc[select_date - pd.Timedelta(days=20):select_date][['AFDW (filter)', 'Depth', 'Split Innoculum']]
            # Get calculated mass column from AFDW and Depth
            pond_growth_df['mass'] = afdw_depth_to_mass(pond_growth_df['AFDW (filter)'], pond_growth_df['Depth'], pond_name)
            
            def next_since_harvest_check(row):
                if row['mass'] != 0:
                    if row['tmp harvest prev 1'] == 'Y': # if the previous day was harvested ('H' in 'Split Innoculum' col)
                        return 'Y'
                    elif row['tmp data prev 1'] == 'N' and row['tmp harvest prev 2'] == 'Y': # if 
                        return 'Y'
                    elif row['tmp data prev 1'] == 'N' and row['tmp data prev 2'] == 'N' and row['tmp harvest prev 3'] == 'Y':
                        return 'Y'
                    else:
                        return ''
                else:
                    return ''

            # find if row is the next data since harvest (i.e., if there is a few days delay in data since it was harvested), to ensure negative change from harvest is always zeroed out for growth calcs
            pond_growth_df['tmp data prev 1'] = pond_growth_df['mass'].shift(1).apply(lambda val: 'Y' if val != 0 else 'N')
            pond_growth_df['tmp data prev 2'] = pond_growth_df['mass'].shift(2).apply(lambda val: 'Y' if val != 0 else 'N')
            pond_growth_df['tmp harvest prev 1'] = pond_growth_df['Split Innoculum'].shift(1).apply(lambda val: 'Y' if val == 'H' or val == 'S' else 'N')
            pond_growth_df['tmp harvest prev 2'] = pond_growth_df['Split Innoculum'].shift(2).apply(lambda val: 'Y' if val == 'H' or val == 'S' else 'N')
            pond_growth_df['tmp harvest prev 3'] = pond_growth_df['Split Innoculum'].shift(3).apply(lambda val: 'Y' if val == 'H' or val == 'S' else 'N')
            pond_growth_df['next data since harvest'] = pond_growth_df.apply(lambda row: next_since_harvest_check(row), axis=1) 
            del pond_growth_df['tmp data prev 1']
            del pond_growth_df['tmp data prev 2'] 
            del pond_growth_df['tmp harvest prev 1'] 
            del pond_growth_df['tmp harvest prev 2'] 
            del pond_growth_df['tmp harvest prev 3']

            if remove_outliers: # calculate and drop outliers
                # get a new df for outlier detection and drop the last/ most recent row, 
                # assuming it is highly likely an outlier (due to data entry error, etc), so exclude that value from calc of mean and std dev for detecting outliers
                mass_nonzero_ = pond_growth_df.drop([select_date], axis=0) 
                mass_nonzero_ = mass_nonzero_[mass_nonzero_['mass'] != 0]['mass'] # get a pandas series with only nonzero values for calculated mass
                # limit selection for outlier detection to the last 10 non-zero values
                mass_nonzero_ = mass_nonzero_.iloc[-10:]
                
                if pond_name == pond_test:
                    print('std dev: ', mass_nonzero_.std())
                    print('mean: ', mass_nonzero_.mean())
                    print('TEST LENGTH QUANTILESDF', len(mass_nonzero_), mass_nonzero_)
                
                if weighted_stats_for_outliers and len(mass_nonzero_) >= 5: # use weighted stats if specified, UNLESS there are less than 5 non-zero "mass" datapoints in total
                    from statsmodels.stats.weightstats import DescrStatsW
                    
                    # init weights for weighted outlier detection (list should sum to 1 total)
                    outlier_weights_ = [0.16]*5 # first 5 values weighted to 0.8 total
                    if len(mass_nonzero_) > 5: # add more weights if necessary (between 1 and 5 more)
                        outlier_weights_ += [((1-sum(outlier_weights_)) / (len(mass_nonzero_)-len(outlier_weights_)))] * (len(mass_nonzero_)-len(outlier_weights_))
                    
                    weighted_stats = DescrStatsW(mass_nonzero_.iloc[::-1], weights=outlier_weights_, ddof=0)

                    if pond_name == pond_test:
                        print(f'Weighted mean: {weighted_stats.mean}, Weighted Std Dev: {weighted_stats.std}, weighted outlier range (+- {outlier_stddev_thresh} std dev): <{weighted_stats.mean-(outlier_stddev_thresh*weighted_stats.std)}, >{weighted_stats.mean+(outlier_stddev_thresh*weighted_stats.std)}')
                        print(f'Non-Weighted mean: {mass_nonzero_.mean()}, Non-Weighted Std Dev: {mass_nonzero_.std()}, non-weighted outlier range (+- {outlier_stddev_thresh} std dev): <{mass_nonzero_.mean()-(outlier_stddev_thresh*mass_nonzero_.std())}, >{mass_nonzero_.mean()+(outlier_stddev_thresh*mass_nonzero_.std())}')
                        print('test df', test_mean := mass_nonzero_[-5:])
                    
                    mass_nonzero_mean_ = weighted_stats.mean
                    mass_nonzero_std_ = weighted_stats.std
                else: # using regular statistics / non-weighted
                    mass_nonzero_mean_ = mass_nonzero_.mean()
                    mass_nonzero_std_ = mass_nonzero_.std()

                if pond_name == pond_test:
                    print(f'TESTING FOR POND {pond_name}')
                    print('outliers\n', pond_growth_df[(pond_growth_df['mass']-mass_nonzero_mean_).apply(lambda x: -x if x < 0 else x) >= (outlier_stddev_thresh*mass_nonzero_std_)], 'testing outliers')
                
                #outliers_df = pond_growth_df[np.abs(pond_growth_df['mass']-mass_nonzero_mean_) >= (outlier_stddev_thresh*mass_nonzero_std_)]
                outliers_df = pond_growth_df[(pond_growth_df['mass']-mass_nonzero_mean_).apply(lambda x: -x if x < 0 else x) >= (outlier_stddev_thresh*mass_nonzero_std_)]
                pond_growth_df = pond_growth_df.assign(outlier=pond_growth_df.index.isin(outliers_df.index)) # assign 'outliers' column equal to True if data point is an outlier

                if pond_name == pond_test:
                    print('Starting data\n')
                    from IPython.display import display
                    display(pond_growth_df)

                # set the mass of outliers to 0
                pond_growth_df['mass'] = pond_growth_df.apply(lambda row: 0 if row['outlier'] == True else row['mass'], axis=1)

                if pond_name == pond_test:
                    print('\nAfter removing outliers\n')
                    display(pond_growth_df)
            
            # check if enough data exists for calculating growth, and return 'n/a' early if not
            growth_period_data_count = pond_growth_df.loc[select_date - pd.Timedelta(days=num_days-1):select_date][pond_growth_df['mass'] != 0]['mass'].count()
            if growth_period_data_count < data_count_threshold:
                return 'n/a'
            
            # forward fill zero-values
            pond_growth_df['mass'] = pond_growth_df['mass'].replace(0, float('nan')).fillna(method='ffill') # replace 0 with nan for fillna() to work

            if pond_name == pond_test:
                print('\nAfter forward-filling zero vals\n')
                display(pond_growth_df)

            # calculate estimated harvest amount, not used elsewhere but could be useful for some aggregate tracking
            pond_growth_df['tmp prev mass'] = pond_growth_df['mass'].shift(1)
            pond_growth_df['tmp next mass'] = pond_growth_df['mass'].shift(-1)
            pond_growth_df['harvested estimate'] = pond_growth_df.apply(lambda row: row['tmp prev mass'] - row['mass'] if row['next data since harvest'] == 'Y' else 0, axis=1)

            pond_growth_df['day chng'] = pond_growth_df.apply(lambda row: row['mass'] - row['tmp prev mass'] if row['next data since harvest'] != 'Y' else 0, axis=1)
            del pond_growth_df['tmp prev mass'] 
            del pond_growth_df['tmp next mass'] 

            if pond_name == pond_test:
                print('\nAfter calculating estimated harvest and daily change in mass')
                display(pond_growth_df)

            # calculate growth
            try:
                daily_growth_rate_ = int(pond_growth_df.loc[select_date - pd.Timedelta(days=num_days-1):select_date]['day chng'].sum() / num_days)
                if pond_name == pond_test:
                    display(daily_growth_rate_)
                return daily_growth_rate_
            except:
                return 'n/a: Error'

                ## TODO: outlier detection (pond 0402 - 1/16/23 as example of outlier to remove)
                # import numpy as np
#                         quantiles_df = pond_growth_df[pond_growth_df['mass'] != 0]['mass']
#                         print('std dev: ', quantiles_df.std())
#                         print('mean: ', quantiles_df.mean())
#                         print(quantiles_df[np.abs(quantiles_df-quantiles_df.mean()) <= (3*quantiles_df.std())])

#                         quantile_low = quantiles_df.quantile(0.01)
#                         quantile_high = quantiles_df.quantile(0.99)
#                         iqr = quantile_high - quantile_low
#                         low_bound = quantile_low - (1.5*iqr)
#                         high_bound = quantile_high + (1.5*iqr)
#                         print('low quantile: ', quantile_low, '| high quantile: ', quantile_high)
#                         print('low bound: ', low_bound, '| high bound: ', high_bound)
        
        def subplot_ax_format(subplot_ax):
            subplot_ax.spines['top'].set_visible(False)
            subplot_ax.spines['right'].set_visible(False)
            subplot_ax.spines['bottom'].set_visible(False)
            subplot_ax.spines['left'].set_visible(False)
            subplot_ax.get_xaxis().set_ticks([])
            subplot_ax.get_yaxis().set_ticks([])
            fig.add_subplot(subplot_ax)
         
        # function to plot outlined legend boxes, using a dict of labels/data, and a separate dict of x_alignments (see examples of dict structure in calls to this function)
        # def plot_legend(legend_data, x_align, y_spacing, y_align=0.9):
        #     y_align = y_align # start y-coord for plotting legend
        #     max_coords = [1,1,0,0] #  initialize for tracking the max coordinates of legend items for printing a box around all the legend items (x0, y0, x1, y1)
        #     data_rows_x_bound = [1,0]
        #     data_rows_y_bounds = [] 
        #     indicator_plots = []
        #     for row in legend_data:
        #         row_y_bound = [1,0]
        #         row_fill_color = 'white'
        #         if 'fill_color' in row:
        #             row_fill_color = row['fill_color']

        #         for idx, (label, properties) in enumerate(row['labels'].items()):
        #             underline = False
        #             text_weight = 'normal'
        #             text_color = 'black'

        #             if 'weight' in properties:
        #                 if properties['weight'] == 'underline': # check if 'underline' option since it's not an option in matplotlib and needs to be added after text with an annotation using text bounding box coords
        #                     underline = True
        #                 elif properties['weight'] == 'bold': 
        #                     text_weight = 'bold'

        #             if 'color' in properties:
        #                 text_color = properties['color']

        #             if 'indicator_fill' in properties: # plot indicators instead of normal text
        #                 # Get transform info to properly plot circles, using Ellipse. Otherwise they are deformed 
        #                 # ref: https://stackoverflow.com/questions/9230389/why-is-matplotlib-plotting-my-circles-as-ovals
        #                 ax_x0, ax_y0 = ax.transAxes.transform((0, 0)) # lower left in pixels
        #                 ax_x1, ax_y1 = ax.transAxes.transform((1, 1)) # upper right in pixes
        #                 ax_dx = ax_x1 - ax_x0
        #                 ax_dy = ax_y1 - ax_y0
        #                 ax_maxd = max(ax_dx, ax_dy)
        #                 circle_width = .07 * ax_maxd / ax_dx
        #                 circle_height = .07 * ax_maxd / ax_dy

        #                 # plot indicator
        #                 t = ax.add_patch(Ellipse((x_align[properties['align']][0], y_align), circle_width, circle_height, color=properties['indicator_fill'], fill=properties['indicator_fill'], clip_on=False))
        #                 ax.text(x_align[properties['align']][0], y_align, label, color='white', ha='center', va='center', fontweight='bold', fontsize='x-small', linespacing=0.7, clip_on=False)

        #             else:
        #                 t = ax.text(x_align[properties['align']][0], y_align, label, ha = x_align[properties['align']][1], va = 'center', weight=text_weight, color=text_color)

        #             # get bounding box of plotted items
        #             bb = t.get_window_extent(renderer=fig.canvas.get_renderer()).transformed(ax.transAxes.inverted())

        #             if underline:
        #                 ax.annotate('', xy=(bb.x0-0.01,bb.y0), xytext=(bb.x1+0.01,bb.y0), xycoords="axes fraction", arrowprops=dict(arrowstyle="-", color='k'))

        #             # update the max coord boundaries for the rows that are assigned a color
        #             if row_fill_color != 'white':  
        #                 #print(label, bb)
        #                 if 'excl_color' not in properties: # add option of excluding a row item from being colored
        #                     if bb.x0 < data_rows_x_bound[0]:
        #                         data_rows_x_bound[0] = bb.x0
        #                     if bb.x1 > data_rows_x_bound[1]:
        #                         data_rows_x_bound[1] = bb.x1
        #                     if bb.y0 < row_y_bound[0]:
        #                         row_y_bound[0] = bb.y0
        #                     if bb.y1 > row_y_bound[1]:
        #                         row_y_bound[1] = bb.y1

        #             # update max_coords for bounding box around legend and for x-bounds of rows that will be filled with color
        #             if bb.x0 < max_coords[0]:
        #                 max_coords[0] = bb.x0
        #             if bb.y0 < max_coords[1]:
        #                 max_coords[1] = bb.y0
        #             if bb.x1 > max_coords[2]:
        #                 max_coords[2] = bb.x1
        #             if bb.y1 > max_coords[3]:
        #                 max_coords[3] = bb.y1

        #         y_align -= y_spacing # decrease y coord for each line for spacing       

        #         if row_fill_color != 'white':
        #             data_rows_y_bounds.append((row_y_bound,row_fill_color))

        #     #print('rows x-bound:', data_rows_x_bound)    
        #     #print('row y-bounds:', data_rows_y_bounds)
            
        #     # plot colored rectangle for corresponding row
        #     row_box_padding = 0.01
        #     for row_y in data_rows_y_bounds:
        #         ax.add_patch(Rectangle((data_rows_x_bound[0]-row_box_padding,row_y[0][0]-row_box_padding),data_rows_x_bound[1]-data_rows_x_bound[0]+row_box_padding*2,row_y[0][1]-row_y[0][0]+row_box_padding*2,linewidth=0,edgecolor=None,facecolor=row_y[1], clip_on=False))
            
        #     # plot color filled box around legend text
        #     box_padding = 0.03
        #     box_xy = (max_coords[0]-box_padding, max_coords[1]-box_padding)
        #     box_width = max_coords[1]-max_coords[0]
        #     ax.add_patch(Rectangle((max_coords[0]-box_padding,max_coords[1]-box_padding),max_coords[2]-max_coords[0]+box_padding*2,max_coords[3]-max_coords[1]+box_padding*2,linewidth=1,edgecolor='black',facecolor='none',clip_on=False))

        
        # function to plot each pond to ensure that data for each plot is kept within a local scope
        def plot_each_pond(fig, pond_plot, pond_name, select_date):
            inner_plot = gridspec.GridSpecFromSubplotSpec(5,3,subplot_spec=pond_plot, wspace=-0.01, hspace=-0.01) # each individual plot (from gridspec) is divided into subplots with 5 rows and 3 columns

            # Check prior 5 days of data to see if pond is active/in-use
            pond_active = self.active_dict[pond_name]

            try:
                single_pond_data = self.scorecard_dataframe[pond_name]
                pond_data_error = False
            except:
                pond_data_error = True
            
            # Get the last harvest date for each pond, done separately from other data queries due to needing full range of dates, and regardless of whether it's active
            # the date found is relative to the select_date parameter (so reports can always be generated relative to a specific date even for past dates)
            try:
                last_harvest_idx = single_pond_data['Split Innoculum'].loc[:select_date].last_valid_index()
                pond_last_harvest_str = last_harvest_idx.date().strftime('%-m-%-d-%y')
                pond_days_since_harvest = (select_date - last_harvest_idx).days 
            except:
                last_harvest_idx = 'n/a'
                pond_last_harvest_str = 'n/a'
            
            # Gather current data for pond if it's active
            if pond_active == True:
                # update aggregations for number of active ponds
                nonlocal num_active_ponds, num_active_ponds_sm, num_active_ponds_lg
                #check if pond is in the '06' or '08' column by checking the last values in name
                if pond_name[2:] == '06' or pond_name[2:] == '08':
                     num_active_ponds_lg += 1
                else:
                    num_active_ponds_sm += 1
                num_active_ponds += 1

                # get most recent EPA value
                try:
                    epa_date = list(self.epa_data_dict[pond_name].keys())[0] # get 0 index since the epa data should be sorted by most-recent first
                    epa_val = self.epa_data_dict[pond_name][epa_date]
                except:
                    epa_val = ''
                
                # get dataframe for individual pond for current date
                date_single_pond_data = single_pond_data.loc[select_date] 

                # calculate data for pond subplot display
                pond_data_afdw, pond_data_afdw_noncurrent_flag = current_or_prev_query(single_pond_data, 'AFDW (filter)', 1, return_noncurrent_flag=True)
                pond_data_depth, pond_data_depth_noncurrent_flag = current_or_prev_query(single_pond_data, 'Depth', 1, return_noncurrent_flag=True)
                pond_data_depth = math.floor(pond_data_depth*8)/8 # round depth to nearest 1/8 inches (data should already be input this way, but this ensures that data entry errors are corrected)
                # update 'any_noncurrent_flag' var to True if either AFDW or Depth is flagged as noncurrent, for use with adding explanation note to the plot when this flag is true
                if pond_data_afdw_noncurrent_flag == 'noncurrent' or pond_data_depth_noncurrent_flag == 'noncurrent': 
                    nonlocal any_noncurrent_flag
                    any_noncurrent_flag = True
                # calculate mass and harvestable amount if density and depth are both nonzero
                if (pond_data_afdw != 0 or pond_data_depth != 0) == False:
                    pond_data_total_mass = int(afdw_depth_to_mass(pond_data_afdw, pond_data_depth, pond_name))
                    # calculate harvestable depth (in inches) based on depth and afdw and the target_topoff_depth & target_to_density global function parameters
                    # rounding down to nearest 1/8 inch
                    pond_data_harvestable_depth = math.floor((((pond_data_depth * pond_data_afdw) - (target_topoff_depth * target_to_density)) / pond_data_afdw)*8)/8
                    if pond_data_harvestable_depth < 0: 
                        pond_data_harvestable_depth = 0
                    # calculate the depth to harvest pond to (i.e., the resulting depth after it has been harvested with the pond_data_harvestable_depth number of inches)   
                    pond_data_target_depth_harvest_to = pond_data_depth - pond_data_harvestable_depth
                    # calculate harvestable volume (in gallons) based on harvestable depth and conversion factor (35,000) to gallons. Double for the '06' and '08' column ponds since they are double size
                    pond_data_harvestable_gallons = pond_data_harvestable_depth * 35000
                    if pond_name[-2:] == '06' or pond_name[-2:] == '08':
                        pond_data_harvestable_gallons *= 2
                    pond_data_harvestable_mass = int(afdw_depth_to_mass(pond_data_afdw, pond_data_harvestable_depth, pond_name))

                    # Add pond info to global counters/data for the entire farm
                    nonlocal total_mass_all
                    total_mass_all += pond_data_total_mass # add pond mass to the total_mass_all counter for entire farm
                    if pond_data_afdw > harvest_density and pond_data_harvestable_depth > 0 and epa_val >= 0.03: # add these only if current pond density is greater than the global function parameter 'harvest_density'
                        nonlocal potential_harvests, potential_total_harvest_mass, potential_total_harvest_gals
                        pond_column = pond_name[2:]
                        potential_harvests['data'].setdefault(pond_column, []) # use .setdefault methods to first populate dict key for column (if it doesn't already exist), and an empty list to collect data for each
                        tmp_dict = {}
                        tmp_dict['Pond Number'] = pond_name
                        tmp_dict['Drop To'] = pond_data_target_depth_harvest_to
                        tmp_dict['Days Since Harvested'] = pond_days_since_harvest
                        tmp_dict['Harvestable Depth'] = pond_data_harvestable_depth 
                        tmp_dict['Harvestable Gallons'] = pond_data_harvestable_gallons
                        tmp_dict['Harvestable Mass'] = pond_data_harvestable_mass
                        potential_harvests['data'][pond_column].append(tmp_dict)
                        
                        # update total potential harvest amounts
                        potential_total_harvest_mass += pond_data_harvestable_mass
                        potential_total_harvest_gals += pond_data_harvestable_gallons
                else:
                    pond_data_total_mass = 0
                    pond_data_harvestable_depth = 0
                    pond_data_harvestable_mass = 0
                
                # Query pests data for each pond
                indicators_data = date_single_pond_data[['pH', 'Rotifers ','Attached FD111','Free Floating FD111', 'Golden Flagellates', 'Diatoms', 
                                    'Tetra','Green Algae']].apply(pd.to_numeric, errors='coerce').fillna(0)
             
                # Get the % nanno indicator separately and join to indicators_data
                # This is due to the value sometimes being missing, so look back to previous days (up to 2) to get it
                pct_nanno = current_or_prev_query(single_pond_data, '% Nanno', 2)
                indicators_data['% Nanno'] = pct_nanno 
                
                # key for pests/indicators, each value a list with comparison operator to use, the threshold for flagging 
                # and the color to display it as
                indicator_dict = {'% Nanno': ['less', 80, '%N', 'xkcd:fire engine red'],
                                 'pH': ['out-of-range', [7.8, 8.2], 'pH', 'xkcd:fire engine red'],
                                 'Rotifers ': ['greater-equal', 1, 'R', 'xkcd:deep sky blue'],   
                                'Attached FD111': ['greater-equal', 1, 'FD\nA', 'xkcd:poo brown'],    
                                'Free Floating FD111': ['greater-equal', 1, 'FD\nFF', 'orange'],  
                                'Golden Flagellates': ['greater-equal', 1, 'GF', 'y'], 
                                'Diatoms': ['greater-equal', 0.5, 'D', 'xkcd:powder pink'],
                                'Tetra': ['greater-equal', 0.5, 'T', 'purple'],
                                'Green Algae': ['greater-equal', 0.5 , 'GA', 'green']} 
                # append the abbreviated symbol to plot, and format color of indicator, to the pond_indicator_data list
                pond_indicator_data = []
                for idx, (key, val) in enumerate(indicator_dict.items()):
                    comparison_operator = val[0]
                    threshold = val[1]
                    if comparison_operator == 'greater-equal':
                        if indicators_data[key] >= threshold:
                            pond_indicator_data.append([val[2], val[3]])
                    if comparison_operator == 'less':
                        if indicators_data[key] < threshold:
                            pond_indicator_data.append([val[2], val[3]])
                    if comparison_operator == 'out-of-range':
                        if indicators_data[key] < threshold[0] or indicators_data[key] > threshold[1]:
                            pond_indicator_data.append([val[2], val[3]])
                
                # set fill color 
                if pond_data_afdw == 0:
                    fill_color = 'lightgrey'
                else:    
                    density_color_dict = {(0.000000001,0.25): 0, 
                                          (0.25,0.5): 1,
                                          (0.50,0.8): 2, 
                                          (0.80,999999999): 3}
                    color_list = ['red', 'yellow', 'mediumspringgreen', 'tab:green']
                    for idx, (key, val) in enumerate(density_color_dict.items()):
                        if pond_data_afdw >= key[0] and pond_data_afdw < key[1]:
                            color_idx = val
                            fill_color = color_list[color_idx] 
                    # secondary fill color by EPA percentage
                    if epa_val == '':
                        fill_color='lightgrey'
                    elif color_idx > 0 and epa_val < 2.5:
                        fill_color = 'tan' # indicate out-of-spec EPA value
                    elif color_idx > 1 and epa_val < 3.0:
                        fill_color = 'yellow'
                
            ## Calculate growth rates 
            if pond_active == True:   
                daily_growth_rate_5_days = calculate_growth(pond_data_df=single_pond_data, select_date=select_date, num_days=5, remove_outliers=False)
                daily_growth_rate_5_days_corrected = calculate_growth(pond_data_df=single_pond_data, select_date=select_date, num_days=5, remove_outliers=True, weighted_stats_for_outliers=True)  
                
            # title subplot
            title_ax = plt.Subplot(fig, inner_plot[:2,:])

            # plot pond EPA data
            try:
                if epa_val == '':
                    epa_val_str = 'No EPA data'
                else:
                    epa_val_str = f'{epa_val:.2f}% EPA'
                    title_ax.text(0.95, 0.8, epa_date.strftime('as of\n%-m/%-d'), ha='center', va='center', fontsize='x-small')
                title_ax.text(0.78, 0.8, epa_val_str, ha='center', va='center')
            except:
                pass

            title_ax.text(0.5, 0.8, r'$\bf{' + pond_name + '}$', ha = 'center', va='center', fontsize='large')
            title_ax.text(0.5, 0.5, f'last harvest/split: {pond_last_harvest_str}', ha='center', va='center')

            # Display pond pest/indicator info under subplot title for active ponds
            if pond_active == True:
                try:
                    # Get transform info to properly plot circles, using Ellipse. Otherwise they are deformed 
                    # ref: https://stackoverflow.com/questions/9230389/why-is-matplotlib-plotting-my-circles-as-ovals
                    x0, y0 = title_ax.transAxes.transform((0, 0)) # lower left in pixels
                    x1, y1 = title_ax.transAxes.transform((1, 1)) # upper right in pixes
                    dx = x1 - x0
                    dy = y1 - y0
                    maxd = max(dx, dy)
                    circle_width = .07 * maxd / dx
                    circle_height = .07 * maxd / dy

                    width_indicator_data = len(pond_indicator_data)*(circle_width+.01) # get the width of the indicator data (when plotted in circles)
                    calc_start_x = .5 - (width_indicator_data/2) + ((circle_width+0.01)/2) # calculate which X coord to start plotting indicators (to center them on each subplot)
                    indicator_plot_y = .18
                    x_space = circle_width + 0.01
                    #if width_indicator_data > 0: 
                    #    ax.text(calc_start_x - 0.05,indicator_plot_y, 'Indicators:', ha='right', va='center')
                    for [indicator_string,color] in pond_indicator_data:
                        title_ax.add_patch(Ellipse((calc_start_x, indicator_plot_y + 0.02), circle_width, circle_height, color=color, fill=color))
                        title_ax.text(calc_start_x, indicator_plot_y, indicator_string, color='white', ha='center', va='center', fontweight='bold', fontsize='x-small', linespacing=0.7)
                        calc_start_x += x_space
                except: 
                    title_ax.text(0.5, 0.15, 'ERROR PLOTTING INDICATORS', ha='center')                   
            subplot_ax_format(title_ax)
            
            if pond_active == True: 
               
                # format data for plotting
                if pond_data_depth == 0:
                    pond_data_depth = 'No data'
                else:    
                    pond_data_depth = f'{"**" if pond_data_depth_noncurrent_flag == "noncurrent" else ""}{pond_data_depth}"' # print two asterisks before if data was flagged as 'non-current'
                
                if pond_data_afdw == 0:
                    pond_data_afdw = 'No data'
                else:
                    pond_data_afdw = f'{"**" if pond_data_afdw_noncurrent_flag == "noncurrent" else ""}{round(pond_data_afdw,3)}' # print two asterisks before if data was flagged as 'non-current'
                
                if pond_data_harvestable_mass == 0:
                    pond_data_harvestable_mass = '-'
                else:
                    pond_data_harvestable_mass = f'{pond_data_harvestable_mass} kg' 
                
                if pond_data_harvestable_depth == 0:
                    pond_data_harvestable_depth = '-'
                else:
                    pond_data_harvestable_depth = f'{pond_data_harvestable_depth}"'
                
                # format corrected growth rate prior to formatting regular growth rate since equality check needs to be done (otherwise adding "kg/day" to end causes issues/or extra steps)
                if daily_growth_rate_5_days_corrected == daily_growth_rate_5_days:
                    daily_growth_rate_5_days_corrected = '' # set corrected rate to empty string so it wont be shown, if it's not different from the "regular" rate
                elif daily_growth_rate_5_days_corrected == 'n/a':
                    if daily_growth_rate_5_days == 'n/a':
                        daily_growth_rate_5_days_corrected = '' # set non-calculable corrected rate to empty string, if "regular" rate is also non-calculable
                    else:
                        pass # keep 'n/a' as value for corrected growth rate if it is non-calculable, but there is a valid "regular" growth rate (to indicate that there's something wrong with the "regular" rate)
                else: 
                    daily_growth_rate_5_days_corrected = f'{daily_growth_rate_5_days_corrected} kg/day'
                
                if daily_growth_rate_5_days == 'n/a':
                    pass # don't need to reformat in this case
                else:
                    daily_growth_rate_5_days = f'{daily_growth_rate_5_days} kg/day'
                
                data_plot_dict = {'Measurement': {'Depth:': pond_data_depth, 'AFDW:': pond_data_afdw},
                                  'Growth': {'5 Days:': daily_growth_rate_5_days,  f'{"Corrected:" if daily_growth_rate_5_days_corrected != "" else ""}': daily_growth_rate_5_days_corrected},
                                  'Harvestable': {'Mass:': pond_data_harvestable_mass, 'Depth:': pond_data_harvestable_depth}
                                  }

            # data subplots   
            for idx in range(3): 
            # this would be easier to iterate through the data_plot_dict since accessing its items
            # however, this dict doesn't exist for ponds that aren't active, so just iterating through a range instead
                ax = plt.Subplot(fig, inner_plot[2:,idx])
                
                if pond_active == True: 
                    subplot_dict_key = list(data_plot_dict.keys())[idx] 
                    text_y = 0.83 # starting y-coordinate for text on subplot, updated after each text plot to decrease y-coord
                    sub_heading = ax.text(0.5, text_y, subplot_dict_key, ha='center', va='center') # subplot heading text
                    # get sub_heading coord bounding box and use to draw a line with annotate arrowprops / using this workaround for underlining text
                    bb = sub_heading.get_window_extent(renderer=fig.canvas.get_renderer()).transformed(ax.transAxes.inverted())  
                    ax.annotate('', xy=(bb.x0-0.02,bb.y0), xytext=(bb.x1+0.025,bb.y0), xycoords="axes fraction", arrowprops=dict(arrowstyle="-", color='k'))
                    text_y -= 0.19
                    for idx2, (key, val) in enumerate(data_plot_dict[subplot_dict_key].items()):
                        ax.text(0.5, text_y, key, ha='center', va='center') # subplot item heading
                        text_y -= 0.14
                        ax.text(0.5, text_y, val, ha='center', va='center', weight='bold')
                        text_y -= .2
                    ax.set_facecolor(fill_color)
                    
                else: # if pond is inactive or data_error
                    if idx == 1: # plot in the middle of the lower 3 subplots
                        if pond_data_error == True:
                            t = ax.text(0.5, 0.9, 'Data Error', ha='center', va='top')
                        else:
                            t = ax.text(0.5, 0.9, 'Inactive', ha='center', va='top')
                    ax.set_facecolor('snow')
                subplot_ax_format(ax)
            
        ##############################
        ### START OF PLOTTING CODE ###
        ##############################

        n_rows_small = 12
        n_rows_large = 10
        n_cols_small = 4
        n_cols_large = 2

        # generate labels for ponds in a list (with each row being a sublist)
        title_labels = []
        for row in range(1,n_rows_small+1):
            title_labels.append([])
            for col in range(1,n_cols_small+1):
                if row < 10:
                    title_labels[row-1].append(f'0{row}0{col}')  
                else:
                    title_labels[row-1].append(f'{row}0{col}')  
        for idx_row, row in enumerate(title_labels):
            if idx_row < n_rows_large:
                for col in [6,8]:
                    if idx_row+1 < 10:
                        row.append(f'0{idx_row+1}0{col}')
                    else:
                        row.append(f'{idx_row+1}0{col}')
            else: # append blanks for the two large columns with only 10 rows 
                [row.append(f'BLANK {idx_row+1}-{c}') for c in [6,8]]
                 
        # Add 2 blank rows at end to make room for extra data aggregations to be listed        
        [title_labels.append([f'BLANK {r}-{c}' for c in range(1,7)]) for r in range(13,15)]
        
        # flatten title_labels for indexing from plot gridspec plot generation
        flat_title_labels = [label for item in title_labels for label in item]
        
        # Initialize main plot
        plt_width = 8.5
        plt_height = 11
        scale_factor = 3
        fig = plt.figure(figsize=(plt_width*scale_factor, plt_height*scale_factor))

        # setup GridSpec for subplots in a grid
        outer_plots = gridspec.GridSpec(len(title_labels), len(title_labels[0]), wspace=0.05, hspace=0.1)
        
        #title_date = '/'.join([select_date.split('-')[i].lstrip('0') for i in [1,2,0]])
        title_date = select_date.strftime('%-m/%-d/%Y')
        fig.suptitle(f'{plot_title}\n{title_date}', fontweight='bold', fontsize=16, y=0.905)
            
        for idx_plot, pond_plot in enumerate(outer_plots):
            pond_name = flat_title_labels[idx_plot]
            if 'BLANK' not in pond_name:
                
                # plot each pond with a function to ensure that data for each is isolated within a local scope
                plot_each_pond(fig, pond_plot, pond_name, select_date)

            else: # for plotting subplots labeled 'BLANK' in 'title_labels' list:  (the four lower right subplots) and the BLANK rows
                ax = plt.Subplot(fig, pond_plot)
                ax.axis('off')
                if 'BLANK 11-6' in pond_name: # plot the color key in the first blank subplot in row 11

                    legend_data = [{'labels':{'Color key': {'align': '1-l', 'weight': 'bold'}}},
                                   {'labels':{'AFDW': {'align': '1-c', 'weight': 'underline'}, 'EPA %': {'align': '3-c', 'weight': 'underline'}}},
                                   {'labels':{r'$<$ 0.25': {'align': '1-r'}, 'and':{'align':'2'}, 'any': {'align': '3-r'}}, 'fill_color': 'red'},
                                   {'labels':{r'$\geq$ 0.25': {'align': '1-r'}, 'and': {'align': '2'}, r'$<$ 2.5%': {'align': '3-r'}, 'out of spec EPA but AFDW OK': {'align': '4-l', 'excl_color':'Y'}}, 'fill_color': 'tan'},
                                   {'labels':{'0.25 - 0.49': {'align': '1-r'}, 'or': {'align': '2'}, '2.5% - 2.99%': {'align': '3-r'}}, 'fill_color': 'yellow'},
                                   {'labels':{'0.50 - 0.79': {'align': '1-r'}, 'and': {'align': '2'}, r'$\geq$ 3.0%': {'align': '3-r'}}, 'fill_color': 'mediumspringgreen'},
                                   {'labels':{r'$\geq$ 0.80': {'align': '1-r'}, 'and': {'align': '2'}, r'$\geq$ 3.0%': {'align': '3-r'}}, 'fill_color': 'tab:green'},
                                   {'labels':{'Incomplete data': {'align': '2'}},'fill_color': 'lightgrey'}
                                  ]
                    x_align = {'1-l': [0.1, 'left'], 
                               '1-c': [0.23, 'center'],
                               '1-r': [0.35, 'right'],
                               '2': [0.48, 'center'],
                               '3-c': [0.75, 'center'],
                               '3-r': [0.9, 'right'],
                               '4-l': [0.94, 'left'],
                               '4-c': [1.05, 'center']}
                                                                                                                                                                                                                        
                    self.plot_legend(fig, ax, legend_data, x_align, y_spacing=0.12)
                    
                if 'BLANK 12-6' in pond_name: # plot the indicator legend in the first blank subplot in row 12                       
                    legend_data = [{'labels':{'Health/Pest Indicators': {'align': '1-l', 'weight': 'bold'}}},
                                             {'labels':{'Item': {'align': '1-l', 'weight': 'underline'}, 'Threshold': {'align': '2', 'weight': 'underline'}, 'Symbol': {'align': '3-c', 'weight': 'underline'}}},
                                             {'labels':{'% Nanno': {'align': '1-l'}, r'$<$ 80%': {'align': '2'}, '%N': {'align': '3-c', 'indicator_fill': 'xkcd:fire engine red'}}},
                                             {'labels':{'pH ': {'align': '1-l'}, 'out of 7.8 - 8.2 range': {'align': '2'}, 'pH': {'align': '3-c', 'indicator_fill': 'xkcd:fire engine red'}}},
                                             {'labels':{'Rotifers': {'align': '1-l'}, r'$\geq$ 1': {'align': '2'}, 'R': {'align': '3-c', 'indicator_fill': 'xkcd:deep sky blue'}}},
                                             {'labels':{'Attached FD111': {'align': '1-l'}, r'$\geq$ 1': {'align': '2'}, 'FD\nA': {'align': '3-c', 'indicator_fill': 'xkcd:poo brown'}}},
                                             {'labels':{'Free Floating FD111': {'align': '1-l'}, r'$\geq$ 1': {'align': '2'}, 'FD\nFF': {'align': '3-c', 'indicator_fill': 'orange'}}},
                                             {'labels':{'Golden Flagellates': {'align': '1-l'}, r'$\geq$ 1': {'align': '2'}, 'GF': {'align': '3-c', 'indicator_fill': 'y'}}},
                                             {'labels':{'Diatoms': {'align': '1-l'}, r'$\geq$ 0.5': {'align': '2'}, 'D': {'align': '3-c', 'indicator_fill': 'xkcd:powder pink'}}},
                                             {'labels':{'Tetra': {'align': '1-l'}, r'$\geq$ 0.5': {'align': '2'}, 'T': {'align': '3-c', 'indicator_fill': 'purple'}}},
                                             {'labels':{'Green Algae': {'align': '1-l'}, r'$\geq$ 0.5': {'align': '2'}, 'GA': {'align': '3-c', 'indicator_fill': 'green'}}},
                                  ]
                    # x coord alignments dict, with list of x-coord position text justification info
                    x_align = {'1-l': [0.1, 'left'], 
                               '2': [0.85, 'center'],
                               '3-c': [1.25, 'center']}
                    
                    self.plot_legend(fig, ax, legend_data, x_align, y_spacing=0.15)
                    
                    
                elif 'BLANK 13-1' in pond_name:
                    ''' 
                    Plot information of various daily aggregates in this cell (13th row, column 1)
                    '''
                    # add note for noncurrent AFDW or Depth data when they are being used
                    if any_noncurrent_flag: 
                        ax.text(0,0.95, "** indicates AFDW or Depth data that is not available for the current day, so using data from previous day", ha='left', va='center', fontsize='small')
                    
                    print_string = (r'$\bf{Total\/\/Active\/\/ponds: }$' + f'{num_active_ponds}\n' 
                                    + r'$\bf{Active\/\/1.1\/\/acre\/\/ponds: }$' + f'{num_active_ponds_sm}\n'
                                    + r'$\bf{Active\/\/2.2\/\/acre\/\/ponds: }$' + f'{num_active_ponds_lg}\n'
                                   + r'$\bf{Calculated\/\/total\/\/mass:}$' + f'{total_mass_all:,} kg')
                    t = ax.text(0,0.75, print_string, ha='left', va='top', fontsize=16, multialignment='left')        
          
                elif 'BLANK 14-1' in pond_name:
                    '''
                    Plot information of prior day processing totals from self.processing_dataframe
                    on row 13, column 3
                    '''
                    processing_columns = {'Zobi Volume:': 
                                              {'column_name': 'Zobi Permeate Volume (gal)',
                                                'data_format': int,
                                                'str_label': 'gal'
                                                },
                                          'SF Volume:': 
                                              {'column_name': ['Calculated SF Permeate Volume (gal)', 'SF Reported Permeate Volume (gal)'],  # 1st is primary source, 2nd is secondary source if no data for primary
                                               'data_format': int,
                                               'str_label': 'gal'
                                                },
                                          'Dryer SW:': 
                                              {'column_name': 'SW Dryer Biomass (MT)', 
                                               'data_format': float,
                                               'str_label': 'Mt'
                                                },
                                          'Dryer DD:': 
                                              {'column_name': 'Drum Dryer Biomass (MT)', 
                                               'data_format': float,
                                               'str_label': 'Mt'
                                                },
                                          'Gallons Dropped:': 
                                              {'column_name': 'Gallons dropped',
                                               'data_format': int,
                                               'str_label': 'gal'
                                              },
                                          'Processing Notes:':
                                              {'column_name': 'Notes',
                                               'data_format': str,
                                               'str_label': ''
                                              }
                                         }
                    prev_date = select_date - pd.Timedelta(days=1)
                    
                    processing_data_str = f'Previous Day ({(select_date-pd.Timedelta(days=1)).strftime("%-m/%-d")}) Processing Totals'
                    proc_t = ax.text(0,1, processing_data_str, ha='left', va='top', fontsize='large', multialignment='left')
                    bb = proc_t.get_window_extent(renderer=fig.canvas.get_renderer()).transformed(ax.transAxes.inverted())
                    ax.annotate('', xy=(bb.x0-0.01,bb.y0), xytext=(bb.x1+0.01,bb.y0), xycoords="axes fraction", arrowprops=dict(arrowstyle="-", color='k'))

                    try: # use try/except to catch the case when processing data isn't available (would raise an exception otherwise)
                        prev_date_data = self.processing_dataframe.loc[prev_date]
                        for k, subdict in processing_columns.items():
                            if type(subdict['column_name']) == list: # when 'column_name' is a list, get the first item with data (so first item in list is primary source)
                                for i in subdict['column_name']:
                                    try: # force the data item to the 'data_format' type, with try/except to catch errors
                                        data_i = subdict['data_format'](prev_date_data[i])
                                        if (subdict['data_format'] == str and (data_i != '' or data_i != 'nan')) or (subdict['data_format'] != str and data_i != 0):
                                            prev_day_val = data_i
                                            break # if valid data is found, then break and stop evaluating any more columns
                                        else:
                                            prev_day_val = None
                                    except: 
                                        prev_day_val = None
                            else: 
                                try:
                                    prev_day_val = subdict['data_format'](prev_date_data[subdict["column_name"]])
                                    if (subdict['data_format'] == str and (prev_day_val == '' or prev_day_val == 'nan')) or (subdict['data_format'] != str and prev_day_value == 0):
                                        prev_day_val = None
                                except:
                                    prev_day_val = None
                                
                            if pd.isna(prev_day_val):
                                if subdict['data_format'] == str:
                                    prev_day_val = ''
                                else:
                                    prev_day_val = 0
                                
                            if type(prev_day_val) == str:
                                prev_day_val = prev_day_val.replace("/ ", "\n  ").replace(". ", ".\n  ")
                                processing_data_str += f'\n{k}\n  {prev_day_val}'
                            elif type(prev_day_val) != str:
                                prev_day_val = f'{subdict["data_format"](prev_day_val):,} {subdict["str_label"]}'
                                processing_data_str += f'\n{k} {prev_day_val}'
                    except:
                        processing_data_str += f'\nError: processing data unavailable for {prev_date.strftime("%-m/%-d/%y")}'
                    proc_t.update({'text': processing_data_str})
                    
                    # print_string = (r'$\bf{Harvest\/\/Depth} =$' + '\n' + r'$\frac{(Current\/\/Depth * Current\/\/AFDW) - (Target\/\/Top\/\/Off\/\/Depth * Target\/\/Harvest\/\/Down\/\/to\/\/AFDW)}{Current\/\/AFDW}$' +
                    #                 '\n' + r'$\bf{Target\/\/Harvest\/\/at\/\/AFDW}:$' + r'$\geq$' + str(harvest_density) +
                    #                 '\n' + r'$\bf{Target\/\/Harvest\/\/Down\/\/to\/\/AFDW}: $' + str(target_to_density) + 
                    #                 '\n' + r'$\bf{Target\/\/Top\/\/Off\/\/Depth}: $' + str(target_topoff_depth) + '"')
                    # t = ax.text(0,.8, print_string, ha='left', va='top', fontsize=16) 
                    
                # elif 'BLANK 14-3' in pond_name:
                    # print_string = (r'$\bf{Harvestable\/\/Mass} =$' + '\n' + r'$Harvest\/\/Depth * 132,489 (\frac{liters}{inch}) * Current\/\/AFDW$'
                    #                 + '\n(doubled for 06 and 08 columns)')
                    # t = ax.text(1.25,.8, print_string, ha='left', va='top', fontsize=16) 
                
                fig.add_subplot(ax) # add the subplot for the 'BLANK' entries
        
        # Update Suggested Harvests data
        # first sort potential_harvests by column (numbered 1, 2, 3, 4, 6, 8 if any ponds are active in them), then within each column by 'days since harvested' then 'harvestable mass', both in descending order
        potential_harvests['data'] = dict(sorted(potential_harvests['data'].items(), key=lambda x: x[0]))
        for idx, col in enumerate(potential_harvests['data'].keys()):
            potential_harvests['data'][col] = sorted(potential_harvests['data'][col], key=lambda x: (x['Days Since Harvested'], x['Harvestable Mass']), reverse=True)                                                                                                                                                                                                                                                                                                       
            # add cumulative totals to each pond now that they are sorted
            tmp_gals = 0
            tmp_mass = 0
            for pond in potential_harvests['data'][col]:
                tmp_gals += pond['Harvestable Gallons']
                tmp_mass += pond['Harvestable Mass']
                pond['Running Total Gallons'] = f'{tmp_gals:,.0f} gal'
                pond['Running Total Mass'] = f'{tmp_mass:,} kg'

                # update formatting of potential_harvests to display
                pond['Drop To'] = f'{pond["Drop To"]}"'
                pond['Days Since Harvested'] = f'{pond["Days Since Harvested"]} day{"s" if pond["Days Since Harvested"] != 1 else ""}'
                pond['Harvestable Depth'] = f'{pond["Harvestable Depth"]}"' # set dict values normally since keys should now exist
                pond['Harvestable Gallons'] = f'{pond["Harvestable Gallons"]:,.0f} gal'
                pond['Harvestable Mass'] = f'{pond["Harvestable Mass"]:,} kg'
        
        # add 'aggregates' to potential_harvests dict
        potential_harvests['aggregates'] = {'potential total mass': f'{potential_total_harvest_mass:,} kg', 'potential total volume': f'{potential_total_harvest_gals:,.0f} gallons'}
         
        self.potential_harvests_dict = potential_harvests # save potential_harvests as class variable to be available to plot_potential_harvests function

        if self.save_output:
            out_filename = f'./output_files/{plot_title} {select_date.strftime("%Y-%m-%d")}.pdf'
            plt.savefig(out_filename, bbox_inches='tight')
            return out_filename
    
    def plot_potential_harvests(self, overall_spacing=0.03): 
        potential_harvests_dict = self.potential_harvests_dict
        select_date = self.select_date

        # add 'columns' values to potential_harvests dict, get list of keys from the first column/first pond in 'data'
        if len(potential_harvests_dict.get('data', {})) > 0:
            potential_harvests_dict['columns'] = list(potential_harvests_dict['data'][list(potential_harvests_dict['data'].keys())[0]][0].keys())
        else:
            potential_harvests_dict['columns'] = []

        def gen_fig(title=False):
            # Initialize plot
            plt_width = 8.5
            plt_height = 11
            scale_factor = 1
            fig, ax = plt.subplots(figsize=(plt_width*scale_factor, plt_height*scale_factor))
            ax_border_pad = 0.5 # margin padding (in inches) between ax and full fig on all sides, just to maintain a margin for printing without needing to rescale, as well as room for header/footer info to be added (page numbers, etc.)
            ax_width = (plt_width - ax_border_pad*2) / plt_width
            ax_height = (plt_height - ax_border_pad*2) / plt_height 
            ax_left_x = (1 - ax_width)/2
            ax_bottom_y = (1 - ax_height)/2
            ax.set_position([ax_left_x, ax_bottom_y, ax_width, ax_height]) # set ax to fill entire figure (except for the border padding)
            ax.axis('off')
            #ax.yaxis.get_ticklocs(minor=True)
            #ax.minorticks_on()
            if title:
                title_text1 = f'Potential Harvests - {str(select_date.strftime("%-m/%-d/%y"))}' 
                title_text2 = ('\nponds with' + r'$\geq$' + '0.50 AFDW and' + r'$\geq$' + '3% EPA, with estimated harvest depth to reach 0.40 AFDW after top off to 13"\n\n' + 
                              f'Total estimated potential harvest mass: {potential_harvests_dict["aggregates"]["potential total mass"]}\n' +
                              f'Total estimated potential harvest volume: {potential_harvests_dict["aggregates"]["potential total volume"]}')
                t1 = ax.text(0.5, 1, title_text1, ha='center', va='top', fontsize=14, weight='bold')
                t2 = ax.text(0.5,0.992, title_text2, ha='center', va='top', fontsize=8)  
                # get the y-coordinate where tables can begin plotting on figure (after title text, or at top of ax if title text isn't present: i.e., page 2+)
                y0 = t2.get_window_extent(renderer=fig.canvas.get_renderer()).transformed(ax.transAxes.inverted()).y0 - overall_spacing
            else:
                y0 = 1
            return fig, ax, y0

        def plot_table(table_title, ax, y_start, data_fontsize=6.4, title_fontsize=8):
            table_title = ax.text(0.5, y_start, f'Column {table_title}', ha='center', va='top', fontsize=title_fontsize, weight='bold')
            table_title_dims = table_title.get_window_extent(renderer=fig.canvas.get_renderer()).transformed(ax.transAxes.inverted())

            table_y_max = table_title_dims.y0 - 0.0025 # add 0.0025 spacing (in terms of ax y-coordinate) between the title and top of table

            # plot a temporary table to get its dimensions, then remove it 
            # need to do this so that tables can be plotted on a single ax with even vertical spacing between them
            # the important dimension to capture here is the table height (after setting fontsize) because that will be variable depending on the length of the data, 
            # and the 'bbox' parameter for a matplotlib table (to plot in an exact specified location) requires a lower y-bound value, which isn't possible without knowing its height relative to where the table should start
            # (it would be more efficient to simply move this table, but that doesn't seem to be possible in matplotlib, so just removing and regenerating it instead once bounding box coordinates are calculated)
            temp_table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center')
            temp_table.auto_set_font_size(False)
            temp_table.set_fontsize(data_fontsize)
            temp_table.auto_set_column_width(col=list(range(len(df.columns))))
            temp_table_dims = temp_table.get_window_extent(renderer=fig.canvas.get_renderer()).transformed(ax.transAxes.inverted())
            temp_table.remove()

            table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', colColours=['whitesmoke']*len(df.columns), bbox=[temp_table_dims.x0, table_y_max - temp_table_dims.height, temp_table_dims.width, temp_table_dims.height])
            table.auto_set_font_size(False)
            table.set_fontsize(data_fontsize)
            table.auto_set_column_width(col=list(range(len(df.columns))))
            table_dims = table.get_window_extent(renderer=fig.canvas.get_renderer()).transformed(ax.transAxes.inverted())

            # catch if the lower y-bound of the table is less than 0 (meaning it exceeded the ax size), if so then remove the plotted data and return signal to start a new page
            if table_dims.y0 < 0: 
                table_title.remove()
                table.remove()
                return 'start_new_page'     
            else:
                return table_dims.y0 # return the minimum/bottom y_coordinate value for the table

        fig_list = [] # initialize list of figs (one for each output page, as necessary)
        tables_list = list(potential_harvests_dict['data'].keys())

        fig, ax, y_align = gen_fig(title=True)

        table_spacing = 0.035

        while tables_list:
            table_title = tables_list.pop(0) # set 'table_title' equal to the key from data

            # create dataframe from data in dict format 
            df = pd.DataFrame(potential_harvests_dict['data'][table_title], columns=potential_harvests_dict['columns']) 

            # call plot_table function, which returns the next y_coordinate for plotting a table to
            # once the y_coordinates for a table are outside of the ax bound (< 0) then the function will return the string 'start_new_page'
            # in that case, append the current figure to fig_list, then generate a new figure and re-plot the table on the new page
            y_align = plot_table(table_title, ax, y_align)
            if y_align == 'start_new_page':
                plt.show() # show plot for testing in jupyter nb
                fig_list.append(fig)
                fig, ax, y_align = gen_fig()
                y_align = plot_table(table_title, ax, y_align)
            y_align -= overall_spacing
        plt.show() # show plot for testing in jupyter nb
        fig_list.append(fig)     

        filename = f'./output_files/Potential Harvests {select_date.strftime("%Y-%m-%d")}.pdf'
        out_filename = generate_multipage_pdf(fig_list, filename, add_pagenum=True, bbox_inches=None)
        return out_filename

    def plot_epa(self, plot_title='Pond EPA Overview'):
        def subplot_ax_format(subplot_ax, fill_color):
            subplot_ax.set_facecolor(fill_color)
            subplot_ax.spines['top'].set_visible(False)
            subplot_ax.spines['right'].set_visible(False)
            subplot_ax.spines['bottom'].set_visible(False)
            subplot_ax.spines['left'].set_visible(False)
            subplot_ax.get_xaxis().set_ticks([])
            subplot_ax.get_yaxis().set_ticks([])
            fig.add_subplot(subplot_ax)

        # function to plot each pond to ensure that data for each plot is kept within a local scope
        def plot_each_pond(fig, pond_plot, pond_name):
            inner_plot = gridspec.GridSpecFromSubplotSpec(5,3,subplot_spec=pond_plot, wspace=0, hspace=0)

            # get pond active status (True or False) from ponds_active_status dict
            pond_active = self.active_dict[pond_name]

            # get data for individual pond only, as a list
            # each list entry should consist of a tuple containing the date and the epa value, sorted in descending order by date
            single_pond_data = list(self.epa_data_dict[pond_name].items())

             # check and update the latest_sample_date global variable if this pond has a more recent date
            if len(single_pond_data) != 0:
                nonlocal latest_sample_date
                if single_pond_data[0][0] > latest_sample_date:
                    latest_sample_date = single_pond_data[0][0]

            # title subplot
            title_ax = plt.Subplot(fig, inner_plot[0,:])
            title_str = r'$\bf{' + pond_name + '}$' 
            # if len(ponds_data[pond_name]['source']) > 0:
            #     title_str += '\nsource: ' + ponds_data[pond_name]['source']
            title_ax.text(0.5, 0.1, title_str , ha = 'center', va='bottom', fontsize='large')
            subplot_ax_format(title_ax, 'white')

            # epa data subplots
            epa_ax = plt.Subplot(fig, inner_plot[1:3,:])

            if pond_active:
                # set fill color based on latest EPA reading
                if len(single_pond_data) > 0:
                    # set lastest_epa_data as the first value in single_pond_data (which should be reverse sorted)
                    latest_epa_data = single_pond_data[0][1] 
                    fill_color_dict = {(0,1.99999999):'red', 
                                       (2.0,3.49999999): 'yellow',
                                       (3.5,99999999999): 'mediumspringgreen'}
                    for idx, (key, val) in enumerate(fill_color_dict.items()):
                        if latest_epa_data >= key[0] and latest_epa_data < key[1]:
                            fill_color = val 
                else:
                    # set fill color as light grey if there is no EPA data for the pond
                    fill_color = 'lightgrey'

                if len(single_pond_data) > 0:
                    # set the center-point of each data point on the subplot, depending on number of data points available
                    if len(single_pond_data) == 1:
                        text_x = [0.5]
                    elif len(single_pond_data) == 2:
                        text_x = [0.3, 0.7]
                    else:
                        text_x = [0.195, 0.5, 0.805]

                    for idx, item in enumerate(single_pond_data):
                        text_date_formatted =  item[0].strftime("%-m/%-d/%y")
                        epa_ax.text(text_x[idx], 0.7, text_date_formatted, ha='center', va='center')
                        text_epa_data_formatted =  f'{item[1]: .2f}%'
                        epa_ax.text(text_x[idx], 0.3, text_epa_data_formatted, ha='center', va='center', fontsize='large', weight='bold')
                else:
                    epa_ax.text(0.5, 0.5, 'Active but no data', ha='center', va='center')
            else: # if pond is inactive
                fill_color = 'whitesmoke'
                epa_ax.text(0.5, 0.5, 'Inactive', ha='center', va='center')

            subplot_ax_format(epa_ax, fill_color)

            # epa change subplot
            epa_pct_ax = plt.Subplot(fig, inner_plot[3:,:])

            if pond_active:
            # Get the % change in EPA values if there is >1 entry (use the first and last indexes of the single_pond_data list)
                if len(single_pond_data) != 0:
                    epa_pct_fill = 'xkcd:light grey'    

                    def calc_epa_pct_chg(data_curr: list, data_prev: list):
                        '''
                        inputs:
                            data_curr: [datetime value, epa float value]
                            data_prev: [datetime value, epa float value]

                        output:
                            pct_chg: calculated absolute change in percentage rounded to 2 decimal places and formatted with % character (str)
                            delta_days: number of days between readings (int)
                            pct_format_color: color for displaying percentage (str)
                        '''
                        delta_days = (data_curr[0] - data_prev[0]).days    

                        pct_chg = data_curr[1] - data_prev[1]
                        if pct_chg > 0:
                            pct_format_color = 'xkcd:emerald green'
                            pct_chg = f'+{pct_chg:.2f}%'
                        elif pct_chg < 0: 
                            pct_format_color = 'xkcd:fire engine red'
                            pct_chg = f'{pct_chg:.2f}%'
                        else:
                            pct_format_color = 'black'
                            pct_chg = f'{pct_chg:.2f}%'
                        return [pct_chg, delta_days, pct_format_color]
                        # else:
                        #     return ['n/a', delta_days, 'black']

                    if len(single_pond_data) == 2:
                        epa_pct_chg, delta_days, epa_pct_color = calc_epa_pct_chg(single_pond_data[0], single_pond_data[1])
                        text_formatted1 =  f'Change ({delta_days} day{"s" if delta_days > 1 else ""}):' 
                        epa_pct_ax.text(0.5, 0.7, text_formatted1, ha='center', va='center', fontsize='large')
                        epa_pct_ax.text(0.5, 0.3, epa_pct_chg, ha='center', va='center', fontsize='large', color=epa_pct_color, weight='bold')
                    elif len(single_pond_data) == 3:
                        epa_pct_chg1, delta_days1, epa_pct_color1 = calc_epa_pct_chg(single_pond_data[0], single_pond_data[1])
                        epa_pct_chg2, delta_days2, epa_pct_color2 = calc_epa_pct_chg(single_pond_data[0], single_pond_data[2])
                        text_formatted1 =  f'Change ({delta_days1} day{"s" if delta_days1 > 1 else ""}, {delta_days2} days):'
                        epa_pct_ax.text(0.5, 0.7, text_formatted1, ha='center', va='center', fontsize='large')
                        epa_pct_ax.text(0.3, 0.3, epa_pct_chg1, ha='center', va='center', fontsize='large', color=epa_pct_color1, weight='bold')
                        epa_pct_ax.text(0.7, 0.3, epa_pct_chg2, ha='center', va='center', fontsize='large', color=epa_pct_color2, weight='bold')
                    else: # if there is only one data point so no percentage change
                        epa_pct_ax.text(0.5, 0.7, 'Change:', ha='center', va='center', fontsize='large')
                        epa_pct_ax.text(0.5, 0.3, 'n/a', ha='center', va='center', fontsize='large')

                else: # when there is no data for this pond
                    epa_pct_fill = 'lightgrey'

            else: # if pond is inactive
                epa_pct_fill = 'whitesmoke'

            subplot_ax_format(epa_pct_ax, epa_pct_fill) 

        ##############################
        ### START OF PLOTTING CODE ###
        ##############################

        n_rows_small = 12
        n_rows_large = 10
        n_cols_small = 4
        n_cols_large = 2

        # generate labels for ponds in a list (with each row being a sublist)
        title_labels = []
        for row in range(1,n_rows_small+1):
            title_labels.append([])
            for col in range(1,n_cols_small+1):
                if row < 10:
                    title_labels[row-1].append(f'0{row}0{col}')  
                else:
                    title_labels[row-1].append(f'{row}0{col}')  
        for idx_row, row in enumerate(title_labels):
            if idx_row < n_rows_large:
                for col in [6,8]:
                    if idx_row+1 < 10:
                        row.append(f'0{idx_row+1}0{col}')
                    else:
                        row.append(f'{idx_row+1}0{col}')
            else: # append blanks for the two large columns with only 10 rows 
                [row.append(f'BLANK {idx_row+1}-{c}') for c in [6,8]]

        # Add 2 blank rows at end to make room for extra data aggregations to be listed        
        #[title_labels.append([f'BLANK {r}-{c}' for c in range(1,7)]) for r in range(13,15)]

        # flatten title_labels for indexing from plot gridspec plot generation
        flat_title_labels = [label for item in title_labels for label in item]

        # initialize latest_sample_date as a global variable to track the most recent date to print in the title
        latest_sample_date = datetime.min

        # Initialize main plot
        plt_width = 8.5
        plt_height = 11
        scale_factor = 2.5
        fig = plt.figure(figsize=(plt_width*scale_factor, plt_height*scale_factor))

        outer_plots = gridspec.GridSpec(len(title_labels), len(title_labels[0]), wspace=0.05, hspace=0.2)

        for idx_plot, pond_plot in enumerate(outer_plots):
            pond_name = flat_title_labels[idx_plot]
            if 'BLANK' not in pond_name:

                # plot each pond with a function to ensure that data for each is isolated within a local scope
                plot_each_pond(fig, pond_plot, pond_name)

            else: # for plotting subplots labeled 'BLANK' in 'title_labels' list:  (the four lower right subplots) and the BLANK rows
                ax = plt.Subplot(fig, pond_plot)
                ax.axis('off')
                if 'BLANK 11-6' in pond_name: # plot the color key in the first blank subplot
                    # legend_text = (
                    #                 r'$\bf{Color\/\/key\/\/(latest\/\/EPA\/\/reading)}$:' "\n"
                    #                 "0% - 1.99%:      Red\n"
                    #                 "2% - 3.49%:      Yellow\n"
                    #                 "3.5% and up:    Green\n"
                    #                 "No EPA data:     Grey"
                    #                )
                    # t = ax.text(0.1,0.8,legend_text,ha='left', va='top', fontsize = 'x-large',
                    #        bbox=dict(facecolor='xkcd:bluegrey', alpha=0.5), multialignment='left')


                    legend_data = [{'labels':{'Color key (latest measurement)': {'align': '1', 'weight': 'bold'}}},
                                   {'labels':{'EPA %': {'align': '1', 'weight': 'underline'}}},
                                   {'labels':{'0% - 1.99%'.center(50): {'align': '1'}}, 'fill_color': 'red'}, # use .center() to pad spacing for one row to make legend plot wider
                                   {'labels':{'2% - 3.49%': {'align': '1'}}, 'fill_color': 'yellow'},
                                   {'labels':{'3.5% and up': {'align': '1'}}, 'fill_color': 'mediumspringgreen'},
                                   {'labels':{'No EPA data': {'align': '1'}}, 'fill_color': 'lightgrey'}
                                  ]
                    x_align = {'1': [1, 'center']}
                                                                                                                                                                                                                        
                    self.plot_legend(fig, ax, legend_data, x_align, y_spacing=0.15, y_align=0.8)
                  
                if 'BLANK 12-6' in pond_name: # plot the legend in the first blank subplot    
                    pass
                elif 'BLANK 13-1' in pond_name:
                    pass
                elif 'BLANK 13-3' in pond_name:
                    pass            
                elif 'BLANK 14-1' in pond_name:
                    pass
                elif 'BLANK 14-3' in pond_name:
                    pass

                fig.add_subplot(ax) # add the subplot for the 'BLANK' entries

        # Add title to EPA plot
        fig.suptitle(f'{plot_title} - {self.select_date.strftime("%-m/%-d/%Y")}', fontweight='bold',fontsize=16, y=0.905, va='bottom')
        fig.text(0.75, 0.905, f'\nLatest sample date: {latest_sample_date.strftime("%-m/%-d/%y")}', transform=fig.transFigure, ha='center', va='bottom', fontsize='large')
            
        if self.save_output:
            out_filename = f'./output_files/{plot_title} {self.select_date.strftime("%Y-%m-%d")}.pdf'
            plt.savefig(out_filename, bbox_inches='tight')     
            return out_filename
