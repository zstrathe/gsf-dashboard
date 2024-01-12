from datetime import datetime
import functools
import pandas as pd
import warnings
import math
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse, Rectangle
from .utils import generate_multipage_pdf
from .db_utils import query_data_table_by_date_range

class PondsOverviewPlots:
    def __init__(self, select_date, source_db='gsf_data', run=True, save_output=True): 
        self.select_date = pd.to_datetime(select_date).normalize() # Normalize select_date to remove potential time data and prevent possible key errors when selecting date range from data
        self.save_output = save_output
        self.source_db = source_db
        if run:
            self.output_filenames = [] # initialize a list to collect output filenames
            [self.output_filenames.append(x) for x in (self.plot_scorecard(), self.plot_potential_harvests(), self.plot_epa())] 

    # function to plot outlined legend boxes, using a dict of labels/data, and a separate dict of x_alignments (see examples of dict structure in calls to this function)
    def plot_legend(self, fig, ax, legend_data, x_align, y_spacing, y_align=0.9):
        y_align = y_align # start upper y-coord for plotting legend
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
    
    def plot_scorecard(self, target_to_density=0.4, target_topoff_depth=13, harvest_density=0.5, harvest_epa=3.00, plot_title='Pond Health Overview'): 
        print('Plotting ponds overview...', flush=True)
    
        select_date = self.select_date
        
        def subplot_ax_format(subplot_ax):
            subplot_ax.spines['top'].set_visible(False)
            subplot_ax.spines['right'].set_visible(False)
            subplot_ax.spines['bottom'].set_visible(False)
            subplot_ax.spines['left'].set_visible(False)
            subplot_ax.get_xaxis().set_ticks([])
            subplot_ax.get_yaxis().set_ticks([])
            fig.add_subplot(subplot_ax)
        
        # function to plot each pond to ensure that data for each plot is kept within a local scope
        def plot_each_pond(fig, pond_plot, pond_name, select_date):
            inner_plot = gridspec.GridSpecFromSubplotSpec(5,3,subplot_spec=pond_plot, wspace=-0.01, hspace=-0.01) # each individual plot (from gridspec) is divided into subplots with 5 rows and 3 columns

            # get measurements data from db for pond
            pond_data = measurements_df[measurements_df['PondID'] == pond_name].iloc[0]
          
            # Check prior 5 days of data to see if pond is active/in-use
            pond_active = pond_data['active_status']
       
            # Get the last harvest date for each pond, done separately from other data queries due to needing full range of dates, and regardless of whether it's active
            # the date found is relative to the select_date parameter (so reports can always be generated relative to a specific date even for past dates)
            last_harvest_idx = harvest_idx_df.loc[harvest_idx_df['PondID'] == pond_name, 'Split Innoculum'].last_valid_index()
            if last_harvest_idx:
                pond_last_harvest_date = harvest_idx_df.iloc[last_harvest_idx]['Date']
                pond_last_harvest_str = pond_last_harvest_date.strftime('%-m-%-d-%y')
                pond_days_since_harvest = (select_date - pond_last_harvest_date).days
            else:
                pond_last_harvest_str = 'n/a'
            
            # Gather current data for pond if it's active
            if pond_active == True:
                # get most recent EPA value and measurement date
                epa_val, epa_date = epa_df[(epa_df['PondID'] == pond_name) & (epa_df['Date'] == self.select_date)].iloc[0].loc[['epa_val', 'measurement_date_actual']]
                if pd.isna(epa_val):
                     epa_val = None
                     epa_date = None

                # get indicators data for individual pond
                # fill n/a values with 0 so that comparison operators don't throw an error if vals are empty
                indicators_data = indicators_df.loc[indicators_df['PondID'] == pond_name].copy().fillna(0).iloc[0]
               
                # key for pests/indicators, each value a list with comparison operator to use, the threshold for flagging 
                # and the color to display it as
                indicator_dict = {'% Nanno': ['less', 80, '%N', 'xkcd:fire engine red'],
                                 'Handheld pH': ['out-of-range', [7.8, 8.2], 'pH', 'xkcd:fire engine red'],
                                 'Rotifers': ['greater-equal', 1, 'R', 'xkcd:deep sky blue'],   
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
                status_color_dict = {1: 'lightgrey', 2: 'red', 3: 'tan', 4: 'yellow', 5: 'mediumspringgreen', 6: 'tab:green'}
                fill_color = status_color_dict[pond_data['status_code']]
              
            # title subplot
            title_ax = plt.Subplot(fig, inner_plot[:2,:])

            # plot pond EPA data
            try:
                if epa_val == None:
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
                pond_data_depth = pond_data['Depth']
                if pd.isna(pond_data_depth) or pond_data_depth == 0:
                    pond_data_depth = 'No data'
                else:    
                    pond_data_depth = f'{"**" if pond_data["non_current_data_flag_depth"] == True else ""}{pond_data_depth}"' # print two asterisks before if data was flagged as 'non-current'

                pond_data_afdw = pond_data['Filter AFDW']
                if pd.isna(pond_data_afdw) or pond_data_afdw == 0:
                    pond_data_afdw = 'No data'
                else:
                    pond_data_afdw = f'{"**" if pond_data["non_current_data_flag_afdw"] == True else ""}{round(pond_data_afdw,3)}' # print two asterisks before if data was flagged as 'non-current'

                pond_data_harvestable_mass = pond_data['harvestable_mass_nanno_corrected']
                if pd.isna(pond_data_harvestable_mass) or pond_data_harvestable_mass == 0:
                    pond_data_harvestable_mass = '-'
                else:
                    pond_data_harvestable_mass = f'{pond_data_harvestable_mass:,.0f} kg' 

                pond_data_harvestable_depth = pond_data['harvestable_depth_inches']
                if pd.isna(pond_data_harvestable_depth) or pond_data_harvestable_depth == 0:
                    pond_data_harvestable_depth = '-'
                else:
                    pond_data_harvestable_depth = f'{pond_data_harvestable_depth}"'
                
                running_avg_growth_5d = pond_data['running_avg_norm_growth_5d']
                if pd.isna(running_avg_growth_5d):
                    running_avg_growth_5d = 'n/a'
                else: 
                    running_avg_growth_5d = f'{running_avg_growth_5d:.3f} g/L/d'
                
                running_avg_growth_14d = pond_data['running_avg_norm_growth_14d']
                if pd.isna(running_avg_growth_14d):
                    running_avg_growth_14d = 'n/a'
                else: 
                    running_avg_growth_14d = f'{running_avg_growth_14d:.3f} g/L/d'
                     
                data_plot_dict = {'Measurement': {'Depth:': pond_data_depth, 'AFDW:': pond_data_afdw},
                                  'Growth': {'5 Days:': running_avg_growth_5d,  '14 Days:': running_avg_growth_14d},
                                  'Harvestable': {'Mass:': pond_data_harvestable_mass, 'Inches:': pond_data_harvestable_depth}
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
                        # if pond_data_error == True:
                        #     t = ax.text(0.5, 0.9, 'Data Error', ha='center', va='top')
                        # else:
                        t = ax.text(0.5, 0.9, 'Inactive', ha='center', va='top')
                    ax.set_facecolor('snow')
                subplot_ax_format(ax)
            
        ##############################
        ### START OF PLOTTING CODE ###
        ##############################

        # query epa data from db table
        epa_df = query_data_table_by_date_range(db_name_or_engine=self.source_db, 
                                                table_name='epa_data', 
                                                query_date_start=self.select_date, 
                                                query_date_end=self.select_date, 
                                                col_names=['epa_val', 'measurement_date_actual'])

        # query pest/health indicators data
        indicators_df = query_data_table_by_date_range(db_name_or_engine=self.source_db, 
                                                       table_name='ponds_data', 
                                                       query_date_start=self.select_date, 
                                                       query_date_end=self.select_date, 
                                                       col_names=['% Nanno', 'Handheld pH', 'Rotifers', 'Attached FD111', 'Free Floating FD111', 'Golden Flagellates', 'Diatoms', 'Tetra', 'Green Algae']) 

        # query measurements data
        measurements_df = query_data_table_by_date_range(db_name_or_engine=self.source_db, 
                                                         table_name='ponds_data', 
                                                         query_date_start=self.select_date-pd.Timedelta(days=1), 
                                                         query_date_end=self.select_date, 
                                                         col_names=['Depth', 'Filter AFDW', 'Column']) 
        _tmp_calculated_query_df = query_data_table_by_date_range(db_name_or_engine=self.source_db, 
                                                                  table_name='ponds_data_calculated', 
                                                                  query_date_start=self.select_date-pd.Timedelta(days=1), 
                                                                  query_date_end=self.select_date, 
                                                                  col_names=['active_status', 'status_code', 'calc_mass_nanno_corrected', 'harvestable_depth_inches', 'harvestable_mass_nanno_corrected', 'running_avg_norm_growth_5d', 'running_avg_norm_growth_14d'])
        measurements_df = pd.merge(measurements_df, _tmp_calculated_query_df, how='outer', on=['Date', 'PondID'])
        
        # if either Depth or AFDW values are missing for current day, check prev day
        # and if prev_day data is used instead, set non_current_data_flag to True for adding "**" to the value, and an explanation footnote to report
        for pond_id in measurements_df['PondID'].unique():
            curr_date_mask = (measurements_df['PondID'] == pond_id) & (measurements_df['Date'] == self.select_date)
            prev_date_mask = (measurements_df['PondID'] == pond_id) & (measurements_df['Date'] == self.select_date-pd.Timedelta(days=1))
            active_status = measurements_df.loc[curr_date_mask]['active_status'].iloc[0]
            if active_status == True:
                curr_depth, curr_afdw = measurements_df.loc[curr_date_mask][['Depth', 'Filter AFDW']].iloc[0].values
                
                # if current date "Depth" value is missing or zero, then look at previous day value, and use it if available
                # set 'non_current_data_flag" to True if using non-current data
                if pd.isna(curr_depth) or curr_depth <= 0:
                    prev_depth = measurements_df.loc[prev_date_mask]['Depth'].iloc[0]
                    if not pd.isna(prev_depth) or prev_depth > 0:
                        measurements_df.loc[curr_date_mask, 'Depth'] = prev_depth
                        measurements_df.loc[curr_date_mask, 'non_current_data_flag_depth'] = True
                    else:
                        measurements_df.loc[curr_date_mask, 'non_current_data_flag_depth'] = False
                else:
                    measurements_df.loc[curr_date_mask, 'non_current_data_flag_depth'] = False

                # if current date "Filter AFDW" value is missing or zero, then look at previous day value, and use it if available
                # set 'non_current_data_flag" to True if using non-current data
                if pd.isna(curr_afdw) or curr_afdw <= 0:
                    prev_afdw = measurements_df.loc[prev_date_mask]['Filter AFDW'].iloc[0]
                    if not pd.isna(prev_afdw) or (not pd.isna(prev_afdw) and prev_afdw > 0):
                        measurements_df.loc[curr_date_mask, 'Filter AFDW'] = prev_afdw
                        measurements_df.loc[curr_date_mask, 'non_current_data_flag_afdw'] = True
                    else:
                        measurements_df.loc[curr_date_mask, 'non_current_data_flag_afdw'] = False
                else:
                    measurements_df.loc[curr_date_mask, 'non_current_data_flag_afdw'] = False

                # check if both current values are now valid, otherwise if either is still invalid, then set them back to None and they will be noted as 'missing data' on report
                curr_depth, curr_afdw = measurements_df.loc[curr_date_mask][['Depth', 'Filter AFDW']].iloc[0].values
                if (pd.isna(curr_depth) or curr_depth <= 0) or (pd.isna(curr_afdw) or curr_afdw <= 0):
                    measurements_df.loc[curr_date_mask, ['Depth', 'Filter AFDW']] = None
                    measurements_df.loc[curr_date_mask, ['non_current_data_flag_depth', 'non_current_data_flag_afdw']] = False
     
        # filter to only the current day data
        measurements_df = measurements_df[measurements_df['Date'] == self.select_date]

        # check if any non-current data for current date of report
        if measurements_df[(measurements_df['non_current_data_flag_depth'] == True) | (measurements_df['non_current_data_flag_afdw'] == True)].shape[0] > 0:
            any_noncurrent_flag = True
        else:
            any_noncurrent_flag = False
                    
        # query df just for using to determine on what date a pond was last harvested, if ever
        # look back 90 days for data
        harvest_idx_df = query_data_table_by_date_range(db_name_or_engine=self.source_db, 
                                                        table_name='ponds_data', 
                                                        query_date_start=self.select_date-pd.Timedelta(days=90), 
                                                        query_date_end=self.select_date, 
                                                        col_names=['Split Innoculum']) 
        
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
                    status_code_mass_df = measurements_df[['Date', 'PondID', 'status_code', 'calc_mass_nanno_corrected']].copy()
                 
                    total_mass_by_code = status_code_mass_df.groupby(by='status_code').agg(pond_count=('status_code', 'count'), calc_mass_nanno_corrected=('calc_mass_nanno_corrected', 'sum')).reset_index()
                  
                    for code in [0,1,2,3,4,5,6]:
                        if code not in total_mass_by_code['status_code'].values:
                            total_mass_by_code = total_mass_by_code.append({'status_code': code, 'pond_count': 0, 'calc_mass_nanno_corrected': 0}, ignore_index=True)
                   
                    total_mass_by_code = total_mass_by_code.set_index('status_code').to_dict(orient='index')
                    
                    for code, vals_dict in total_mass_by_code.copy().items():
                        pond_count, calc_mass = vals_dict.values()
                        total_mass_by_code[code]['pond_count'] = pond_count if pond_count != None and pond_count > 0 else "-"
                        total_mass_by_code[code]['calc_mass_nanno_corrected'] = f'{calc_mass:,.0f} kg' if calc_mass != None and calc_mass > 0 else "-"
                    
                    '''
                    status codes: from DB
                        - 0: inactive pond 
                        - 1: grey: incomplete data (either 'afdw' or 'epa_val' are missing)
                        - 2: red: afdw less than 0.25
                        - 3: brown: afdw >= 0.25; epa_val < 2.5%
                        - 4: yellow: (afdw >= 0.25 and afdw < 0.50) OR (epa_val >=2.5% and epa_val < 3%)
                        - 5: light green: afdw >= 0.50 and < 0.80; epa_val > 3%
                        - 6: dark green: afdw >= 0.80; epa_val > 3%
                        - ERROR: should not happen, possible bug in code / missing conditions / etc

                    '''
                    
                    legend_data = [{'labels':
                                        {'Color key': {'align': '1-l', 'weight': 'bold'}}},
                                   {'labels':
                                        {'AFDW': {'align': '1-c', 'weight': 'underline'}, 
                                         'EPA %': {'align': '3-c', 'weight': 'underline'}, 
                                         'Count of Ponds': {'align': '4-c', 'weight': 'underline'},
                                         'Calc Nanno Mass': {'align': '5-c', 'weight': 'underline'}}},
                                   {'labels':
                                        {r'$<$ 0.25': {'align': '1-r'}, 
                                         'and': {'align': '2'}, 
                                         'any': {'align': '3-r'},
                                         total_mass_by_code.get(2).get('pond_count'): {'align': '4-c', 'excl_color': 'Y'},
                                         total_mass_by_code.get(2).get("calc_mass_nanno_corrected"): {'align': '5-c', 'excl_color': 'Y'}}, 
                                    'fill_color': 'red'},
                                   {'labels':
                                        {r'$\geq$ 0.25': {'align': '1-r'}, 
                                         'and': {'align': '2'}, 
                                         r'$<$ 2.5%': {'align': '3-r'}, 
                                         #'out of spec EPA but AFDW OK': {'align': '4-l', 'excl_color':'Y'}
                                         total_mass_by_code.get(3).get('pond_count'): {'align': '4-c', 'excl_color': 'Y'},
                                         total_mass_by_code.get(3).get("calc_mass_nanno_corrected"): {'align': '5-c', 'excl_color': 'Y'}}, 
                                    'fill_color': 'tan'},
                                   {'labels':
                                        {'0.25 - 0.49': {'align': '1-r'}, 
                                         'or': {'align': '2'}, 
                                         '2.5% - 2.99%': {'align': '3-r'},
                                         total_mass_by_code.get(4).get('pond_count'): {'align': '4-c', 'excl_color': 'Y'},
                                         total_mass_by_code.get(4).get("calc_mass_nanno_corrected"): {'align': '5-c', 'excl_color': 'Y'}}, 
                                    'fill_color': 'yellow'},
                                   {'labels':
                                        {'0.50 - 0.79': {'align': '1-r'}, 
                                         'and': {'align': '2'}, 
                                         r'$\geq$ 3.0%': {'align': '3-r'},
                                         total_mass_by_code.get(5).get('pond_count'): {'align': '4-c', 'excl_color': 'Y'},
                                         total_mass_by_code.get(5).get("calc_mass_nanno_corrected"): {'align': '5-c', 'excl_color': 'Y'}}, 
                                    'fill_color': 'mediumspringgreen'},
                                   {'labels':
                                        {r'$\geq$ 0.80': {'align': '1-r'}, 
                                         'and': {'align': '2'}, 
                                         r'$\geq$ 3.0%': {'align': '3-r'},
                                         total_mass_by_code.get(6).get('pond_count'): {'align': '4-c', 'excl_color': 'Y'},
                                         total_mass_by_code.get(6).get("calc_mass_nanno_corrected"): {'align': '5-c', 'excl_color': 'Y'}}, 
                                    'fill_color': 'tab:green'},
                                   {'labels':
                                        {'Incomplete data': {'align': '2'},
                                         total_mass_by_code.get(1).get('pond_count'): {'align': '4-c', 'excl_color': 'Y'},
                                         total_mass_by_code.get(1).get("calc_mass_nanno_corrected"): {'align': '5-c', 'excl_color': 'Y'}},
                                    'fill_color': 'lightgrey'}
                                  ]
                    x_align = {'1-l': [0.1, 'left'], 
                               '1-c': [0.23, 'center'],
                               '1-r': [0.35, 'right'],
                               '2': [0.48, 'center'],
                               '3-c': [0.75, 'center'],
                               '3-r': [0.9, 'right'],
                               '4-l': [0.94, 'left'],
                               '4-c': [1.13, 'center'],
                               '5-c': [1.55, 'center']}
                                                                                                                                                                                                                        
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

                    total_active_ponds = measurements_df[measurements_df["active_status"] == True].shape[0]
                    num_active_1_acre = measurements_df[(measurements_df["Column"].isin(['1', '2', '3', '4'])) & (measurements_df["active_status"] == True)].shape[0]
                    num_active_2_acre = measurements_df[(measurements_df["Column"].isin(['6', '8'])) & (measurements_df["active_status"] == True)].shape[0]
                    total_active_acres = (num_active_1_acre * 1.1) + (num_active_2_acre * 2.2)

                    # forward fill mass for missing days on active ponds
                    # first, add a string placeholder to stop filling on inactive days (to prevent carrying value over if a pond was re-started)
                    total_calc_mass = query_data_table_by_date_range(db_name_or_engine=self.source_db, 
                                                                  table_name='ponds_data_calculated', 
                                                                  query_date_start=self.select_date-pd.Timedelta(days=3), 
                                                                  query_date_end=self.select_date, 
                                                                  col_names=['active_status', 'calc_mass_nanno_corrected'])
                    total_calc_mass['calc_mass_nanno_corrected'] = total_calc_mass.apply(lambda row: '_tmp_placeholder_for_ffill' if row['active_status'] == False else row['calc_mass_nanno_corrected'], axis=1)
                    for pond_id in total_calc_mass['PondID'].unique():
                        mask = total_calc_mass['PondID'] == pond_id
                        total_calc_mass.loc[mask, 'calc_mass_nanno_corrected'] = total_calc_mass.loc[mask, 'calc_mass_nanno_corrected'].ffill(limit=2)
                    total_calc_mass['calc_mass_nanno_corrected'] = total_calc_mass['calc_mass_nanno_corrected'].replace('_tmp_placeholder_for_ffill', None)
                    total_calc_mass = total_calc_mass.loc[(total_calc_mass["active_status"] == True) & (total_calc_mass['Date'] == self.select_date), "calc_mass_nanno_corrected"].sum()
                    
                    print_string = (r'$\bf{Total\/\/Active\/\/ponds: }$' + f'{total_active_ponds}\n' 
                                    + r'$\bf{Active\/\/1.1\/\/acre\/\/ponds: }$' + f'{num_active_1_acre}\n'
                                    + r'$\bf{Active\/\/2.2\/\/acre\/\/ponds: }$' + f'{num_active_2_acre}\n'
                                    + r'$\bf{Total\/\/Active\/\/Acres: }$' + f'{total_active_acres:,.0f}\n'
                                   + r'$\bf{Calculated\/\/total\/\/mass:}$' + f'{total_calc_mass:,.0f} kg')
                    t = ax.text(0,0.75, print_string, ha='left', va='top', fontsize=16, multialignment='left')        

                elif 'BLANK 13-3' in pond_name:
                    ''' 
                    Plot growth rate info
                    '''
                    growth_data = query_data_table_by_date_range(db_name_or_engine=self.source_db, 
                                                         table_name='ponds_data_aggregate', 
                                                         query_date_start=self.select_date, 
                                                         query_date_end=self.select_date, 
                                                         col_names=['agg_running_avg_norm_growth_5d', 'agg_running_avg_norm_growth_14d']).iloc[0]
                    run_avg_growth_5d = f'{growth_data["agg_running_avg_norm_growth_5d"]:.3f}' if not pd.isna(growth_data["agg_running_avg_norm_growth_5d"]) else 'n/a'
                    run_avg_growth_14d = f'{growth_data["agg_running_avg_norm_growth_14d"]:.3f}' if not pd.isna(growth_data["agg_running_avg_norm_growth_14d"]) else 'n/a'
                    
                    print_string = 'Total Growth (g/L/day normalized)'
                    growth_title = ax.text(0,.75, print_string, ha='left', va='top', fontsize=16, multialignment='left')
                    bb = growth_title.get_window_extent(renderer=fig.canvas.get_renderer()).transformed(ax.transAxes.inverted())
                    ax.annotate('', xy=(bb.x0-0.01,bb.y0), xytext=(bb.x1+0.01,bb.y0), xycoords="axes fraction", arrowprops=dict(arrowstyle="-", color='k'))
                    
                    print_string += ('\n'
                                    + r'$\bf{Total\/\/5\/\/Day\/\/Running\/\/Avg\/\/Growth: }$' + run_avg_growth_5d + '\n' 
                                    + r'$\bf{Total\/\/14\/\/Day\/\/Running\/\/Avg\/\/Growth: }$' + run_avg_growth_14d + '\n')
            
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

                    #prev_date_data = self.processing_dataframe.loc[prev_date]
                    prev_date_processing_data = query_data_table_by_date_range(db_name_or_engine=self.source_db, 
                                                         table_name='daily_processing_data', 
                                                         query_date_start=self.select_date-pd.Timedelta(days=1), 
                                                         query_date_end=self.select_date-pd.Timedelta(days=1)).iloc[0] 

                    for label, subdict in processing_columns.items():
                        if type(subdict['column_name']) == list: # when 'column_name' is a list, get the first item with data (so first item in list is primary source)
                            for data_col_name in subdict['column_name']:
                                try: # force the data item to the 'data_format' type, with try/except to catch errors
                                    data_i = prev_date_processing_data[data_col_name]
                                    if (subdict['data_format'] == str and (data_i != '' or data_i != 'nan')) or (subdict['data_format'] != str and not pd.isna(data_i) and data_i > 0):
                                        prev_day_val = data_i
                                        break # if valid data is found, then break and stop evaluating any more columns
                                    else:
                                        prev_day_val = None
                                except: 
                                    prev_day_val = None
                        else: 
                            try:
                                prev_day_val = prev_date_processing_data[subdict["column_name"]]
                                if (subdict['data_format'] == str and (prev_day_val == '' or prev_day_val == 'nan')) or (subdict['data_format'] != str and prev_day_value == 0):
                                    prev_day_val = None
                            except:
                                prev_day_val = None
    
                        if pd.isna(prev_day_val):
                            if subdict['data_format'] == str:
                                prev_day_val = ''
                            else:
                                prev_day_val = 0

                        prev_day_val = subdict['data_format'](prev_day_val)
                            
                        if type(prev_day_val) == str:
                            prev_day_val = prev_day_val.replace("/ ", "\n  ").replace(". ", ".\n  ")
                            processing_data_str += f'\n{label}\n  {prev_day_val}'
                        elif type(prev_day_val) != str:
                            prev_day_val = f'{subdict["data_format"](prev_day_val):,} {subdict["str_label"]}'
                            processing_data_str += f'\n{label} {prev_day_val}'
                            
                    proc_t.update({'text': processing_data_str})
                
                fig.add_subplot(ax) # add the subplot for the 'BLANK' entries

        if self.save_output:
            out_filename = f'./output_files/{plot_title} {select_date.strftime("%Y-%m-%d")}.pdf'
            plt.savefig(out_filename, bbox_inches='tight')
            return out_filename
    
    def plot_potential_harvests(self, min_afdw=0.50, min_epa=3.0, overall_spacing=0.03): 

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
                title_text1 = f'Recommended Harvests - {str(select_date.strftime("%-m/%-d/%y"))}' 
                title_text2 = ('\nponds with' + r'$\geq$' + f'{min_afdw:.2f} AFDW and' + r'$\geq$' + f'{min_epa:.0f}% EPA, with estimated harvest depth to reach 0.40 AFDW after top off to 13"\n\n' + 
                              f'Total recommended harvest mass: {total_harvestable_mass:,.0f} kg\n' +
                              f'Total recommended harvest volume: {total_harvestable_gallons:,.0f} gal')
                t1 = ax.text(0.5, 1, title_text1, ha='center', va='top', fontsize=14, weight='bold')
                t2 = ax.text(0.5,0.992, title_text2, ha='center', va='top', fontsize=8)  
                # get the y-coordinate where tables can begin plotting on figure (after title text, or at top of ax if title text isn't present: i.e., page 2+)
                y0 = t2.get_window_extent(renderer=fig.canvas.get_renderer()).transformed(ax.transAxes.inverted()).y0 - overall_spacing
            else:
                y0 = 1
            return fig, ax, y0

        def plot_table(table_title_id, ax, y_start, data_fontsize=6.4, title_fontsize=8):
            table_title = ax.text(0.5, y_start, f'Column 0{table_title_id}', ha='center', va='top', fontsize=title_fontsize, weight='bold')
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

        select_date = self.select_date

        # query potential harvest data and pre-process (get running sums by 'column')
        df1 = potential_harvests_df = query_data_table_by_date_range(db_name_or_engine=self.source_db, 
                                                         table_name='ponds_data', 
                                                         query_date_start=self.select_date, 
                                                         query_date_end=self.select_date,
                                                         col_names=['Depth', 'Column', 'Filter AFDW'])
        df2 = query_data_table_by_date_range(db_name_or_engine=self.source_db, 
                                                         table_name='ponds_data_calculated', 
                                                         query_date_start=self.select_date, 
                                                         query_date_end=self.select_date,
                                                         col_names=['harvestable_depth_inches', 'harvestable_gallons', 'harvestable_mass_nanno_corrected', 'days_since_harvested_split'])
        df3 = query_data_table_by_date_range(db_name_or_engine=self.source_db, 
                                                         table_name='epa_data', 
                                                         query_date_start=self.select_date, 
                                                         query_date_end=self.select_date,
                                                         col_names=['epa_val'])
        # Combine dfs
        potential_harvests_df = functools.reduce(lambda df1, df2: pd.merge(df1, df2, on=['Date','PondID'], how='outer'), [df1, df2, df3])

        # filter by min_afdw and min_epa
        potential_harvests_df = potential_harvests_df[(potential_harvests_df['Filter AFDW'] >= min_afdw) & (potential_harvests_df['epa_val'] >= min_epa)]
        
        # drop empty rows (rows with no 'harvestable_mass')
        potential_harvests_df = potential_harvests_df[(potential_harvests_df['harvestable_mass_nanno_corrected'] != 0) & (~pd.isna(potential_harvests_df['harvestable_mass_nanno_corrected']))]
        
        # calculate "Drop To" (current pond level less the "harvestable_depth_inches")
        potential_harvests_df['Drop To'] = potential_harvests_df['Depth'] - potential_harvests_df['harvestable_depth_inches']

        # sort by Column id (ascending) then by 'days since harvested' (desc) then by harvestable mass (desc)
        potential_harvests_df = potential_harvests_df.sort_values(by=['Column', 'days_since_harvested_split', 'harvestable_mass_nanno_corrected'], ascending=[True, False, False])

        # compute running sums for each column
        h_cols = potential_harvests_df['Column'].unique()
        # if there is data to calculate, then compute running totals
        if len(h_cols) > 0:
            for col_id in h_cols:
                mask = potential_harvests_df['Column'] == col_id
                potential_harvests_df.loc[mask, 'Running Total Gallons'] = potential_harvests_df.loc[mask, 'harvestable_gallons'].cumsum()
                potential_harvests_df.loc[mask, 'Running Total Mass'] = potential_harvests_df.loc[mask, 'harvestable_mass_nanno_corrected'].cumsum()                                                   
        # if no data is available to calculate, add running total columns as None for everything
        else:
            potential_harvests_df['Running Total Gallons'] = None
            potential_harvests_df['Running Total Mass'] = None
            
        # rename columns for table display
        potential_harvests_df = potential_harvests_df.rename(columns={'PondID': 'Pond ID', 
                                                                      'harvestable_depth_inches': 'Harvestable Inches', 
                                                                      'harvestable_mass_nanno_corrected': 'Harvestable Mass', 
                                                                      'harvestable_gallons': 'Harvestable Gallons',
                                                                      'days_since_harvested_split': 'Days Since Harvested/Split'})

        # re-order columns for table display
        potential_harvests_df = potential_harvests_df[['Column', 'Pond ID', 'Drop To', 'Days Since Harvested/Split', 'Harvestable Inches', 'Harvestable Gallons', 'Harvestable Mass', 'Running Total Gallons', 'Running Total Mass']]

        # calc aggregates
        total_harvestable_mass = potential_harvests_df['Harvestable Mass'].sum()
        total_harvestable_gallons = potential_harvests_df['Harvestable Gallons'].sum()
        
        # format column data for table display
        potential_harvests_df['Drop To'] = potential_harvests_df['Drop To'].apply(lambda x: f'{x}"')
        potential_harvests_df['Days Since Harvested/Split'] = potential_harvests_df['Days Since Harvested/Split'].apply(lambda x: f'{x} days') 
        potential_harvests_df['Harvestable Inches'] = potential_harvests_df["Harvestable Inches"].apply(lambda x: f'{x}"')
        potential_harvests_df['Harvestable Gallons'] = potential_harvests_df["Harvestable Gallons"].apply(lambda x: f'{x:,.0f} gal')
        potential_harvests_df['Harvestable Mass'] = potential_harvests_df['Harvestable Mass'].apply(lambda x: f'{x:,.0f} kg')
        potential_harvests_df['Running Total Gallons'] = potential_harvests_df["Running Total Gallons"].apply(lambda x: f'{x:,.0f} gal')
        potential_harvests_df['Running Total Mass'] = potential_harvests_df["Running Total Mass"].apply(lambda x: f'{x:,.0f} kg')
        
        fig_list = [] # initialize list of figs (one for each output page, as necessary)
        tables_list = list(potential_harvests_df['Column'].unique()) # init list of tables to generate (one for each Column (1,2,3,4,6,8) with valid data (at least one pond with harvestable mass)

        fig, ax, y_align = gen_fig(title=True)

        table_spacing = 0.035

        while tables_list:
            column_id = tables_list.pop(0) # set 'table_title' equal to the key column identifier ('1', '2', '3', '4', '6', or '8')
            # get dataframe for table (data for each Column id)
            df = potential_harvests_df[potential_harvests_df['Column'] == column_id].drop('Column', axis=1)

            # call plot_table function, which returns the next y_coordinate for plotting a table to
            # once the y_coordinates for a table are outside of the ax bound (< 0) then the function will return the string 'start_new_page'
            # in that case, append the current figure to fig_list, then generate a new figure and re-plot the table on the new page
            y_align = plot_table(table_title_id=column_id, ax=ax, y_start=y_align)
            if y_align == 'start_new_page':
                #plt.show() # show plot for testing in jupyter nb
                fig_list.append(fig)
                fig, ax, y_align = gen_fig()
                y_align = plot_table(table_title_id=column_id, ax=ax, y_start=y_align)
            y_align -= overall_spacing
        #plt.show() # show plot for testing in jupyter nb
        fig_list.append(fig)     

        filename = f'./output_files/Potential Harvests {select_date.strftime("%Y-%m-%d")}.pdf'
        out_filename = generate_multipage_pdf(fig_list, filename, add_pagenum=True, bbox_inches=None)
        return out_filename

    def plot_epa(self, plot_title='Pond EPA Overview'):
        def _load_epa_dict():
            epa_df = query_data_table_by_date_range(db_name_or_engine=self.source_db, 
                                                    table_name='epa_data', 
                                                    query_date_start=self.select_date-pd.Timedelta(days=90), 
                                                    query_date_end=self.select_date, 
                                                    col_names=['epa_val', 'epa_actual_measurement'], 
                                                    check_safe_date=True)
            epa_dict = {}
            for pond_id in epa_df['PondID'].unique():
                pond_df = epa_df[(epa_df['PondID'] == pond_id)].copy() # & (~pd.isna(epa_df['epa_actual_measurement']))
                pond_epa_dict = {}
                # iterate through rows (from newest to oldest), and if any Null values are detected, stop iterating
                # because when epa_val == None, indicates that pond was inactive
                for row in pond_df[::-1].iterrows():
                    if len(pond_epa_dict) >= 3: # limit to displaying only last 3 values for epa measurements
                        break
                    row_data = row[1]
                    if pd.isna(row_data['epa_val']):
                        break
                    if row_data['epa_actual_measurement'] == True:
                        pond_epa_dict[row_data['Date'].to_pydatetime()] = row_data['epa_val']
                epa_dict[pond_id] = pond_epa_dict
            return epa_dict
        
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

            # get pond active status (True or False) 
            pond_active = _active_status_dict[pond_name]

            # get data for individual pond only, as a list
            # each list entry should consist of sublists containing the date and the epa value, sorted in descending order by date (last 3 values only for active ponds)
            single_pond_data = list(_epa_dict[pond_name].items())

            # check and update the latest_sample_date global variable if this pond has a more recent date
            if len(single_pond_data) != 0:
                nonlocal latest_sample_date
                if single_pond_data[0][0] > latest_sample_date:
                    latest_sample_date = single_pond_data[0][0]

            # title subplot
            title_ax = plt.Subplot(fig, inner_plot[0,:])
            title_str = r'$\bf{' + pond_name + '}$' 
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
                                       (2.0,2.99999999): 'yellow',
                                       (3.0,99999999999): 'mediumspringgreen'}
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

        # load epa_dict
        _epa_dict = _load_epa_dict()

        # load active_status
        active_status_df = query_data_table_by_date_range(db_name_or_engine=self.source_db, 
                                                          table_name='ponds_data_calculated', 
                                                          query_date_start=self.select_date, 
                                                          query_date_end=self.select_date, 
                                                          col_names=['active_status'])
        _active_status_dict = {row['PondID']: row['active_status'] for row in active_status_df.to_dict(orient='records')}
        
        # init vars for generating grid
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
                    legend_data = [{'labels':{'Color key (latest measurement)': {'align': '1', 'weight': 'bold'}}},
                                   {'labels':{'EPA %': {'align': '1', 'weight': 'underline'}}},
                                   {'labels':{'0% - 1.99%'.center(50): {'align': '1'}}, 'fill_color': 'red'}, # use .center() to pad spacing for one row to make legend plot wider
                                   {'labels':{'2% - 2.99%': {'align': '1'}}, 'fill_color': 'yellow'},
                                   {'labels':{'3% and up': {'align': '1'}}, 'fill_color': 'mediumspringgreen'},
                                   {'labels':{'No EPA data': {'align': '1'}}, 'fill_color': 'lightgrey'}
                                  ]
                    x_align = {'1': [1, 'center']}
                                                                                                                                                                                                                        
                    self.plot_legend(fig, ax, legend_data, x_align, y_spacing=0.15, y_align=0.8)

                fig.add_subplot(ax) # add the subplot for the 'BLANK' entries

        # Add title to EPA plot
        fig.suptitle(f'{plot_title} - {self.select_date.strftime("%-m/%-d/%Y")}', fontweight='bold',fontsize=16, y=0.905, va='bottom')
        fig.text(0.75, 0.905, f'\nLatest sample date: {latest_sample_date.strftime("%-m/%-d/%y")}', transform=fig.transFigure, ha='center', va='bottom', fontsize='large')
            
        if self.save_output:
            out_filename = f'./output_files/{plot_title} {self.select_date.strftime("%Y-%m-%d")}.pdf'
            plt.savefig(out_filename, bbox_inches='tight')     
            return out_filename


    # def plot_epa_temp_new_(self, plot_title='Pond EPA Health\n(EPA/AFDW)'):
    #     def subplot_ax_format(subplot_ax, fill_color):
    #         subplot_ax.set_facecolor(fill_color)
    #         subplot_ax.spines['top'].set_visible(False)
    #         subplot_ax.spines['right'].set_visible(False)
    #         subplot_ax.spines['bottom'].set_visible(False)
    #         subplot_ax.spines['left'].set_visible(False)
    #         subplot_ax.get_xaxis().set_ticks([])
    #         subplot_ax.get_yaxis().set_ticks([])
    #         fig.add_subplot(subplot_ax)

    #     # function to plot each pond to ensure that data for each plot is kept within a local scope
    #     def plot_each_pond(fig, pond_plot, pond_name):
    #         inner_plot = gridspec.GridSpecFromSubplotSpec(5,3,subplot_spec=pond_plot, wspace=0, hspace=0)

    #         # get pond active status (True or False) from ponds_active_status dict
    #         pond_active = self.active_dict[pond_name]

    #         # get data for individual pond only, as a list
    #         # each list entry should consist of a tuple containing the date and the epa value, sorted in descending order by date
    #         single_pond_data = list(self.epa_data_dict[pond_name].items())
    #         print('testestts', single_pond_data)
    #         try:
    #             single_pond_afdw_data = self.scorecard_dataframe[pond_name]['AFDW (filter)'].replace(0,None).ffill()
    #         except:
    #             single_pond_afdw_data = None
    #             print('Error, no data for', pond_name)
                
    #         # iterate through epa data and get corresponding afdw (density) data 
    #         pond_epa_afdw_calcs = []
    #         for idx, (date, epa_val) in enumerate(self.epa_data_dict.get(pond_name).items()): 
    #             if type(single_pond_afdw_data) == pd.Series:
    #                 afdw_val = single_pond_afdw_data.loc[date]
    #             pond_epa_afdw_calcs.append((date, (epa_val / (afdw_val*100))*100))
    #         single_pond_data = pond_epa_afdw_calcs
    #         print('testtesttest', pond_name, pond_epa_afdw_calcs)
            
    #         # check and update the latest_sample_date global variable if this pond has a more recent date
    #         if len(single_pond_data) != 0:
    #             nonlocal latest_sample_date
    #             if single_pond_data[0][0] > latest_sample_date:
    #                 latest_sample_date = single_pond_data[0][0]

    #         # title subplot
    #         title_ax = plt.Subplot(fig, inner_plot[0,:])
    #         title_str = r'$\bf{' + pond_name + '}$' 
    #         # if len(ponds_data[pond_name]['source']) > 0:
    #         #     title_str += '\nsource: ' + ponds_data[pond_name]['source']
    #         title_ax.text(0.5, 0.1, title_str , ha = 'center', va='bottom', fontsize='large')
    #         subplot_ax_format(title_ax, 'white')

    #         # epa data subplots
    #         epa_ax = plt.Subplot(fig, inner_plot[1:3,:])

    #         if pond_active:
    #             # set fill color based on latest EPA reading
    #             if len(single_pond_data) > 0:
    #                 # set lastest_epa_data as the first value in single_pond_data (which should be reverse sorted)
    #                 latest_epa_data = single_pond_data[0][1] 
    #                 fill_color_dict = {(0,1.99999999):'red', 
    #                                    (2.0,3.49999999): 'yellow',
    #                                    (3.5,99999999999): 'mediumspringgreen'}
    #                 for idx, (key, val) in enumerate(fill_color_dict.items()):
    #                     if latest_epa_data >= key[0] and latest_epa_data < key[1]:
    #                         fill_color = val 
    #             else:
    #                 # set fill color as light grey if there is no EPA data for the pond
    #                 fill_color = 'lightgrey'

    #             if len(single_pond_data) > 0:
    #                 # set the center-point of each data point on the subplot, depending on number of data points available
    #                 if len(single_pond_data) == 1:
    #                     text_x = [0.5]
    #                 elif len(single_pond_data) == 2:
    #                     text_x = [0.3, 0.7]
    #                 else:
    #                     text_x = [0.195, 0.5, 0.805]

    #                 for idx, item in enumerate(single_pond_data):
    #                     text_date_formatted =  item[0].strftime("%-m/%-d/%y")
    #                     epa_ax.text(text_x[idx], 0.7, text_date_formatted, ha='center', va='center')
    #                     text_epa_data_formatted =  f'{item[1]: .2f}%'
    #                     epa_ax.text(text_x[idx], 0.3, text_epa_data_formatted, ha='center', va='center', fontsize='large', weight='bold')
    #             else:
    #                 epa_ax.text(0.5, 0.5, 'Active but no data', ha='center', va='center')
    #         else: # if pond is inactive
    #             fill_color = 'whitesmoke'
    #             epa_ax.text(0.5, 0.5, 'Inactive', ha='center', va='center')

    #         subplot_ax_format(epa_ax, fill_color)

    #         # epa change subplot
    #         epa_pct_ax = plt.Subplot(fig, inner_plot[3:,:])

    #         if pond_active:
    #         # Get the % change in EPA values if there is >1 entry (use the first and last indexes of the single_pond_data list)
    #             if len(single_pond_data) != 0:
    #                 epa_pct_fill = 'xkcd:light grey'    

    #                 def calc_epa_pct_chg(data_curr: list, data_prev: list):
    #                     '''
    #                     inputs:
    #                         data_curr: [datetime value, epa float value]
    #                         data_prev: [datetime value, epa float value]

    #                     output:
    #                         pct_chg: calculated absolute change in percentage rounded to 2 decimal places and formatted with % character (str)
    #                         delta_days: number of days between readings (int)
    #                         pct_format_color: color for displaying percentage (str)
    #                     '''
    #                     delta_days = (data_curr[0] - data_prev[0]).days    

    #                     pct_chg = data_curr[1] - data_prev[1]
    #                     if pct_chg > 0:
    #                         pct_format_color = 'xkcd:emerald green'
    #                         pct_chg = f'+{pct_chg:.2f}%'
    #                     elif pct_chg < 0: 
    #                         pct_format_color = 'xkcd:fire engine red'
    #                         pct_chg = f'{pct_chg:.2f}%'
    #                     else:
    #                         pct_format_color = 'black'
    #                         pct_chg = f'{pct_chg:.2f}%'
    #                     return [pct_chg, delta_days, pct_format_color]
    #                     # else:
    #                     #     return ['n/a', delta_days, 'black']

    #                 if len(single_pond_data) == 2:
    #                     epa_pct_chg, delta_days, epa_pct_color = calc_epa_pct_chg(single_pond_data[0], single_pond_data[1])
    #                     text_formatted1 =  f'Change ({delta_days} day{"s" if delta_days > 1 else ""}):' 
    #                     epa_pct_ax.text(0.5, 0.7, text_formatted1, ha='center', va='center', fontsize='large')
    #                     epa_pct_ax.text(0.5, 0.3, epa_pct_chg, ha='center', va='center', fontsize='large', color=epa_pct_color, weight='bold')
    #                 elif len(single_pond_data) == 3:
    #                     epa_pct_chg1, delta_days1, epa_pct_color1 = calc_epa_pct_chg(single_pond_data[0], single_pond_data[1])
    #                     epa_pct_chg2, delta_days2, epa_pct_color2 = calc_epa_pct_chg(single_pond_data[0], single_pond_data[2])
    #                     text_formatted1 =  f'Change ({delta_days1} day{"s" if delta_days1 > 1 else ""}, {delta_days2} days):'
    #                     epa_pct_ax.text(0.5, 0.7, text_formatted1, ha='center', va='center', fontsize='large')
    #                     epa_pct_ax.text(0.3, 0.3, epa_pct_chg1, ha='center', va='center', fontsize='large', color=epa_pct_color1, weight='bold')
    #                     epa_pct_ax.text(0.7, 0.3, epa_pct_chg2, ha='center', va='center', fontsize='large', color=epa_pct_color2, weight='bold')
    #                 else: # if there is only one data point so no percentage change
    #                     epa_pct_ax.text(0.5, 0.7, 'Change:', ha='center', va='center', fontsize='large')
    #                     epa_pct_ax.text(0.5, 0.3, 'n/a', ha='center', va='center', fontsize='large')

    #             else: # when there is no data for this pond
    #                 epa_pct_fill = 'lightgrey'

    #         else: # if pond is inactive
    #             epa_pct_fill = 'whitesmoke'

    #         subplot_ax_format(epa_pct_ax, epa_pct_fill) 

    #     ##############################
    #     ### START OF PLOTTING CODE ###
    #     ##############################

    #     n_rows_small = 12
    #     n_rows_large = 10
    #     n_cols_small = 4
    #     n_cols_large = 2

    #     # generate labels for ponds in a list (with each row being a sublist)
    #     title_labels = []
    #     for row in range(1,n_rows_small+1):
    #         title_labels.append([])
    #         for col in range(1,n_cols_small+1):
    #             if row < 10:
    #                 title_labels[row-1].append(f'0{row}0{col}')  
    #             else:
    #                 title_labels[row-1].append(f'{row}0{col}')  
    #     for idx_row, row in enumerate(title_labels):
    #         if idx_row < n_rows_large:
    #             for col in [6,8]:
    #                 if idx_row+1 < 10:
    #                     row.append(f'0{idx_row+1}0{col}')
    #                 else:
    #                     row.append(f'{idx_row+1}0{col}')
    #         else: # append blanks for the two large columns with only 10 rows 
    #             [row.append(f'BLANK {idx_row+1}-{c}') for c in [6,8]]

    #     # Add 2 blank rows at end to make room for extra data aggregations to be listed        
    #     #[title_labels.append([f'BLANK {r}-{c}' for c in range(1,7)]) for r in range(13,15)]

    #     # flatten title_labels for indexing from plot gridspec plot generation
    #     flat_title_labels = [label for item in title_labels for label in item]

    #     # initialize latest_sample_date as a global variable to track the most recent date to print in the title
    #     latest_sample_date = datetime.min

    #     # Initialize main plot
    #     plt_width = 8.5
    #     plt_height = 11
    #     scale_factor = 2.5
    #     fig = plt.figure(figsize=(plt_width*scale_factor, plt_height*scale_factor))

    #     outer_plots = gridspec.GridSpec(len(title_labels), len(title_labels[0]), wspace=0.05, hspace=0.2)

    #     for idx_plot, pond_plot in enumerate(outer_plots):
    #         pond_name = flat_title_labels[idx_plot]
    #         if 'BLANK' not in pond_name:

    #             # plot each pond with a function to ensure that data for each is isolated within a local scope
    #             plot_each_pond(fig, pond_plot, pond_name)

    #         else: # for plotting subplots labeled 'BLANK' in 'title_labels' list:  (the four lower right subplots) and the BLANK rows
    #             ax = plt.Subplot(fig, pond_plot)
    #             ax.axis('off')
    #             if 'BLANK 11-6' in pond_name: # plot the color key in the first blank subplot
    #                 # legend_text = (
    #                 #                 r'$\bf{Color\/\/key\/\/(latest\/\/EPA\/\/reading)}$:' "\n"
    #                 #                 "0% - 1.99%:      Red\n"
    #                 #                 "2% - 3.49%:      Yellow\n"
    #                 #                 "3.5% and up:    Green\n"
    #                 #                 "No EPA data:     Grey"
    #                 #                )
    #                 # t = ax.text(0.1,0.8,legend_text,ha='left', va='top', fontsize = 'x-large',
    #                 #        bbox=dict(facecolor='xkcd:bluegrey', alpha=0.5), multialignment='left')


    #                 legend_data = [{'labels':{'Color key (latest measurement)': {'align': '1', 'weight': 'bold'}}},
    #                                {'labels':{'EPA %': {'align': '1', 'weight': 'underline'}}},
    #                                {'labels':{'0% - 1.99%'.center(50): {'align': '1'}}, 'fill_color': 'red'}, # use .center() to pad spacing for one row to make legend plot wider
    #                                {'labels':{'2% - 3.49%': {'align': '1'}}, 'fill_color': 'yellow'},
    #                                {'labels':{'3.5% and up': {'align': '1'}}, 'fill_color': 'mediumspringgreen'},
    #                                {'labels':{'No EPA data': {'align': '1'}}, 'fill_color': 'lightgrey'}
    #                               ]
    #                 x_align = {'1': [1, 'center']}
                                                                                                                                                                                                                        
    #                 self.plot_legend(fig, ax, legend_data, x_align, y_spacing=0.15, y_align=0.8)
                  
    #             if 'BLANK 12-6' in pond_name: # plot the legend in the first blank subplot    
    #                 pass
    #             elif 'BLANK 13-1' in pond_name:
    #                 pass
    #             elif 'BLANK 13-3' in pond_name:
    #                 pass            
    #             elif 'BLANK 14-1' in pond_name:
    #                 pass
    #             elif 'BLANK 14-3' in pond_name:
    #                 pass

    #             fig.add_subplot(ax) # add the subplot for the 'BLANK' entries

    #     # Add title to EPA plot
    #     fig.suptitle(f'{plot_title} - {self.select_date.strftime("%-m/%-d/%Y")}', fontweight='bold',fontsize=16, y=0.905, va='bottom')
    #     fig.text(0.75, 0.905, f'\nLatest sample date: {latest_sample_date.strftime("%-m/%-d/%y")}', transform=fig.transFigure, ha='center', va='bottom', fontsize='large')
            
    #     if self.save_output:
    #         out_filename = f'./output_files/{plot_title} {self.select_date.strftime("%Y-%m-%d")}.pdf'
    #         plt.savefig(out_filename, bbox_inches='tight')     
    #         return out_filename

