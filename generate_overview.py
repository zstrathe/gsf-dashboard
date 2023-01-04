import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys
import re
import argparse

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--input_file", required=True, help="Excel (.xls or .xlsx) input file")
    parser.add_argument("-d","--date", required=True, help="Date to generate report for in 'yyyy-mm-dd' format; year must be from 2020 to 2023")
    parser.add_argument("-t", "--target_density", type=float, default=0.4, help="Optional target density of AFDW (Default = 0.40)")
    args = parser.parse_args()
    
    # check that file is an excel file
    file_extension = args.input_file.split(".")[-1]
    if not (file_extension == 'xlsx' or file_extension == 'xls'):
        print("ERROR: input filetype should be .xlsx or .xls")
        sys.exit(2)
    
    # check if date argument is valid
    date_error_flag = False
    date_pattern = re.compile(r'[0-9]{4}-[0-9]{2}-[0-9]{2}')
    if not re.fullmatch(date_pattern, args.date):
        date_error_flag = True
    else:    
        date_split = args.date.split("-")
        for idx, number in enumerate(date_split):
            # check that year is between 2020 and 2023
            if idx == 0:
                if int(number) < 2020 or int(number) > 2023:
                    date_error_flag = True
                    break
            # check that month value (as an integer) is from 1-12        
            elif idx == 1:
                if int(number) < 1 or int(number) > 12:
                    error_flag = True
                    break
            # check that date value (as an integer) is from 1-31
            elif idx == 2:
                if int(number) < 1 or int(number) > 31:
                    error_flag = True
                    break
        if date_error_flag == True:
            print("ERROR: invalid date (should be 'yyyy-mm-ddd' between 2020 and 2023)")
            sys.exit(2)
    
    # check if optional target_density argument is valid (between 0 and 1)
    if args.target_density:
        if not (args.target_density > 0 and args.target_density < 1):
            print("ERROR: target density (AFDW) should be between 0 and 1")
            sys.exit(2)
    
    print('Loading data...')
    overview = ponds_overview(excel_filename = args.input_file)
    print('Plotting data...')
    overview.plot(select_date=args.date, target_to_density=args.target_density, save_output=True) 
    sys.exit(0) # exit with status 0 to indicate successful execution
    
class ponds_overview:
    def __init__(self, excel_filename):
        self.ponds_data = self.load_data(excel_filename)
        
    def load_data(self, excel_filename):
        # Load the Daily Pond Scorecard excel file
        excel_sheets = pd.ExcelFile(excel_filename)

        # Get list of all sheet names
        sheetnames = sorted(excel_sheets.sheet_names)

        # initialize list of ponds corresponding to excel sheet names #### NOTE: '1201 ' has a trailing space that should be fixed in excel file
        ponds_list = ['0001', '0002', '0003', '0004', '0101', '0102', '0103', '0104', '0106', '0108', '0201', '0202', '0203', '0204', 
                      '0206', '0208', '0301', '0302', '0303', '0304', '0306', '0308', '0401', '0402', '0403', '0404', '0406', '0408', 
                      '0501', '0502', '0503', '0504', '0506', '0508', '0601', '0602', '0603', '0604', '0606', '0608', '0701', '0702', 
                      '0703', '0704', '0706', '0708', '0801', '0802', '0803', '0804', '0806', '0808', '0901', '0902', '0903', '0904', 
                      '0906', '0908', '1001', '1002', '1003', '1004', '1101', '1102', '1103', '1104', '1201 ', '1202', '1203', '1204']
        
        # create a dict containing a dataframe for each pond sheet
        all_ponds_data = {}
        for i in ponds_list:
            all_ponds_data[i] = excel_sheets.parse(i)

        # clean the pond data (convert data from string to datetime, set date as index, drop empty columns   
        updated_ponds = {} # dict for storing cleaned data
        for key in enumerate(all_ponds_data.keys()):
            df = all_ponds_data[key[1]] 
            df.iloc[:,0] = pd.to_datetime(df.iloc[:,0], errors='coerce') # convert date column from string
            df = df.set_index(df.iloc[:,0].name) # set date column as index
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')] # drop columns without a header, assumed to be empty or unimportant
            updated_ponds[key[1]] = df # add updated df to new dict of pond dataframes, which will later overwrite original 
        return updated_ponds # return the cleaned data dict

    def plot(self, select_date, target_to_density=0.4, target_topoff_depth=13, harvest_density=0.5, save_output=False, plot_title='Pond Health Overview'): 
        # Initialize variables for aggregations
        total_mass_all = 0 # total mass for entire farm (regardless of density)
        total_harvestable_mass = 0 # all available mass with afdw > target_to_density
        target_harvest_mass = 0 # all harvestable mass with resulting afdw > target_to_density but only ponds with current afdw > harvest_density
        target_harvest_gals = 0 # target_harvest_mass in terms of volume in gallons
        suggested_harvests = {} # dict to store calculated harvest depths for ponds with density >= harvest_density
        num_active_ponds = 0 # number of active ponds (defined by the plot_ponds().check_active() function)
        
        # helper function to check if a pond has data from prior n-days from the selected date 
        def check_active(pond_name, num_days_prior):
            def check_active_query(prev_days): 
                try:
                    dat = self.ponds_data[pond_name]['Fo'].shift(prev_days).loc[select_date]
                    if pd.isna(dat) or dat == 0:
                        return False
                    else:
                        return True
                except:
                    return False
            for i in range(1,num_days_prior+1):
                if check_active_query(i) == True:
                    nonlocal num_active_ponds
                    num_active_ponds += 1
                    return True
                elif i == num_days_prior and check_active_query(i) == False:
                    return False
        
        # helper function to query single data points from each pond dataframe
        def data_query(pond_data, pond_name, query_name):
            # Select the data to return for each query
            # convert to float to catch exception of an empty dataframe being returned (will be Inactive/Error from try/except statement that calls this function)
            return_data = None

            if query_name == 'afdw':
                return_data = pond_data['AFDW (filter)']
                if return_data == 0: # if 0, set as None to be flagged as "No data"
                    return_data = None
                else:
                    return_data = round(return_data, 3)

            elif query_name == 'depth':
                return_data = pond_data['Depth']
                if return_data == 0: # if 0, set as None to be flagged as "No data"
                    return_data = None

            elif query_name == 'harvestable_mass':
                data = pond_data[['AFDW (filter)', 'Depth']]
                curr_density = data[0]
                curr_depth = data[1]
                if curr_density != 0 and curr_depth != 0:
                    depth_to_liters = 35000 * 3.78541 # save conversion factor for depth (in inches) to liters
                    # for the 6 and 8 columns, double the depth to liters conversion because they are twice the size
                    if pond_name[-2:] == '06' or pond_name[-2:] == '08': 
                        depth_to_liters *= 2
                    
                    # calculate total mass in each pond and add to aggregation of all ponds
                    curr_pond_mass = curr_depth * depth_to_liters * curr_density
                    curr_pond_mass = int(curr_pond_mass/1000) # convert grams to kg and round by converting to int
                    nonlocal total_mass_all
                    total_mass_all += curr_pond_mass
                    
                    # calculate harvest depth to reach target density
                    harvest_depth = ((curr_depth * curr_density) - (target_topoff_depth * target_to_density)) / curr_density
                    harvestable_gals = harvest_depth * 35000 # calc harvestable gallons

                    #calculate harvestable mass as (harvest depth * depth_to_liters) * current density
                    harvestable_mass = (harvest_depth * depth_to_liters) * curr_density

                    # convert grams to kg and round by converting to int
                    harvestable_mass = int(harvestable_mass / 1000)

                    if harvestable_mass > 0:
                        return_data = harvestable_mass

                        # save harvest_depth in suggested_harvests dictionary ONLY IF curr_density >= harvest_density
                        if round(curr_density,3) >= harvest_density:
                            nonlocal suggested_harvests
                            nonlocal target_harvest_mass
                            nonlocal target_harvest_gals
                            suggested_harvests[pond_name] = round(harvest_depth,2)
                            target_harvest_mass += harvestable_mass
                            target_harvest_gals += harvestable_gals
                    else:
                        return_data = 0
                else:
                    return_data = None # None value which will be converted to "No data", because missing either density or depth data

            elif query_name == 'growth':
                return_data = None
          
            # Check if pond data is n/a
            if pd.isna(return_data):
                return "No data"
            else: 
                return return_data
        
        # helper function to get the fill color for each pond plot
        def get_fill_color(disp_data, color_dict): # helper function for getting the fill color for each pond subplot
            if disp_data == 'No data':
                return 'lightgrey'
            
            color_dict_all = {'pests_colors' : {(0,0.0000000001):'tab:green', (0,0.4999999999): 'tab:orange', (0.5,9999999): 'tab:red'},
                              'afdw_colors' : {(0,0.249999999):'tab:red', (0.25,0.499999999): 'yellow',
                                                        (0.50,0.799999999): 'mediumspringgreen', (0.80,999999999): 'tab:green'}
                             }
            color_dict = color_dict_all[color_dict]

            
            for idx, (key, val) in enumerate(color_dict.items()):
                if disp_data >= key[0] and disp_data < key[1]:
                    return val
        
        # helper function for formatting text of pond data
        def text_display_format(data_list): 
            # check whether data is a string, if not, then format with a ',' separator for thousands
            if type(data_list[1]) == str:
                return f'{data_list[0]}:\n{data_list[1]}'
            else:
                return f'{data_list[0]}:\n{data_list[1]:,}'
        
        def get_pests_format(pond_data, pond_name):
            pests_df = pond_data[['Rotifers ','Attached FD111','Free Floating FD111', 'Golden Flagellates', 'Diatoms', 
                    'Tetra','Green Algae']].fillna(0)
            # key for pests, each value a list with threshold for flagging (when >= threshold) and the color to format as
            pest_key = {'Rotifers ': [0.01,'blue'],   
                        'Attached FD111': [0.01,'red'],    
                        'Free Floating FD111': [0.01,'deeppink'],  
                        'Golden Flagellates': [0.01,'purple'], 
                        'Diatoms': [0.01,'cyan'],
                        'Tetra': [0.01,'orange'],
                        'Green Algae': [0.01 ,'green']} 
            colors_list = []
            for idx, (key, val) in enumerate(pest_key.items()):
                threshold = val[0]
                if pests_df[key] >= threshold:
                    colors_list.append(val[1])
            return colors_list
        
        def plot_each_pond(fig, pond_plot, pond_name, select_date):
            inner_plot = gridspec.GridSpecFromSubplotSpec(4,3,subplot_spec=pond_plot, wspace=0, hspace=0)

            # setup data dict as: query_string: [data_label, data_point]
            data_dict = {'afdw':['AFDW', None],
                         'depth':['Depth', None], 
                         'harvestable_mass':['Avail. to\nHarvest (kg)', None],
                         'growth':['Growth',None],
                         'pests':['Pests',None] 
                         }

            # Check prior 5 days of data to see if pond is active/in-use
            pond_active = check_active(pond_name, 5)

            try:
                pond_data = self.ponds_data[pond_name]
                pond_data_error = False
            except:
                pond_data_error = True

            if pond_active == True:
                date_pond_data = pond_data.loc[select_date] # get dataframe for individual pond

                # Query the data and update in data_dict
                for idx, (key, value) in enumerate(data_dict.items()):
                    query_name = key
                    value[1] = data_query(date_pond_data, pond_name, query_name)

                # Query and format (color code) of pests for each pond
                pond_pest_colors = get_pests_format(date_pond_data, pond_name)

                # set fill color based on afdw for now
                fill_color = get_fill_color(data_dict['afdw'][1], 'afdw_colors') 

            # Get the last harvest date for each pond, done separately from other data queries due to needing all dates
            try:
                pond_last_harvest = str(pond_data['Split Innoculum'].last_valid_index().date())
            except:
                pond_last_harvest = 'n/a'

            data_displays = ['title', ['depth', 'afdw'], 'harvestable_mass', 'growth']
            for idx, item in enumerate(data_displays):
                if item == 'title':
                    # title subplot
                    ax = plt.Subplot(fig, inner_plot[0,:])
                    t = ax.text(0.5, 0.001, r'$\bf{' + pond_name + '}$' + f'\nlast harvest/split: {pond_last_harvest}', 
                                ha = 'center', va='bottom')
                    # Display pond pest info next to subplot title
                    try:
                        start_x = 0.64
                        x_space = 0.05
                        for c in pond_pest_colors:
                            t = ax.text(start_x, 0.33, '*', ha='center', va='bottom', color=c, fontweight='bold', fontsize=17)
                            start_x += x_space
                    except: 
                        pass # if pond isn't active i.e., nothing to display
                    #ax.set_facecolor('whitesmoke')
                else:
                    # data subplots
                    ax = plt.Subplot(fig, inner_plot[1:,idx-1])
                    
                    # format data list item with a newline separator if there's more than one item (i.e., for 'depth' and 'afdw')
                    if pond_active == True: 
                        if type(item) == list:
                            text_formatted = ''
                            for idx, subitem in enumerate(item):
                                text_formatted += text_display_format(data_dict[subitem])
                                if idx+1 != len(item):
                                    text_formatted += '\n'
                        else:
                            text_formatted = text_display_format(data_dict[item])

                        # Plot data text for each item
                        t = ax.text(0.5, 0.5, text_formatted, ha='center', va='center')

                        ax.set_facecolor(fill_color)

                    else: # if pond is inactive or data_error
                        if idx == 2: # plot in the middle of the lower 3 subplots
                            if pond_data_error == True:
                                t = ax.text(0.5, 1, 'Data Error', ha='center', va='top')
                            else:
                                t = ax.text(0.5, 1, 'Inactive', ha='center', va='top')
                        ax.set_facecolor('snow')

                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])
                fig.add_subplot(ax)
            
            
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
        fig = plt.figure(figsize=(25,25))
        outer_plots = gridspec.GridSpec(len(title_labels), len(title_labels[0]), wspace=0.05, hspace=0.4)

        fig.suptitle(f'{plot_title}\n{select_date}', fontweight='bold', fontsize=16, y=0.91)
        
        total_available_to_harvest = 0 # keep track of total available for harvest across all ponds

        for idx_plot, pond_plot in enumerate(outer_plots):
            pond_name = flat_title_labels[idx_plot]
            if 'BLANK' not in pond_name:
                
                # plot each pond with a function to ensure that data for each is isolated within a local scope
                plot_each_pond(fig, pond_plot, pond_name, select_date)

            else: # for plotting subplots labeled 'BLANK' in 'title_labels' list:  (the four lower right subplots) and the BLANK rows
                ax = plt.Subplot(fig, pond_plot)
                ax.axis('off')
                if 'BLANK 11-6' in pond_name: # plot the legend in the first blank subplot
                    legend_text = ("Color key (AFDW):\n"

                                                       "0 - 0.24:                           Red\n"
                                                       "0.25 - 0.49:                      Yellow\n"
                                                       "0.50 - 0.79:                      Light Green\n"
                                                       "0.80 and up:                    Dark Green\n"
                                                       "No data for current day:  Grey")
                    t = ax.text(0.1,0.5,legend_text,ha='left', va='center', 
                           bbox=dict(facecolor='tab:red', alpha=0.5), multialignment='left')
                if 'BLANK 12-6' in pond_name: # plot the legend in the first blank subplot
                    pests_text = ("Color key (Pests > 0):\n"
                                  "Rotifers:                      Blue\n"
                                  "Attached FD111:         Red\n"
                                  "Free Floating FD111:  Pink\n"
                                  "Golden Flagellates:     Purple\n"
                                  "Diatoms:                     Cyan\n"
                                  "Tetra:                           Orange\n"
                                  "Green Algae:               Green") 
                    t = ax.text(0.1,0.5,pests_text,ha='left', va='center', 
                           bbox=dict(facecolor='tab:red', alpha=0.5), multialignment='left')
                elif 'BLANK 13-1' in pond_name:
                    print_string = (r'$\bf{Active\/\/ponds: }$' + f'{num_active_ponds}\n' 
                                   + r'$\bf{Total\/\/mass\/\/(all\/\/ponds):}$' + f'{total_mass_all:,} kg')
                    t = ax.text(0,1, print_string, ha='left', va='top', fontsize=16, multialignment='left')        
                    bottom_data_added_flag = True
                elif 'BLANK 13-3' in pond_name:
                    # sort the suggested_harvests dict by the pond column first (last 2 digits of name)
                    suggested_harvests = dict(sorted(suggested_harvests.items(), key=lambda item: item[0][-2:]))
                    # generate a formatted string for displaying
                    print_string = (r'$\bf{Suggested\/\/Harvest\/\/Depths}$' +  
                                    '\n(ponds with' + r'$\geq$' + f'{harvest_density} AFDW, with estimated harvest depth to reach {target_to_density:.1f}' + 
                                    f' AFDW with top off to {target_topoff_depth}"):\n')
                    for idx, (key, value) in enumerate(suggested_harvests.items()):
                        print_string += r'$\bf{' + str(key) + '}$' + ": " + f'{value:.2f}"' 
                        if (idx+1) % 6 == 0:
                            print_string += "\n"
                        elif (idx+1) != len(suggested_harvests):
                            print_string += "  |  "
                    if print_string[-1] != '\n': # check if last portion ended with newline char, append one if not
                        print_string += '\n'
                    print_string +=  r'$\bf{Suggested\/\/harvest\/\/mass:} $' + f'{target_harvest_mass:,} kg'
                    print_string += '\n' + r'$\bf{Suggested\/\/harvest\/\/volume:} $' + f'{target_harvest_gals:,.0f} gallons'
                    t = ax.text(1.1,1, print_string, ha='left', va='top', fontsize=14, multialignment='left')               
                elif 'BLANK 14-1' in pond_name:
                    print_string = (r'$\bf{Harvest\/\/Depth} = \frac{(Current\/\/Depth * Current\/\/AFDW) - (Target\/\/Top\/\/Off\/\/Depth * Target\/\/Harvest\/\/Down\/\/to\/\/AFDW)}{Current\/\/AFDW}$' +
                                    '\n' + r'$\bf{Target\/\/Harvest\/\/at\/\/AFDW}:$' + r'$\geq$' + str(harvest_density) +
                                    '\n' + r'$\bf{Target\/\/Harvest\/\/Down\/\/to\/\/AFDW}: $' + str(target_to_density) + 
                                    '\n' + r'$\bf{Target\/\/Top\/\/Off\/\/Depth}: $' + str(target_topoff_depth) + '"')
                    t = ax.text(0,.8, print_string, ha='left', va='top', fontsize=16) 
                    
                elif 'BLANK 14-3' in pond_name:
                    print_string = (r'$\bf{Harvestable\/\/Mass} = Harvest\/\/Depth * 132,489 (\frac{liters}{inch}) * Current\/\/AFDW$'
                                    + '\n                                   (doubled for 06 and 08 columns)')
                    t = ax.text(1.25,.8, print_string, ha='left', va='top', fontsize=16) 
                
                fig.add_subplot(ax) # add the subplot for the 'BLANK' entries
        
        if save_output == True:
            out_filename = f'./output_files/{plot_title}-{select_date}.pdf'
            plt.savefig(out_filename, bbox_inches='tight')
            print(f'Plot saved to:\n{out_filename}')
        fig.show() 
        

if __name__ == '__main__':   
    # detect if running in jupyter notebook (for development & testing)
    if 'ipykernel_launcher.py' in sys.argv[0]:
        pass
    else:
        main(sys.argv[1:])