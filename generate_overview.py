import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse
import sys
import re
import argparse
from datetime import datetime
from office365.sharepoint.client_context import ClientContext

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--input_file", help="OPTIONAL: Excel (.xls or .xlsx) input file for EPA data (default = None)")
    parser.add_argument("-d","--date", required=True, help="Date to generate report for in 'yyyy-mm-dd' format; year must be from 2020 to 2023")
    parser.add_argument("-t", "--target_density", type=float, default=0.4, help="OPTIONAL: target density of AFDW (Default = 0.40)")
    args = parser.parse_args()
    
    # check that file (OPTIONAL EPA DATA) is an excel file
    file_extension = args.input_file.split(".")[-1]
    if not (file_extension == 'xlsx' or file_extension == 'xls'):
        print("ERROR: scorecard data input filetype should be .xlsx or .xls")
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
                    date_error_flag = True
                    break
            # check that date value (as an integer) is from 1-31
            elif idx == 2:
                if int(number) < 1 or int(number) > 31:
                    date_error_flag = True
                    break
    if date_error_flag == True:
        print("ERROR: invalid date (should be 'yyyy-mm-ddd' between 2020 and 2023)")
        sys.exit(2)
    
    # check if optional target_density argument is valid (between 0 and 1)
    if args.target_density:
        if not (args.target_density > 0 and args.target_density < 1):
            print("ERROR: target density (AFDW) should be between 0 and 1")
            sys.exit(2)
    
    # initialize ponds_overview class
    overview = ponds_overview()
    
    # download latest daily data from sharepoint
    datafile = overview.download_scorecard_data()
    
    # load data, including optional EPA data if provided as arg
    print('Loading data...')
    ponds_data = overview.load_scorecard_data(datafile)
    if args.input_file:
        epa_data = overview.load_epa_data(args.input_file) 
    else:
        epa_data = None
    print('Plotting data...')
    ''' plot() method will return the output filename, so save it as a variable, then print it so that bash automation script 
        can access it from the last line of python stdout '''
    out_filename = overview.plot_scorecard(ponds_data=ponds_data, select_date=args.date, epa_data=epa_data, target_to_density=args.target_density) 
    print(f'Plot saved to:\n{out_filename}')
    sys.exit(0) # exit with status 0 to indicate successful execution
    
class ponds_overview:
    def __init__(self):
        pass
    
    def download_scorecard_data(self):
        from configparser import ConfigParser
        
        # load sharepoint credentials, site, file url from settings.cfg file
        def load_sharepoint_settings(specified_setting):
            cp = ConfigParser()
            config_file = './automation/settings.cfg'
            cp.read(config_file)
            return dict(cp.items(specified_setting))
     
        sharepoint_site, file_url = load_sharepoint_settings('scorecard_data_path').values()
        ctx = ClientContext(sharepoint_site).with_client_certificate(**load_sharepoint_settings('cert_credentials'))
        download_path = './data_sources/scorecard_data.xlsx'
        with open(download_path, "wb") as local_file:
            print('Downloading latest scorecard data')
            for i in range(5):
                try:
                    [print(f'Attempt {i+1}/5') if i > 0 else ''][0]
                    ctx.web.get_file_by_server_relative_url(file_url).download_session(local_file, lambda x: print(f'Downloaded {x/1e6:.2f} MB'),chunk_size=int(5e6)).execute_query()
                    print(f'Successful file download to {download_path}')
                    break
                except:
                    print('Download error...trying again')
        return download_path
        
    def load_scorecard_data(self, excel_filename):
        # Load the Daily Pond Scorecard excel file
        excel_sheets = pd.ExcelFile(excel_filename)

        # Get list of all sheet names
        sheetnames = sorted(excel_sheets.sheet_names)

        # initialize list of ponds corresponding to excel sheet names #### NOTE: '1201 ' has a trailing space that should be fixed in excel file
        ponds_list = ['0101', '0201', '0301', '0401', '0501', '0601', '0701', '0801', '0901', '1001', '1101', '1201 ', 
                      '0102', '0202', '0302', '0402', '0502', '0602', '0702', '0802', '0902', '1002', '1102', '1202',
                      '0103', '0203', '0303', '0403', '0503', '0603', '0703', '0803', '0903', '1003', '1103', '1203',
                      '0104', '0204', '0304', '0404', '0504', '0604', '0704', '0804', '0904', '1004', '1104', '1204',
                      '0106', '0206', '0306', '0406', '0506', '0606', '0706', '0806', '0906', '1006',
                      '0108', '0208', '0308', '0408', '0508', '0608', '0708', '0808', '0908', '1008']
        
        # create a dict containing a dataframe for each pond sheet
        all_ponds_data = {}
        for i in ponds_list:
            try:
                all_ponds_data[i] = excel_sheets.parse(i)
            except:
                print('Failed to load data for pond:', i)
            
        # clean the pond data (convert data from string to datetime, set date as index, drop empty columns   
        updated_ponds = {} # dict for storing cleaned data
        for key in enumerate(all_ponds_data.keys()):
            df = all_ponds_data[key[1]]
            df = df.rename(columns={df.columns[0]:'date'})
            df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.normalize() # convert date column from string *use .normalize method to remove potential time data
            df = df.dropna(subset=['date']) # drop rows that don't contain a date/valid date
            df = df.set_index(df['date'].name) # set date column as index
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')] # drop columns without a header, assumed to be empty or unimportant
            updated_ponds[key[1]] = df # add updated df to new dict of pond dataframes, which will later overwrite original 
        return updated_ponds # return the cleaned data dict
    
    def load_epa_data(self, excel_epa_data, excel_pond_history="") -> dict:

        def process_epa_row(sample_label, epa_val):
            label_split = sample_label.split()

            # Get date from label and convert to datetime format
            date = label_split[0]
            date = datetime.strptime(date, "%y%m%d")

            # Get pond number from label and format to match ponds_data dict keys
            pond_num = label_split[1]
            if pond_num.isnumeric() == False: # ignore the entries without a numeric pond name
                return 
            if len(pond_num) == 3:
                pond_num = '0' + pond_num

            # check if pond number exists as a key in pond data, ignore this data line if not    
            if pond_num not in ponds_data:
                print('KEY ERROR:', pond_num)
                return 
            
            # skip sample if the epa value is not a float or int
            if type(epa_val) not in (float, int):
                return
            
            if date not in ponds_data[pond_num]:
                ponds_data[pond_num]['epa_data'][date] = [epa_val]
            else:
                ponds_data[pond_num]['epa_data'][date].append(epa_val)
                
        def process_pond_history_row(pond_num, source_str):
            pond_num = str(pond_num)
            if len(pond_num) == 3:
                pond_num = '0' + pond_num
            ponds_data[pond_num]['source'] = source_str

        # initialize list of ponds 
        ponds_list = ['0101', '0201', '0301', '0401', '0501', '0601', '0701', '0801', '0901', '1001', '1101', '1201', 
                      '0102', '0202', '0302', '0402', '0502', '0602', '0702', '0802', '0902', '1002', '1102', '1202',
                      '0103', '0203', '0303', '0403', '0503', '0603', '0703', '0803', '0903', '1003', '1103', '1203',
                      '0104', '0204', '0304', '0404', '0504', '0604', '0704', '0804', '0904', '1004', '1104', '1204',
                      '0106', '0206', '0306', '0406', '0506', '0606', '0706', '0806', '0906', '1006',
                      '0108', '0208', '0308', '0408', '0508', '0608', '0708', '0808', '0908', '1008']

        # create a dict with an empty list for each pond number (to store date and epa value)
        ponds_data = {k: {'source': '', 'epa_data':{}} for k in ponds_list} 
        
        epa_df = pd.read_excel(excel_epa_data, sheet_name="%EPA Only")
        # process each row using Pandas vectorization and list comprehension rather than looping for better processing efficiency
        [process_epa_row(a, b) for a, b in (zip(epa_df['Sample'], epa_df['C20:5 Methyl eicosapentaenoate (2734-47-6), 10%']))]
        
        # load and process the pond_history (source of each pond's algae strain) if file provided as an argument
        if excel_pond_history:
            pond_history_df = pd.read_excel(excel_pond_history, sheet_name='Sheet1')
            [process_pond_history_row(a, b) for a, b in (zip(pond_history_df.iloc[:,0], pond_history_df.iloc[:,1]))]
        
        # Iterate through ponds_data, select only the last 3 daily readings 
        # and average the EPA values for each date (when there is more than one value per date)
        for idx, (pond_num, single_pond_data) in enumerate(ponds_data.copy().items()):
            if len(single_pond_data['epa_data']) > 0: 
                # filter data to select only the last 3 daily readings 
                ponds_data[pond_num]['epa_data'] = {k:v for i, (k,v) in enumerate(sorted(single_pond_data['epa_data'].items(), reverse=True), start=1) if i <= 3}
                # get the average of EPA all values for each day 
                for idx2, (date_key, date_val) in enumerate(ponds_data[pond_num]['epa_data'].copy().items()):
                    if len(date_val) > 0:
                        ponds_data[pond_num]['epa_data'][date_key] = sum(date_val) / len(date_val)
        return ponds_data 

    def plot_scorecard(self, ponds_data, select_date, epa_data=None, target_to_density=0.4, target_topoff_depth=13, harvest_density=0.5, save_output=True, plot_title='Pond Health Overview'): 
        # Normalize select_date to remove potential time data and prevent possible key errors when selecting date range from data
        select_date = pd.to_datetime(select_date).normalize()
        
        # Initialize variables for aggregations
        total_mass_all = 0 # total mass for entire farm (regardless of density)
        total_harvestable_mass = 0 # all available mass with afdw > target_to_density
        target_harvest_mass = 0 # all harvestable mass with resulting afdw > target_to_density but only ponds with current afdw > harvest_density
        target_harvest_gals = 0 # target_harvest_mass in terms of volume in gallons
        suggested_harvests = {} # dict to store calculated harvest depths for ponds with density >= harvest_density
        num_active_ponds = 0 # number of active ponds (defined by the plot_ponds().check_active() function)
        num_active_ponds_sm = 0 # number of active 1.1 acre ponds
        num_active_ponds_lg = 0 # number of active 2.2 acre ponds
        
        # helper function to check if a pond has data from prior n-days from the selected date 
        def check_active(ponds_data, pond_name, num_days_prior):
            def check_active_query(prev_day_n): 
                try:
                    dat = ponds_data[pond_name][['Fo','Split Innoculum']].shift(prev_day_n).loc[select_date]
                    if pd.isna(dat[1]) == False:
                        # check if pond is noted as "I" for inactive in the "Split Innoculum" column, 
                        # if so, break immediately and return False in the outer function
                        if dat[1].upper() == 'I': 
                            return 'FalseBreakImmediate'
                    elif pd.isna(dat[0]) or dat[0] == 0:
                        return False
                    else:
                        return True
                except:
                    return False
            for n in range(0,num_days_prior+1):
                prev_n_day_active = check_active_query(n)
                if prev_n_day_active == True:
                    nonlocal num_active_ponds, num_active_ponds_sm, num_active_ponds_lg
                    #check if pond is in the '06' or '08' column by checking the last values in name
                    if pond_name[2:] == '06' or pond_name[2:] == '08':
                        num_active_ponds_lg += 1
                    else:
                        num_active_ponds_sm += 1
                    num_active_ponds += 1
                    return True
                elif prev_n_day_active == 'FalseBreakImmediate':
                    return False
                elif n == num_days_prior and prev_n_day_active == False:
                    return False
                else:
                    pass
        
        # helper function to query data from previous days, if it is not available for current day
        def current_or_prev_query(df, col_name, num_prev_days, return_prev_day_num=False):
                    for day_num in range(0,num_prev_days+1):
                        data = df[col_name].shift(day_num).fillna(0).loc[select_date] # use fillna(0) to force missing data as 0
                        if data > 0:
                            if return_prev_day_num == True:
                                return data, day_num
                            else:
                                return data
                    return 0 # return 0 if no data found (though 'data' variable will be 0 anyway due to fillna(0) method used) 
                
        # helper function to query single data points from each pond dataframe
#         def data_query(pond_data, pond_name, query_name):
        
        # helper function to convert density & depth to mass (in kilograms)
        def afdw_depth_to_mass(afdw, depth, pond_name):
            depth_to_liters = 35000 * 3.78541 # save conversion factor for depth (in inches) to liters
            # for the 6 and 8 columns, double the depth to liters conversion because they are twice the size
            if pond_name[-2:] == '06' or pond_name[-2:] == '08': 
                depth_to_liters *= 2
            # calculate and return total mass (kg)
            return (afdw * depth_to_liters * depth) / 1000
        
        def subplot_ax_format(subplot_ax):
            subplot_ax.spines['top'].set_visible(False)
            subplot_ax.spines['right'].set_visible(False)
            subplot_ax.spines['bottom'].set_visible(False)
            subplot_ax.spines['left'].set_visible(False)
            subplot_ax.get_xaxis().set_ticks([])
            subplot_ax.get_yaxis().set_ticks([])
            fig.add_subplot(subplot_ax)
                    
        # function to plot each pond to ensure that data for each plot is kept within a local scope
        def plot_each_pond(ponds_data, fig, pond_plot, pond_name, select_date):
            inner_plot = gridspec.GridSpecFromSubplotSpec(5,3,subplot_spec=pond_plot, wspace=-0.01, hspace=-0.01)

            # Check prior 5 days of data to see if pond is active/in-use
            pond_active = check_active(ponds_data, pond_name, 5)

            try:
                single_pond_data = ponds_data[pond_name]
                pond_data_error = False
            except:
                pond_data_error = True
            
            # Gather data for pond if it's active
            if pond_active == True:
                date_single_pond_data = single_pond_data.loc[select_date] # get dataframe for individual pond
                
                # calculate data for pond subplot display
                pond_data_afdw = single_pond_data.loc[select_date]['AFDW (filter)']
                pond_data_afdw = [0 if pd.isna(pond_data_afdw) else pond_data_afdw][0] # set to 0 if na value (i.e., nothing entered at all)
                pond_data_depth = single_pond_data.loc[select_date]['Depth']
                pond_data_depth = [0 if pd.isna(pond_data_depth) else pond_data_depth][0] # set to 0 if na value
                if (pond_data_afdw == 0 or pond_data_depth == 0) != True:
                    pond_data_total_mass = int(afdw_depth_to_mass(pond_data_afdw, pond_data_depth, pond_name))
                    # calculate harvestable depth (in inches) based on depth and afdw and the target_topoff_depth & target_to_density global function parameters
                    pond_data_harvestable_depth = round(((pond_data_depth * pond_data_afdw) - (target_topoff_depth * target_to_density)) / pond_data_afdw,2)
                    if pond_data_harvestable_depth < 0: 
                        pond_data_harvestable_depth = 0
                    # calculate harvestable volume (in gallons) based on harvestable depth and conversion factor (35,000) to gallons. Double for the '06' and '08' column ponds since they are double size
                    pond_data_harvestable_gallons = pond_data_harvestable_depth * 35000
                    if pond_name[-2:] == '06' or pond_name[-2:] == '08':
                        pond_data_harvestable_gallons *= 2
                    pond_data_harvestable_mass = int(afdw_depth_to_mass(pond_data_afdw, pond_data_harvestable_depth, pond_name))

                    # Add pond info to global counters/data for the entire farm
                    nonlocal total_mass_all
                    total_mass_all += pond_data_total_mass # add pond mass to the total_mass_all counter for entire farm
                    if pond_data_afdw > harvest_density: # add these only if current pond density is greater than the global function parameter 'harvest_density'
                        nonlocal suggested_harvests
                        nonlocal target_harvest_mass
                        nonlocal target_harvest_gals
                        suggested_harvests[pond_name] = round(pond_data_harvestable_depth,2)
                        target_harvest_mass += pond_data_harvestable_mass
                        target_harvest_gals += pond_data_harvestable_gallons
                else:
                    pond_data_total_mass = 0
                    pond_data_harvestable_depth = 0
                    pond_data_harvestable_mass = 0
                
                # Query and format (color code) of pests for each pond
                indicators_data = date_single_pond_data[['pH', 'Rotifers ','Attached FD111','Free Floating FD111', 'Golden Flagellates', 'Diatoms', 
                                    'Tetra','Green Algae']].fillna(0)
                
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
                
                # get most recent EPA value if it's provided as an argument
                if epa_data:
                    try:
                        epa_val = list(epa_data[pond_name]["epa_data"].values())[0] # get 0 index since data should be sorted by most-recent first
                    except:
                        epa_val = 'n/a'
                
                # set fill color based on density (afdw) for now
                if pond_data_afdw == 0:
                    fill_color = 'lightgrey'
                else:    
                    density_color_dict = {(0.000000001,0.25): 0, 
                                          (0.250000001,0.5): 1,
                                          (0.500000001,0.8): 2, 
                                          (0.800000001,999999999): 3}
                    color_list = ['red', 'yellow', 'mediumspringgreen', 'tab:green']
                    for idx, (key, val) in enumerate(density_color_dict.items()):
                        if pond_data_afdw >= key[0] and pond_data_afdw < key[1]:
                            color_idx = val
                            fill_color = color_list[color_idx] 
                    # secondary fill color by EPA percentage
                    if epa_data:
                        if epa_val == 'n/a':
                            fill_color='lightgrey'
                        elif color_idx > 0 and epa_val < 0.02:
                            fill_color = 'red'
                        elif color_idx > 1 and epa_val < 0.035:
                            fill_color = 'yellow'
                
            # Get the last harvest date for each pond, done separately from other data queries due to needing full range of dates, and regardless of whether it's active
            # Get the date relative to the select_date parameter (so reports can always be generated relative to a specific date even for past dates)
            try:
                last_harvest_idx = single_pond_data['Split Innoculum'].loc[:select_date].last_valid_index()
                pond_last_harvest_str = last_harvest_idx.date().strftime('%-m-%-d-%y')
            except:
                last_harvest_idx = 'n/a'
                pond_last_harvest_str = 'n/a'
                
            ## Calculate growth rates since last harvest / and 5-day   
            ## TODO: Reformulate to calculate based on average of each daily change (currently just checking difference between two values and dividing by # days)
            if last_harvest_idx != 'n/a':    
                new_growth_start_idx = last_harvest_idx + pd.Timedelta(days=1)
                if pond_active == True:       
                        if new_growth_start_idx < select_date: # i.e., if pond has more than 1 day of growth
                            # select only data for the period of growth that's being calculated
                            pond_growth_df = single_pond_data.loc[new_growth_start_idx:select_date][['AFDW (filter)', 'Depth']]
                            # calculate mass to use for calculating growth rate 
                            pond_growth_df['mass'] = afdw_depth_to_mass(pond_growth_df['AFDW (filter)'], pond_growth_df['Depth'], pond_name)
                            # replace 0 values with 'nan' for interpolation to work and fill gaps of missing data
                            pond_growth_df['mass'] = pond_growth_df['mass'].replace(0, float('nan'))
                            # interpolate missing data (will not fill beginning or end values if they're missing)
                            pond_growth_df['mass'] = pond_growth_df['mass'].interpolate()
                            # drop missing values after interpolation (should just be values at beginning of period)
                            # but could also drop the current selection date if AFDW or Depth is missing for this day
                            pond_growth_df = pond_growth_df.dropna(subset=['mass'])
                            
                            # calculate the number of growth days
                            # Do this after dropping empty leading/trailing rows because those are not being included in growth calculation 
                            # because there isn't a great method of linearly extrapolating and estimating mass values for those rows
                            num_growth_days = (pond_growth_df.index[-1] - pond_growth_df.index[0]).days
                            
                            # calculate growth rates
                            # assume last harvest date is not valid if growing days are more than 9
                            # also don't calculate if there are 0 growing days (AKA 0 days of data since last harvest/split)
                            if num_growth_days > 90 or num_growth_days == 0: 
                                daily_growth_rate_since_harvest = 'n/a'
                            else:
                                daily_growth_rate_since_harvest = int((pond_growth_df['mass'].iloc[-1] - pond_growth_df['mass'].iloc[0])/num_growth_days)
                                #daily_growth_rate_since_harvest = round(((pond_growth_df['mass'].iloc[-1] - pond_growth_df['mass'].iloc[0]) / pond_growth_df['mass'].iloc[0])/num_growth_days,4)
                            # just use a try/except clause since the 5 day growth calculation will cause error if there aren't 5 days of available data in the current growth period
                            try:        
                                daily_growth_rate_5_days = int((pond_growth_df['mass'].iloc[-1] - pond_growth_df['mass'].iloc[-5])/5)
                                #daily_growth_rate_5_days = round(((pond_growth_df['mass'].iloc[-1] - pond_growth_df['mass'].iloc[-5]) / pond_growth_df['mass'].iloc[-5])/5,4)
                            except:
                                daily_growth_rate_5_days = 'n/a'
                        else: # if pond has 1 or less days of growth (can't calculate rate)
                            daily_growth_rate_since_harvest = 'n/a'
                            daily_growth_rate_5_days = 'n/a'
                        
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
                                
            # title subplot
            title_ax = plt.Subplot(fig, inner_plot[:2,:])

            # plot pond EPA data
            if epa_data:
                try:
                    if epa_val == 'n/a':
                        epa_val_str = 'No EPA data'
                    else:
                        epa_val_str = f'{epa_val * 100:.2f}% EPA'
                    title_ax.text(0.8, 0.8, epa_val_str, ha='center', va='center')
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
                # format depth data points 
                if pond_data_depth == 0:
                    pond_data_depth = 'No \ data'
                else:    
                    pond_data_depth = str(round(pond_data_depth,2))
                if pond_data_harvestable_depth == 0:
                    pond_data_harvestable_depth = '-'
                else:
                    pass # don't need to update in this case
                
                if pond_data_afdw == 0:
                    pond_data_afdw = 'No \ data'
                else:
                    pond_data_afdw = round(pond_data_afdw,3)
                
                # format harvestable mass
                if pond_data_harvestable_mass == 0:
                    pond_data_harvestable_mass = '-'
                else:
                    pond_data_harvestable_mass = f'{pond_data_harvestable_mass} \ kg' 
                
                # format growth rates due to needing to add "\" before "%" sign with matplotlib boldface formatting, 
                # and being unable to add with f-string 
                if daily_growth_rate_since_harvest != 'n/a':
                    daily_growth_rate_since_harvest_formatted = f'{daily_growth_rate_since_harvest} \ kg'
                    #daily_growth_rate_since_harvest_formatted = str(round(daily_growth_rate_since_harvest * 100,2)) + '\%'
                else:
                    daily_growth_rate_since_harvest_formatted = 'n/a'
                if daily_growth_rate_5_days != 'n/a':
                    daily_growth_rate_5_days_formatted = f'{daily_growth_rate_5_days} \ kg'
                    #daily_growth_rate_5_days_formatted = str(round(daily_growth_rate_5_days * 100,2)) + '\%'
                else: 
                    daily_growth_rate_5_days_formatted = 'n/a'
                    
                data_plot_dict = [{'Measurement': {'Depth': pond_data_depth, 'AFDW': pond_data_afdw}},
                                  {'Harvestable': {'Mass': pond_data_harvestable_mass, 
                                                   'Depth': pond_data_harvestable_depth}},
                                  {'Daily Growth': {'Since Last H/S': daily_growth_rate_since_harvest_formatted,  
                                                    '5 Days': daily_growth_rate_5_days_formatted}
                                  }]
            
            # data subplots   
            for idx in range(3):
                ax = plt.Subplot(fig, inner_plot[2:,idx])

                # format data list item with a newline separator if there's more than one item (i.e., for 'depth' and 'afdw')
                if pond_active == True: 
                    text_formatted = ''
                    for idx2, (key, val) in enumerate(data_plot_dict[idx].items()):
                        text_formatted += f'{key}:\n'
                        if type(val) == dict:
                            for sub_idx, (sub_key, sub_val) in enumerate(val.items(), start=1):
                                text_formatted += f'{sub_key}\n'
                                text_formatted += r'$\bf{' + str(sub_val) + '}$'
                                text_formatted += '%s' %('\n' if sub_idx != len(val) else '') 
                        else:
                            text_formatted += r'$\bf{' + str(val) + '}$'
                        if idx2+1 != len(data_plot_dict[idx]):
                            text_formatted += '\n'
                 
                    # Plot formatted datatext for each item
                    t = ax.text(0.5, 0.1, text_formatted, ha='center', va='bottom')
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
        outer_plots = gridspec.GridSpec(len(title_labels), len(title_labels[0]), wspace=0.05, hspace=0.1)
        
        #title_date = '/'.join([select_date.split('-')[i].lstrip('0') for i in [1,2,0]])
        title_date = select_date.strftime('%-m/%-d/%Y')
        fig.suptitle(f'{plot_title}\n{title_date}', fontweight='bold', fontsize=16, y=0.905)
        
        total_available_to_harvest = 0 # keep track of total available for harvest across all ponds

        for idx_plot, pond_plot in enumerate(outer_plots):
            pond_name = flat_title_labels[idx_plot]
            if 'BLANK' not in pond_name:
                
                # plot each pond with a function to ensure that data for each is isolated within a local scope
                plot_each_pond(ponds_data, fig, pond_plot, pond_name, select_date)

            else: # for plotting subplots labeled 'BLANK' in 'title_labels' list:  (the four lower right subplots) and the BLANK rows
                ax = plt.Subplot(fig, pond_plot)
                ax.axis('off')
                if 'BLANK 11-6' in pond_name: # plot the color key in the first blank subplot
                    # different color keys depending on whether EPA data is provided or not
                    if epa_data:
                        legend_text = (
                                        r'$\bf{Color\/\/key}$:' "\n"
                                       r'    $\bf{AFDW} \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \bf{EPA}$'"\n"         
                                        "       < 0.25     or               < 2%:     Red\n"
                                        "0.25 - 0.49     or      2% - 3.49%:     Yellow\n"
                                        "0.50 - 0.79    and          "r'$\geq$'" 3.5%:     Light Green\n"
                                        "     "r'$\geq$'" 0.80    and          "r'$\geq$'" 3.5%:     Dark Green\n"
                                        "                        Incomplete data:     Grey"
                                       )
                    else:
                           legend_text = (
                                    r'$\bf{Color\/\/key\/\/(AFDW)}$:' "\n"
                                    "0 - 0.24:                           Red\n"
                                    "0.25 - 0.49:                      Yellow\n"
                                    "0.50 - 0.79:                      Light Green\n"
                                    "0.80 and up:                    Dark Green\n"
                                    "No data for current day:  Grey"
                                   )
                        
                    t = ax.text(0.1,0.9,legend_text,ha='left', va='top', 
                           bbox=dict(facecolor='xkcd:bluegrey', alpha=0.5), multialignment='left')
                if 'BLANK 12-6' in pond_name: # plot the legend in the first blank subplot    
                    indicators_legend_text = (
                                  r'$\bf{Indicators }$' "\n"
                                  r'% Nanno: < 80%                  %N (red)' "\n"
                                  r'pH: out of 7.8 - 8.2 range      pH (red)' "\n"
                                  r'Rotifers: $\geq$ 1                         R (blue)' "\n"
                                  r'Attached FD111: $\geq$ 1           FD-A (brown)' "\n"
                                  r'Free Floating FD111: $\geq$ 1     FD-FF (orange)' "\n"
                                  r'Golden Flagellates: $\geq$ 1        GF (yellow)' "\n"
                                  r'Diatoms: $\geq$ 0.5                     D (pink)' "\n"
                                  r'Tetra: $\geq$ 0.5                          T (purple)' "\n"
                                  r'Green Algae: $\geq$ 0.5               GA (green)'
                                 ) 
                    t = ax.text(0.1,0.62,indicators_legend_text,ha='left', va='center', 
                           bbox=dict(facecolor='xkcd:bluegrey', alpha=0.5), multialignment='left')
                elif 'BLANK 13-1' in pond_name:
                    print_string = (r'$\bf{Total\/\/Active\/\/ponds: }$' + f'{num_active_ponds}\n' 
                                    + r'$\bf{Active\/\/1.1\/\/acre\/\/ponds: }$' + f'{num_active_ponds_sm}\n'
                                    + r'$\bf{Active\/\/2.2\/\/acre\/\/ponds: }$' + f'{num_active_ponds_lg}\n'
                                   + r'$\bf{Total\/\/mass\/\/(all\/\/ponds):}$' + f'{total_mass_all:,} kg')
                    t = ax.text(0,.75, print_string, ha='left', va='top', fontsize=16, multialignment='left')        
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
                    t = ax.text(0,.75, print_string, ha='left', va='top', fontsize=14, multialignment='left')               
                # elif 'BLANK 14-1' in pond_name:
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
        
        out_filename = f'./output_files/{plot_title} {select_date.strftime("%Y-%m-%d")}.pdf'
        if save_output == True:
            plt.savefig(out_filename, bbox_inches='tight')
        fig.show() 
        return out_filename
    
    ## TODO implement a way of plotting this EPA overview as an additional page to the PDF output implemented in main()
    def plot_epa(self, ponds_data: dict, save_output=True, plot_title='Pond EPA Overview') -> str: # returns output file name as a string 
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
        def plot_each_pond(ponds_data, fig, pond_plot, pond_name):
            inner_plot = gridspec.GridSpecFromSubplotSpec(5,3,subplot_spec=pond_plot, wspace=0, hspace=0)

            # get data for individual pond only, as a list
            # each list entry should consist of a tuple containing the date and the epa value, sorted in descending order by date
            single_pond_data = list(ponds_data[pond_name]['epa_data'].items())
            
             # check and update the title_date global variable if this pond has a more recent date
            if len(single_pond_data) != 0:
                nonlocal title_date
                if single_pond_data[0][0] > title_date:
                    title_date = single_pond_data[0][0]
            
            # title subplot
            title_ax = plt.Subplot(fig, inner_plot[0,:])
            title_str = r'$\bf{' + pond_name + '}$' 
            if len(ponds_data[pond_name]['source']) > 0:
                title_str += '\nsource: ' + ponds_data[pond_name]['source']
            title_ax.text(0.5, 0.1, title_str , ha = 'center', va='bottom', fontsize='large')
            subplot_ax_format(title_ax, 'white')

            # epa data subplots
            epa_ax = plt.Subplot(fig, inner_plot[1:3,:])

            # set fill color based on latest EPA reading
            if len(single_pond_data) > 0:
                # set lastest_epa_data as the first value in single_pond_data (which should be reverse sorted)
                latest_epa_data = single_pond_data[0][1] 
                fill_color_dict = {(0,0.0199999999):'red', 
                                   (0.02,0.0349999999): 'yellow',
                                   (0.035,99999999999): 'mediumspringgreen'}
                for idx, (key, val) in enumerate(fill_color_dict.items()):
                    if latest_epa_data >= key[0] and latest_epa_data < key[1]:
                        fill_color = val 
            else:
                # set fill color as light grey if there is no EPA data for the pond
                fill_color = 'whitesmoke'

            subplot_ax_format(epa_ax, fill_color)

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
                    text_epa_data_formatted =  f'{item[1]*100: .2f}%'
                    epa_ax.text(text_x[idx], 0.3, text_epa_data_formatted, ha='center', va='center', fontsize='large', weight='bold')
            else:
                epa_ax.text(0.5, 0.5, 'No Data', ha='center', va='center')

            # epa change
            epa_pct_ax = plt.Subplot(fig, inner_plot[3:,:])

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
                    
                    pct_chg = (data_curr[1] - data_prev[1]) * 100
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
        
        # initialize title_date as a global variable to track the most recent date to print in the title
        title_date = datetime.min
        
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
                plot_each_pond(ponds_data, fig, pond_plot, pond_name)

            else: # for plotting subplots labeled 'BLANK' in 'title_labels' list:  (the four lower right subplots) and the BLANK rows
                ax = plt.Subplot(fig, pond_plot)
                ax.axis('off')
                if 'BLANK 11-6' in pond_name: # plot the color key in the first blank subplot
                    legend_text = (
                                    r'$\bf{Color\/\/key\/\/(latest\/\/EPA\/\/reading)}$:' "\n"
                                    "0% - 1.99%:      Red\n"
                                    "2% - 3.49%:      Yellow\n"
                                    "3.5% and up:    Green\n"
                                    "No EPA data:     Grey"
                                   )
                    t = ax.text(0.1,0.8,legend_text,ha='left', va='top', fontsize = 'x-large',
                           bbox=dict(facecolor='xkcd:bluegrey', alpha=0.5), multialignment='left')
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
        
        title_date_formatted = title_date.strftime('%-m/%-d/%Y')
        fig.suptitle(f'{plot_title}\nas of {title_date_formatted}', fontweight='bold', fontsize=16, y=0.905)
        
        out_filename = f'./output_files/{plot_title} {title_date.strftime("%Y-%m-%d")}.pdf'
        if save_output == True:
            plt.savefig(out_filename, bbox_inches='tight')
        fig.show() 
        return out_filename
        
if __name__ == '__main__':   
    # detect if running in jupyter notebook (for development & testing)
    if 'ipykernel_launcher.py' in sys.argv[0]:
        pass
    else:
        main(sys.argv[1:])