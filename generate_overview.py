import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse, Rectangle
import sys
import argparse
from datetime import datetime
from office365.sharepoint.client_context import ClientContext
from configparser import ConfigParser
from os.path import getsize, isfile
from O365 import Account, FileSystemTokenBackend
import traceback
import re

def main(argv):
    def send_email(recipients, subject, msg_body, attachment_path=None):
        email_settings = load_setting('email_cred')
        credentials = (email_settings['client_id'], email_settings['client_secret'])
        tenant = email_settings['tenant']
        token_backend = FileSystemTokenBackend(token_path='./automation/auth_data/', token_filename='outlook_auth_token.txt') # save token for email auth to re-use 
        account = Account(credentials, auth_flow_type='authorization', tenant_id=tenant, token_backend=token_backend)
        if not account.is_authenticated:  # will check if there is a token and has not expired
            # ask for a login 
            account.authenticate(scopes=['basic', 'message_all'])
        message = account.mailbox().new_message()
        message.to.add(recipients) 
        message.subject = subject
        message.body = msg_body
        if attachment_path:
          message.attachments.add(attachment_path)
        message.send()
        print('Email successfully sent!')
    
    def failure_notify_email_exit(failure_reason, traceback=None):
        print(failure_reason)
        email_msg_info = load_setting('email_failure_msg')
        send_email(recipients = [x.strip() for x in email_msg_info['recipients'].split(',')], 
                   # split recipients on ',' and remove whitespace because ConfigParser will import it as a single string, but should be a list if more than 1 address
                    subject = f'{email_msg_info["subject"]} - {datetime.strptime(args.date,"%Y-%m-%d").strftime("%a %b %-d, %Y")}', # add date to the end of the email subject
                    msg_body = f'{failure_reason}{f"<br><br>Traceback:<br>{traceback}" if traceback else ""}'
                  )
        sys.exit(1)
                
    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--input_file", help="OPTIONAL: Excel (.xls or .xlsx) input file for EPA data (default = None)")
    parser.add_argument("-d","--date", required=True, help="Date to generate report for in 'yyyy-mm-dd' format; year must be from 2020 to 2023")
    parser.add_argument("-t", "--target_density", type=float, default=0.4, help="OPTIONAL: target density of AFDW (Default = 0.40)")
    args = parser.parse_args()
    
    # check that file (OPTIONAL EPA DATA) is an excel file
    if args.input_file:
        if not isfile(args.input_file):
            failure_notify_email_exit(f'ERROR: EPA data file specified ({args.input_file}) does not exist!')
        file_extension = args.input_file.split(".")[-1]
        if not (file_extension == 'xlsx' or file_extension == 'xls'):
            failure_notify_email_exit('ERROR: EPA data filetype should be .xlsx or .xls')
    
    # check if date argument is valid, use try/except clause with datetime.strptime because it will generate an error with invalid date
    try:
        date_check = datetime.strptime(args.date, '%Y-%m-%d')
    except:
        invalid_date = args.date
        args.date = '9999-01-01' # set to something ridiculous that's still a valid date so the error email will still generate, since date is added to the subject line
        failure_notify_email_exit(f'Invalid date specified: {invalid_date}', tb)
    
    # check if optional target_density argument is valid (between 0 and 1)
    if args.target_density:
        if not (args.target_density > 0 and args.target_density < 1):
            failure_notify_email_exit("ERROR: target density (AFDW) should be between 0 and 1")
    
    # initialize ponds_overview class
    try:
        overview = ponds_overview(args.date)
    except Exception as ex:
        tb = ''.join(traceback.TracebackException.from_exception(ex).format())
        failure_notify_email_exit(f'Error running pond overview script', tb)
        
    print('Emailing message with attachment...')
    email_msg_info = load_setting('email_msg')
    send_email(recipients = [x.strip() for x in email_msg_info['recipients'].split(',')], 
               # split recipients on ',' and remove whitespace because ConfigParser imports as a single string, but needs to be a list of each email string 
                subject = f'{email_msg_info["subject"]} - {datetime.strptime(args.date,"%Y-%m-%d").strftime("%a %b %-d, %Y")}', # add date to the end of the email subject
                msg_body = email_msg_info['body'],
                attachment_path = overview.out_filename) 
    sys.exit(0) # exit with status 0 to indicate successful execution

# load auth credentials & settings from settings.cfg file
def load_setting(specified_setting):
    cp = ConfigParser()
    cp.read('./automation/settings.cfg')
    return dict(cp.items(specified_setting))   

class ponds_overview:
    def __init__(self, select_date):
        select_date = pd.to_datetime(select_date).normalize() # Normalize select_date to remove potential time data and prevent possible key errors when selecting date range from data
        self.ponds_list = ['0101', '0201', '0301', '0401', '0501', '0601', '0701', '0801', '0901', '1001', '1101', '1201', 
                          '0102', '0202', '0302', '0402', '0502', '0602', '0702', '0802', '0902', '1002', '1102', '1202',
                          '0103', '0203', '0303', '0403', '0503', '0603', '0703', '0803', '0903', '1003', '1103', '1203',
                          '0104', '0204', '0304', '0404', '0504', '0604', '0704', '0804', '0904', '1004', '1104', '1204',
                          '0106', '0206', '0306', '0406', '0506', '0606', '0706', '0806', '0906', '1006',
                          '0108', '0208', '0308', '0408', '0508', '0608', '0708', '0808', '0908', '1008']
        scorecard_datafile = self.download_data('scorecard_data_info')
        scorecard_dataframe = self.load_scorecard_data(scorecard_datafile)
        epa_datafile = self.download_data('epa_data_info')
        epa_data_dict = self.load_epa_data(epa_datafile, select_date)
        self.epa_data_dict_old = self.load_epa_data_old('./data_sources/epa_data_old.xlsx')
        self.num_active_ponds = 0 
        self.num_active_ponds_sm = 0 
        self.num_active_ponds_lg = 0
        self.ponds_active_status = {key:self.check_active(pond_scorecard_data=scorecard_dataframe, select_date=select_date, pond_name=key, num_days_prior=5) for key in self.ponds_list}
        self.out_filename = self.plot_scorecard(ponds_data=scorecard_dataframe, select_date=select_date, epa_data=epa_data_dict)

    def download_data(self, data_item_setting):
        sharepoint_site, file_url, download_path, expected_min_filesize, print_label = load_setting(data_item_setting).values()
        
        # sharepoint auth connect
        ctx = ClientContext(sharepoint_site).with_client_certificate(**load_setting('sharepoint_cert_credentials'))
        
        with open(download_path, "wb") as local_file:
            print(f'Downloading latest {print_label}')
            for i in range(5):
                try:
                    [print(f'Attempt {i+1}/5') if i > 0 else ''][0]
                    ctx.web.get_file_by_server_relative_url(file_url).download_session(local_file, lambda x: print(f'Downloaded {x/1e6:.2f} MB'),chunk_size=int(5e6)).execute_query()
                    break
                except:
                    print('Download error...trying again')
                    if i == 4:
                        print('Daily scorecard file download error')
                        return False
        if getsize(download_path) > int(expected_min_filesize):  # check that downloaded filesize is > 35 MB, in case of any download error
            print(f'{print_label} successfully downloaded to {download_path}')
            return download_path
        else:
            print(f'{print_label} download error')
            return False
        
    def load_scorecard_data(self, excel_filename):
        print('Loading scorecard data...')
        # Load the Daily Pond Scorecard excel file
        excel_sheets = pd.ExcelFile(excel_filename)

        # Get list of all sheet names
        sheetnames = sorted(excel_sheets.sheet_names)

        # initialize list of ponds corresponding to excel sheet names #### NOTE: '1201 ' has a trailing space that should be fixed in excel file
        ponds_list = self.ponds_list
        
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
        print('Scorecard data loaded!')
        return updated_ponds # return the cleaned data dict
    
    ## TODO load date relative to select date (i.e., only data equal to or before select_date)
    def load_epa_data(self, excel_epa_data, select_date) -> dict:
        def process_epa_row(sample_label, epa_val):    
            # check if sample_label string contains the substring "Pond", if not, skip this data row
            if 'Pond' not in sample_label:
                return
            
            try:
                epa_val = float(epa_val)
            except:
                #print('ERROR: EPA val is not a valid number')
                return
            
            # search for pond name in sample_label with regex (looking for 4 digits surrounded by nothing else or whitespace)
            # regex ref: https://stackoverflow.com/questions/45189706/regular-expression-matching-words-between-white-space
            pondname_search = re.search(r'(?<!\S)\d{4}(?!\S)', sample_label)
            if pondname_search:
                pond_name = pondname_search.group()
            else:
                #print('ERROR: no pond name found in sample label')
                return

            # check if pond number exists as a key in pond data, ignore this data line if not    
            if pond_name not in ponds_data:
                print('KEY ERROR:', pond_name)
                return 

            # check if pond_name already has an epa value in the ponds_data dict, if so then skip processing this row
            if ponds_data[pond_name]['epa_val'] != '':
                #print('More recent pond data already found. Skipping...')
                return

            # search for date in sample_label with regex (looking for 6 digits surrounded by nothing else or whitespace)
            date_search = re.search(r'(?<!\S)\d{6}(?!\S)',sample_label)
            if date_search:
                date = datetime.strptime(date_search.group(), "%y%m%d")
                if (select_date - date).days > 60:
                    #print('ERROR: sample over 60 day threshold')
                    return
            else:
                #print('ERROR: no date found in sample label')
                return
            
            ponds_data[pond_name]['date'] = date
            ponds_data[pond_name]['epa_val'] = epa_val

        print('Loading EPA data...')

        ponds_list = self.ponds_list

        # create a dict with an empty list for each pond number (to store date and epa value)
        ponds_data = {k: {'date': '', 'epa_val':''} for k in ponds_list} 

        # load epa_data spreadsheet and parse the first 4 columns to update header (since excel file col headers are on different rows/merged rows)
        epa_df = pd.read_excel(excel_epa_data, sheet_name="Sheet1", header=None)
        epa_df.iloc[0:4] = epa_df.iloc[0:4].fillna(method='bfill', axis=0) # backfill empty header rows in the first 4 header rows, results in header info copied to first row for all cols
        epa_df.iloc[0] = epa_df.iloc[0].apply(lambda x: ' '.join(x.split())) # remove extra spaces from the strings in the first column
        epa_df.columns = epa_df.iloc[0] # set header row
        epa_df = epa_df.iloc[4:] # delete the now unnecessary/duplicate rows of header data

        # initialize a count of the ponds that have had updated epa values, so that processing of the epa data file can end immediately when epa_val count == len(ponds_list)
        epa_val_count = 0

        # process each row using Pandas vectorization and list comprehension rather than looping for better processing efficiency
        # process in reverse order to check for most-recent samples first
        [process_epa_row(a, b) for a, b in (zip(epa_df['Sample type'][::-1], epa_df['EPA % AFDW'][::-1]))]

        print('EPA data loaded!')
        return ponds_data 

####### OLD FOR PREVIOUS EPA DATA FORMAT
    def load_epa_data_old(self, excel_epa_data, excel_pond_history="") -> dict:
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
            
            epa_val = epa_val * 100 # convert from decimal to percentage
            
            if date not in ponds_data[pond_num]:
                ponds_data[pond_num]['epa_data'][date] = [epa_val]
            else:
                ponds_data[pond_num]['epa_data'][date].append(epa_val)
                
        def process_pond_history_row(pond_num, source_str):
            pond_num = str(pond_num)
            if len(pond_num) == 3:
                pond_num = '0' + pond_num
            ponds_data[pond_num]['source'] = source_str
        
        print('Loading EPA data...')
        
        ponds_list = self.ponds_list

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
        print('EPA data loaded!')
        return ponds_data 

    # helper function to check if a pond has data from prior n-days from the selected date 
    def check_active(self, pond_scorecard_data, select_date, pond_name, num_days_prior):
        def check_active_query(prev_day_n): 
            try:
                dat = pond_scorecard_data[pond_name][['Fo','Split Innoculum']].shift(prev_day_n).loc[select_date]
                if pd.isna(dat[1]) == False:
                    # check if pond is noted as "I" for inactive in the "Split Innoculum" column, 
                    # if so, break immediately and return False in the outer function (return that the particular pond is "inactive")
                    if dat[1].upper() == 'I': 
                        return 'FalseBreakImmediate'
                elif pd.isna(dat[0]) or dat[0] == 0: # if data is NaN or 0
                    return False
                else:
                    return True # 
            except:
                return False
        for n in range(0,num_days_prior+1):
            prev_n_day_active = check_active_query(n)
            if prev_n_day_active == True:
                return True
            elif prev_n_day_active == 'FalseBreakImmediate':
                return False
            elif n == num_days_prior and prev_n_day_active == False:
                return False
            else:
                pass

    
    def plot_scorecard(self, ponds_data, select_date, epa_data=None, target_to_density=0.4, target_topoff_depth=13, harvest_density=0.5, save_output=True, plot_title='Pond Health Overview'): 
        print('Plotting pond overview...')
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
#             PLACEHOLDER TO IMPLEMENT WHEN GENERATING PLOT OF CURRENT VERSUS PREV DAY CHANGE
#
#
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
        
        def subplot_ax_format(subplot_ax):
            subplot_ax.spines['top'].set_visible(False)
            subplot_ax.spines['right'].set_visible(False)
            subplot_ax.spines['bottom'].set_visible(False)
            subplot_ax.spines['left'].set_visible(False)
            subplot_ax.get_xaxis().set_ticks([])
            subplot_ax.get_yaxis().set_ticks([])
            fig.add_subplot(subplot_ax)
         
        # function to plot outlined legend boxes, using a dict of labels/data, and a separate dict of x_alignments (see examples of dict structure in calls to this function)
        def plot_legend(legend_data, x_align, y_spacing, y_align=0.9):
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

        
        # function to plot each pond to ensure that data for each plot is kept within a local scope
        def plot_each_pond(ponds_data, fig, pond_plot, pond_name, select_date):
            inner_plot = gridspec.GridSpecFromSubplotSpec(5,3,subplot_spec=pond_plot, wspace=-0.01, hspace=-0.01)

            # Check prior 5 days of data to see if pond is active/in-use
            pond_active = self.ponds_active_status[pond_name]

            try:
                single_pond_data = ponds_data[pond_name]
                pond_data_error = False
            except:
                pond_data_error = True
            
            # Gather data for pond if it's active
            if pond_active == True:
                
                # update aggregations for number of active ponds
                nonlocal num_active_ponds, num_active_ponds_sm, num_active_ponds_lg
                #check if pond is in the '06' or '08' column by checking the last values in name
                if pond_name[2:] == '06' or pond_name[2:] == '08':
                     num_active_ponds_lg += 1
                else:
                    num_active_ponds_sm += 1
                num_active_ponds += 1
                
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
                    epa_date = epa_data[pond_name]["date"]
                    epa_val = epa_data[pond_name]["epa_val"]
                    if epa_val == '':
                        try: # try to get the previous version of the epa data used
                            epa_date = list(self.epa_data_dict_old[pond_name]["epa_data"].keys())[0] # get 0 index since data should be sorted by most-recent first
                            epa_val = list(self.epa_data_dict_old[pond_name]["epa_data"].values())[0] # get 0 index since data should be sorted by most-recent first
                        except:
                            pass
                
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
                    if epa_data:
                        if epa_val == '':
                            fill_color='lightgrey'
                        elif color_idx > 0 and epa_val < 2.5:
                            fill_color = 'tan' # indicate out-of-spec EPA value
                        elif color_idx > 1 and epa_val < 3.0:
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
                    pond_data_depth = f'{round(pond_data_depth,2)}"'
                if pond_data_afdw == 0:
                    pond_data_afdw = 'No data'
                else:
                    pond_data_afdw = round(pond_data_afdw,3)
                if pond_data_harvestable_mass == 0:
                    pond_data_harvestable_mass = '-'
                else:
                    pond_data_harvestable_mass = f'{pond_data_harvestable_mass} kg' 
                if pond_data_harvestable_depth == 0:
                    pond_data_harvestable_depth = '-'
                else:
                    pond_data_harvestable_depth = f'{round(pond_data_harvestable_depth,2)}"'
                if daily_growth_rate_since_harvest == 'n/a':
                    pass # don't need to reformat in this case
                else:
                    daily_growth_rate_since_harvest = f'{daily_growth_rate_since_harvest} kg/day'
                if daily_growth_rate_5_days == 'n/a':
                    pass # don't need to reformat in this case
                else: 
                    daily_growth_rate_5_days = f'{daily_growth_rate_5_days} kg/day'
                
                data_plot_dict = {'Measurement': {'Depth:': pond_data_depth, 'AFDW:': pond_data_afdw},
                                  'Growth': {'From Last H/S:': daily_growth_rate_since_harvest,  '5 Days:': daily_growth_rate_5_days},
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
        
        total_available_to_harvest = 0 # keep track of total available for harvest across all ponds
            
        for idx_plot, pond_plot in enumerate(outer_plots):
            pond_name = flat_title_labels[idx_plot]
            if 'BLANK' not in pond_name:
                
                # plot each pond with a function to ensure that data for each is isolated within a local scope
                plot_each_pond(ponds_data, fig, pond_plot, pond_name, select_date)

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
                    
                    plot_legend(legend_data, x_align, y_spacing=0.12)
                    
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
                    
                    plot_legend(legend_data, x_align, y_spacing=0.15)
                    
                    
                elif 'BLANK 13-1' in pond_name:
                    print_string = (r'$\bf{Total\/\/Active\/\/ponds: }$' + f'{num_active_ponds}\n' 
                                    + r'$\bf{Active\/\/1.1\/\/acre\/\/ponds: }$' + f'{num_active_ponds_sm}\n'
                                    + r'$\bf{Active\/\/2.2\/\/acre\/\/ponds: }$' + f'{num_active_ponds_lg}\n'
                                   + r'$\bf{Total\/\/mass\/\/(all\/\/ponds):}$' + f'{total_mass_all:,} kg')
                    t = ax.text(0,0.75, print_string, ha='left', va='top', fontsize=16, multialignment='left')        
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
                    t = ax.text(-0.75,0.75, print_string, ha='left', va='top', fontsize=14, multialignment='left')               
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
            
            # get pond active status (True or False) from ponds_active_status dict
            pond_active = self.ponds_active_status[pond_name]
            
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
            
            if pond_active:
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
                        text_epa_data_formatted =  f'{item[1]*100: .2f}%'
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