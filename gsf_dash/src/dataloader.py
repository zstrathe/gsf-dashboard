import pandas as pd
from datetime import datetime
from office365.sharepoint.client_context import ClientContext
from os.path import getsize, isfile
import re
from . import load_setting

class Dataloader:
    def __init__(self, select_date):
        select_date = pd.to_datetime(select_date).normalize() # Normalize select_date to remove potential time data and prevent possible key errors when selecting date range from data
        ponds_list = ['0101', '0201', '0301', '0401', '0501', '0601', '0701', '0801', '0901', '1001', '1101', '1201', 
                      '0102', '0202', '0302', '0402', '0502', '0602', '0702', '0802', '0902', '1002', '1102', '1202',
                      '0103', '0203', '0303', '0403', '0503', '0603', '0703', '0803', '0903', '1003', '1103', '1203',
                      '0104', '0204', '0304', '0404', '0504', '0604', '0704', '0804', '0904', '1004', '1104', '1204',
                      '0106', '0206', '0306', '0406', '0506', '0606', '0706', '0806', '0906', '1006',
                      '0108', '0208', '0308', '0408', '0508', '0608', '0708', '0808', '0908', '1008']
        scorecard_datafile = self.download_data('scorecard_data_info')
        scorecard_dataframe = self.load_scorecard_data(scorecard_datafile, ponds_list)
        epa_datafile = self.download_data('epa_data_info')
        epa_data_dict = self.load_epa_data(epa_datafile, select_date, ponds_list)
        #self.epa_data_dict_old = self.load_epa_data_old('./data_sources/epa_data_old.xlsx', ponds_list)
        active_dict = self.generate_active_dict(scorecard_dataframe, select_date, ponds_list, num_days_prior=5)
        self.outdata = {'scorecard_dataframe': scorecard_dataframe, 'epa_dict': epa_data_dict, 'active_dict': active_dict}
 
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

    def load_scorecard_data(self, excel_filename, ponds_list):
        print('Loading scorecard data...')
        # Load the Daily Pond Scorecard excel file
        excel_sheets = pd.ExcelFile(excel_filename)

        # Get list of all sheet names
        sheetnames = sorted(excel_sheets.sheet_names)

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

    def load_epa_data(self, excel_epa_data, select_date, ponds_list) -> dict:
        def process_epa_row(sample_label, epa_val):    
            # check if sample_label is not a string  or contains the substring "Pond", if not, skip this data row
            if type(sample_label) != str or 'Pond' not in sample_label:
                return

            # search for date in sample_label with regex (looking for 6 digits surrounded by nothing else or whitespace)
            date_search = re.search(r'(?<!\S)\d{6}(?!\S)',sample_label)
            if date_search:
                date = datetime.strptime(date_search.group(), "%y%m%d")
                diff_days = (select_date-date).days
                # check if select date is before the epa value date (in which case this epa data row should be skipped to ensure that the report is using data relative to the select_date)
                # or if the epa value date is over 60 days older than the select_date, in which case it should also be skipped because the data is too old to be useful
                if diff_days < 0 or diff_days > 60:
                    #print('ERROR: sample date after 'select_date' or sample over 60 day threshold')
                    return
            else:
                #print('ERROR: no date found in sample label')
                return

            # search for pond name in sample_label with regex (looking for 4 digits surrounded by nothing else or whitespace)
            # regex ref: https://stackoverflow.com/questions/45189706/regular-expression-matching-words-between-white-space
            pondname_search = re.search(r'(?<!\S)\d{4}(?!\S)', sample_label)
            if pondname_search:
                pond_name = pondname_search.group()
            else:
                #print('ERROR: no pond name found in sample label')
                return

            # check if pond already has values for 3 dates selected, if so, then skip further processing of row
            if len(ponds_data[pond_name]) >= 3:
                return

            try:
                epa_val = float(epa_val)
            except:
                #print('ERROR: EPA val is not a valid number')
                return

            # check if pond number exists as a key in pond data, ignore this data line if not    
            if pond_name not in ponds_data:
                print('EPA KEY ERROR:', pond_name)
                return 

            # add epa values for each date (check if a key-value pair already exists for appending multiple values if necessary, to calculate an average later)
            if date not in ponds_data[pond_name]:
                ponds_data[pond_name][date] = [epa_val]
            else:
                ponds_data[pond_name][date].append(epa_val)

        print('Loading EPA data...')

        # create a dict with an empty list for each pond number (to store date and epa value)
        ponds_data = {k: {} for k in ponds_list} 

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
        
        # Iterate through ponds_data, and average the EPA values for each date (when there is more than one value per date)
        for idx, (pond_name, single_pond_data) in enumerate(ponds_data.copy().items()):
            # get the average EPA of all values for each day (or will just convert to a float for a single value)
            for idx2, (date_key, epa_vals) in enumerate(single_pond_data.items()): 
                ponds_data[pond_name][date_key] = sum(epa_vals) / len(epa_vals)
            ponds_data[pond_name] = dict(sorted(ponds_data[pond_name].items(), reverse=True)) # resort the dict just to be certain that the most-recent date is first (in case the source data isn't in correct order)
        
        print('EPA data loaded!')
        return ponds_data 

    ####### OLD FOR PREVIOUS EPA DATA FORMAT
    def load_epa_data_old(self, excel_epa_data, ponds_list, excel_pond_history="") -> dict:
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
    # checking if there is data in the 'Fo' column up to n-days (num_days_prior)
    def generate_active_dict(self, pond_scorecard_data, select_date, ponds_list, num_days_prior):
        def check_active_query(pond_name, prev_day_n): 
            try:
                dat = pond_scorecard_data[pond_name][['Fo','Split Innoculum']].shift(prev_day_n).loc[select_date]
                if pd.isna(dat[1]) == False: # condition when the 'Split Innoculum' column is not empty (i.e., not a NaN value)
                    # check if pond is noted as "I" for inactive in the "Split Innoculum" column, 
                    # if so, break immediately and return False in the outer function (return that the particular pond is "inactive")
                    if dat[1].upper() == 'I': 
                        return 'FalseBreakImmediate'
                elif pd.isna(dat[0]) or dat[0] == 0: # if 'Fo' data is NaN or 0
                    return False
                else:
                    return True  # return true since it can be assumed the dat[0] column contains a value if execution passed the prior conditional check
            except:
                return False # return false if querying the dataframe resulted in an error
        
        # iterate through the ponds_list and create a dict entry for whether that pond is active (val = True) or not (val = False)
        active_ponds_dict = {}
        for p in ponds_list:
            for n in range(0,num_days_prior+1):
                prev_n_day_active = check_active_query(p, n)
                # if pond is found to be active, set val to True and break from iterating through the range of 'num_days_prior'
                if prev_n_day_active == True:
                    active_ponds_dict[p] = True 
                    break
                # if 'FalseBreakImmediate' condition or when the last 'n' day in num_days prior is found to be inactive, then set val to False and break loop (break isn't really necessary for the second case since it should be the end of the loop anyway)
                elif prev_n_day_active == 'FalseBreakImmediate' or (n == num_days_prior and prev_n_day_active == False):
                    active_ponds_dict[p] = False
                    break
                # if the prev 'n' day is not active, but still have more days to check, then do nothing on this loop iteration
                else:
                    pass
        return active_ponds_dict
