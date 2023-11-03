import pandas as pd
from datetime import datetime
from office365.sharepoint.client_context import ClientContext
from os.path import getsize, isfile
import re
from . import load_setting, EmailHandler
import sqlite3

class Dataloader:
    _db_conn = sqlite3.connect('gsf_data.db')
    db_c = _db_conn.cursor()
    
    def __init__(self, select_date, run=True):
        self.select_date = pd.to_datetime(select_date).normalize() # Normalize select_date to remove potential time data and prevent possible key errors when selecting date range from data
        self.ponds_list = ['0101', '0201', '0301', '0401', '0501', '0601', '0701', '0801', '0901', '1001', '1101', '1201', 
                      '0102', '0202', '0302', '0402', '0502', '0602', '0702', '0802', '0902', '1002', '1102', '1202',
                      '0103', '0203', '0303', '0403', '0503', '0603', '0703', '0803', '0903', '1003', '1103', '1203',
                      '0104', '0204', '0304', '0404', '0504', '0604', '0704', '0804', '0904', '1004', '1104', '1204',
                      '0106', '0206', '0306', '0406', '0506', '0606', '0706', '0806', '0906', '1006',
                      '0108', '0208', '0308', '0408', '0508', '0608', '0708', '0808', '0908', '1008']
        self.sharepoint_connections = {} # initialize dict to store sharepoint connection for each unique site, to re-use it when downloading multiple files
        if run:
            scorecard_datafile = self.download_data('scorecard_data_info')
            scorecard_dataframe = self.load_scorecard_data(scorecard_datafile)
            
            epa_datafile1 = self.download_data('epa_data_info1')
            epa_datafile2 = self.download_data('epa_data_info2')
            epa_data_dict = self.load_epa_data([epa_datafile1, epa_datafile2])
            self.sc_df = scorecard_dataframe # TEMP FOR 
            self.epa_dict = epa_data_dict
            active_dict = self.generate_active_dict(scorecard_dataframe, num_days_prior=5)
            processing_datafile = self.download_data('processing_data_info')
            processing_dataframe = self.load_processing_data(processing_datafile)
            self.outdata = {'scorecard_dataframe': scorecard_dataframe, 'epa_data_dict': epa_data_dict, 'active_dict': active_dict, 'processing_dataframe': processing_dataframe}
    
    def download_data(self, data_item_setting):
        sharepoint_site, file_url, download_path, expected_min_filesize, print_label = load_setting(data_item_setting).values()
        if sharepoint_site not in self.sharepoint_connections.keys():
            # sharepoint auth connect
            ctx = ClientContext(sharepoint_site).with_client_certificate(**load_setting('sharepoint_cert_credentials'))
            self.sharepoint_connections[sharepoint_site] = ctx
        else: 
            ctx = self.sharepoint_connections[sharepoint_site]

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
            print(f'Successfully downloaded {print_label} to {download_path}')
            return download_path
        else:
            print(f'DOWNLOAD ERROR: {print_label}: filesize less than expected!')
            return False

    def get_data_from_email(self, email_setting: str):
        print(**load_setting(email_setting))
        return EmailHandler().get_latest_email_attachment_from_folder(**load_setting(email_setting))
            
    def load_scorecard_data(self, excel_filename):
        ponds_list = self.ponds_list
        
        print('Loading scorecard data...')
        # Load the Daily Pond Scorecard excel file
        excel_sheets = pd.ExcelFile(excel_filename)

        # create a dict containing a dataframe for each pond sheet
        all_ponds_data = {}
        for i in ponds_list:
            try:
                all_ponds_data[i] = excel_sheets.parse(i)
            except:
                print('Failed to load data for pond:', i)

        # clean the pond data (convert data from string to datetime, set date as index, drop empty columns   
        updated_ponds_data = {} # dict for storing cleaned data
        for key in enumerate(all_ponds_data.keys()):
            df = all_ponds_data[key[1]]
            df = df.rename(columns={df.columns[0]:'date'})
            df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.normalize() # convert date column from string *use .normalize method to remove potential time data
            df = df.dropna(subset=['date']) # drop rows that don't contain a date/valid date
            df = df.set_index(df['date'].name) # set date column as index
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')] # drop columns without a header, assumed to be empty or unimportant
            updated_ponds_data[key[1]] = df # add updated df to new dict of pond dataframes, which will later overwrite original 
        print('Scorecard data loaded!')
        return updated_ponds_data # return the cleaned data dict
    
    def load_processing_data(self, excel_filename):
        print('Loading processing data')
        df = pd.read_excel(excel_filename, sheet_name='SF Harvest')
        df = df.rename(columns={df.columns[0]:'date'})
        df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.normalize() # convert date column from string *use .normalize method to remove potential time data
        df = df.set_index(df['date'].name) # set date column as index
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')] # drop columns without a header, assumed to be empty or unimportant
        print('Processing data loaded!')
        return df
    
    def load_sfdata(self, excel_filename):
        print('Loading SF Data')
        df = pd.read_excel(excel_filename, sheet_name='Customer Log')
        df = df.rename(columns={df.columns[0]:'date'})
        df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.normalize() # convert date column from string *use .normalize method to remove potential time data
        df = df.set_index(df['date'].name) # set date column as index
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')] # drop columns without a header, assumed to be empty or unimportant
        print('SF data loaded!')
        return df 
    
    def load_epa_data(self, excel_epa_data_filenames: list, debug_print=False) -> dict:
        ponds_list = self.ponds_list
        select_date = self.select_date
        
        def process_excel_file(excel_filename):
            # create a dict with an empty list for each pond number (to store date and epa value)
            ponds_data_dict = {k: {} for k in ponds_list} 
            
            # load epa_data spreadsheet  
            epa_df = pd.read_excel(excel_filename, sheet_name="Sheet1", header=None)

            # check which file is being loaded (2 different sources currently)
            if 'epa_data1.xlsx' in excel_filename: # checking if this is the primary data source (from self lab testing)
                # parse the first 4 columns to update header (since excel file col headers are on different rows/merged rows)
                epa_df.iloc[0:4] = epa_df.iloc[0:4].fillna(method='bfill', axis=0) # backfill empty header rows in the first 4 header rows, results in header info copied to first row for all cols
                epa_df.iloc[0] = epa_df.iloc[0].apply(lambda x: ' '.join(x.split())) # remove extra spaces from the strings in the first column
                epa_df.columns = epa_df.iloc[0] # set header row
                epa_df = epa_df.iloc[4:] # delete the now unnecessary/duplicate rows of header data

                # process each row using Pandas vectorization and list comprehension rather than looping for better processing efficiency
                # process in reverse order to check for most-recent samples first
                [process_epa_row(a, b, ponds_data_dict) for a, b in (zip(epa_df['Sample type'][::-1], epa_df['EPA % AFDW'][::-1]))]

            elif 'epa_data2.xlsx' in excel_filename: #chck if it's the secondary data source (from third party lab)
                epa_df.columns = epa_df.iloc[0] # convert first row to column header
                epa_df = epa_df.iloc[1:] # delete the first row since it's now converted to column headers

                # process each row using Pandas vectorization and list comprehension rather than looping for better processing efficiency
                # process in reverse order to check for most-recent samples first
                [process_epa_row(a, b, ponds_data_dict, convert_val_from_decimal_to_percentage=True) for a, b in (zip(epa_df['Sample'][::-1], epa_df['C20:5 Methyl eicosapentaenoate (2734-47-6), 10%'][::-1]))]     

            # Iterate through ponds_data_dict, and average the EPA values for each date (when there is more than one value per date)
            for idx, (pond_name, single_pond_data) in enumerate(ponds_data_dict.copy().items()):
                # get the average EPA of all values for each day (or will just convert to a float for a single value)
                for idx2, (date_key, epa_vals) in enumerate(single_pond_data.items()): 
                    ponds_data_dict[pond_name][date_key] = sum(epa_vals) / len(epa_vals)
                # resort the dict just to be certain that the most-recent date is first (in case the source data isn't in correct order)
                # and trim data to last 3 most-recent data points
                ponds_data_dict[pond_name] = dict(sorted(ponds_data_dict[pond_name].items(), reverse=True)[:3]) 
            
            return ponds_data_dict 
        
        def process_epa_row(sample_label, epa_val, ponds_data_dict, convert_val_from_decimal_to_percentage=False):    
            if debug_print:
                print(sample_label, end=' | ')
        
            if type(sample_label) != str:
                if debug_print:
                    print('ERROR: sample label not a string')
                return
            
            # search for pond name in sample_label with regex (looking for 4 digits surrounded by nothing else or whitespace)
            # regex ref: https://stackoverflow.com/questions/45189706/regular-expression-matching-words-between-white-space
            pondname_search = re.search(r'(?<!\S)\d{4}(?!\S)', sample_label)
            if pondname_search:
                pond_name = pondname_search.group()
            else:
                # check for pond name with alternate data source (epa_data2.xlsx), where some of the pond names are represented with only 3 digits, missing a leading 0 (e.g., 301 - means '0301') 
                pondname_search = re.search(r'(?<!\S)\d{3}(?!\S)', sample_label)
                if pondname_search:
                    pond_name = pondname_search.group()
                    pond_name = '0' + pond_name
                else:
                    if debug_print:
                        print('ERROR: no pond name found in sample label')
                    return
            
            # search for date in sample_label with regex (looking for 6 digits surrounded by nothing else or whitespace)
            date_search = re.search(r'(?<!\S)\d{6}(?!\S)',sample_label)
            if date_search:
                date = datetime.strptime(date_search.group(), "%y%m%d")
                diff_days = (select_date-date).days
                # check if select date is before the epa value date (in which case this epa data row should be skipped to ensure that the report is using data relative to the select_date)
                # or if the epa value date is over 60 days older than the select_date, in which case it should also be skipped because the data is too old to be useful
                if diff_days < 0 or diff_days > 60:
                    if debug_print:
                        print(f'ERROR: sample date after \'{select_date}\' or sample over 60 day threshold')
                    return
            else:
                if debug_print:
                    print('ERROR: no date found in sample label')
                return

            try:
                epa_val = float(epa_val)
            except:
                if debug_print:
                    print('ERROR: EPA val is not a valid number')
                return

            # check if pond number exists as a key in pond data, ignore this data line if not    
            if pond_name not in ponds_data_dict:
                print('EPA KEY ERROR:', pond_name)
                return 
            
            # convert epa val from decimal to percentage (i.e., 0.01 -> 1.00)
            if convert_val_from_decimal_to_percentage:
                epa_val *= 100
            
            # add epa values for each date (check if a key-value pair already exists for appending multiple values if necessary, to calculate an average later)
            if date not in ponds_data_dict[pond_name]:
                ponds_data_dict[pond_name][date] = [epa_val]
            else:
                ponds_data_dict[pond_name][date].append(epa_val)
            
            if debug_print:
                print('SUCCESS', date, pond_name, epa_val)
            
        def merge_epa_data(epa_dict1, epa_dict2) -> dict:
            ''' 
            Copies EPA data from epa_dict2 into epa_dict1
            These should each already have unique values for the dict[date] key, so 
            If any duplicate data exists for epa values on any date, then take the average of the two values from each dict
            '''
            combined_dict = epa_dict1
            # add merge dict2 with dict1, checking for duplicate 'pond_name' keys, then 'date' keys within that subdict if it already exists
            for idx, (pond_name, pond_date_dict2) in enumerate(epa_dict2.items()):
                if pond_name not in epa_dict1:
                    # set nonexistent dict1 key/value equal to the dict2 key/val,
                    # since this should already be averaged by day and sorted descending by load_epa_data(), 
                    # then there is no need for further processing
                    epa_dict1[pond_name] = pond_date_dict2 
                else:
                    # append dict2 keys and values to dict1 existing values
                    # enumerate throught the subdict and check if the key (date) already exists in dict1
                    for idx, (date, epa_val2) in enumerate(pond_date_dict2.items()):
                        if date not in epa_dict1[pond_name]:
                            # add the epa_val to this date key in dict1 since it does not already exist
                            epa_dict1[pond_name][date] = epa_val2 
                        else:
                            # get the average of dict1 and dict2 values then set it as dict1 value,
                            # since each of the input dicts should already only have 1 value for each date,
                            # then can average them at this step
                            epa_val1 = epa_dict1[pond_name][date]
                            epa_dict1[pond_name][date] = (epa_val1 + epa_val2) / 2
                    # re-sort and filter to only 3 most-recent dates since data for this dict was modified
                    epa_dict1[pond_name] = dict(sorted(epa_dict1[pond_name].items(), reverse=True)[:3]) 
            return epa_dict1
             
        print('Loading EPA data...')
        tmp_list = []
        for excel_filename in excel_epa_data_filenames:
            tmp_list.append(process_excel_file(excel_filename))
        print('EPA data loaded!')
        return merge_epa_data(tmp_list[0], tmp_list[1]) 
    
    # helper function to check if a pond has data from prior n-days from the selected date 
    # checking if there is data in the 'Fo' column up to n-days (num_days_prior)
    def generate_active_dict(self, pond_scorecard_data, num_days_prior):
        ponds_list = self.ponds_list
        select_date = self.select_date
        
        def check_active_query(pond_name, prev_day_n): 
            try:
                dat = pond_scorecard_data[pond_name][['Fo', 'Split Innoculum', 'Comments']].shift(prev_day_n).loc[select_date]
                # check if pond is noted as "I" for inactive in the "Split Innoculum" column, or some variation of 'harvest complete' in the "Comments" column
                # if so, break immediately and return False in the outer function (return that the particular pond is "inactive")
                if str(dat[1]).upper() == 'I' or any(x in str(dat[2]).lower() for x in ['harvest complete', 'complete harvest', 'complete transfer']): 
                    return 'FalseBreakImmediate'
                elif pd.isna(dat[0]) or dat[0] == 0: # if 'Fo' data is NaN or 0
                    return False
                else:
                    return True  # return true since it can be assumed the dat[0] column contains a value if execution passed the prior conditional checks
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
