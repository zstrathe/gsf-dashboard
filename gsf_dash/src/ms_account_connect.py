import pandas as pd
import json
import os
import re
from typing import Type
from datetime import datetime
from O365 import Account, FileSystemTokenBackend
from O365.excel import WorkBook
from . import load_setting

class MSAccount(object):
    connection_settings = load_setting('m365_cred') 
    sharepoint_site_ids = load_setting('sharepoint_sites')
    failure_email_info = load_setting('email_failure_msg')
    
    def __init__(self, auth_manual=False):
        self.account_connection = self.load_account_user(auth_manual)
        # generate a dict of sharepoint site library IDs for each sharepoint site listed in settings config
        self.site_libs = {site_name: self.account_connection.sharepoint().get_site(site_id).get_default_document_library() for (site_name, site_id) in self.sharepoint_site_ids.items()}
    
    def load_account_user(self, auth_manual):
        '''
        Authentication with M365 through a user account
        Before a token is generated and stored, need to run this func with param 'auth_manual = True' to manually generate token and store it
        '''
        # save token for auth to re-use 
        token_backend = FileSystemTokenBackend(token_path='./settings/auth_data/',
                                               token_filename='ms_auth.token')
        account_create = Account(credentials=(self.connection_settings['client_id'], self.connection_settings['client_secret']), 
                          auth_flow_type='authorization', 
                          tenant_id=self.connection_settings['tenant'], 
                          token_backend=token_backend,
                          scopes=self.connection_settings['scopes'])
        if auth_manual:
            # run authentication which requires opening browser link to auth with current microsoft login, then pasting the redirect address into python input 
            if account_create.authenticate(): # will return True if successful authentication, False if it failed
                print('Account authenticated and token saved!')
            else:
                raise Exception('Authentication failed!')   
                    
        # check if account is validated (Account.is_authenticated doesn't seem to work as expected if token file is present but corrupted, etc)
        # so forcing a token refresh and catching when it yields an error with try/except clause
        try:
            account_create.connection.refresh_token()
        except:
            raise Exception('Account is not authenticated! Manually run "MSAccount(auth_manual=True)" to complete Microsoft authentication and store token for re-use!')
        return account_create
    
    def load_account_with_cert(self):
        def create_jwt_assertion(private_key, tenant_id, thumbprint, client_id):
            '''
            From https://github.com/O365/python-o365/blob/master/examples/jwt_assertion.py
            '''
            """
            Create a JWT assertion, used to obtain an auth token.
            @param private_key: Private key in PEM format from the certificate that was registered as credentials for the
             application.
            @param tenant_id: The directory tenant the application plans to operate against, in GUID or domain-name format.
            @param thumbprint: The X.509 certificate thumbprint.
            @param client_id: The application (client) ID that's assigned to the app.
            @return: JWT assertion to be used to obtain an auth token.
            """
            ### Imports inside function because it isn't called often
            import codecs
            import uuid
            from datetime import datetime, timezone, timedelta
            import jwt

            x5t = codecs.encode(codecs.decode(thumbprint, "hex"), "base64").replace(b"\n", b"").decode()
            aud = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"

            now = datetime.now(tz=timezone.utc)
            exp = now + timedelta(hours=1)
            jti = str(uuid.uuid4())

            payload = {
                "aud": aud,
                "exp": exp,
                "iss": client_id,
                "jti": jti,
                "nbf": now,
                "sub": client_id,
                "iat": now
            }
            headers = {
                "alg": "RS256",
                "typ": "JWT",
                "x5t": x5t,
            }
            return jwt.encode(payload, private_key, algorithm="RS256", headers=headers)
        
        tenant, client_id, thumbprint, cert_path = load_setting('sharepoint_cert_credentials').values()
        key = open(cert_path, 'r').read()
        jwt = create_jwt_assertion(private_key=key, tenant_id=tenant, thumbprint=thumbprint, client_id=client_id)
        account = Account((client_id, jwt), auth_flow_type='certificate', tenant_id=tenant)
        if not account.authenticate(scopes=[f'https://{tenant.split(".")[0]}.sharepoint.com/.default']):
            raise Exception('Authentication with certificate failed!')
        return account
    
    def interactive_view_sharepoint_data(self):
        '''
        Method for using interactively to view items in sharepoint folders
        Use to find object_id's of specific items to utilize with get_sharepoint_file_by_id()
        
        To find sharepoint site id, go to https://<tenant>.sharepoint.com/sites/<site name>/_api/site/id
        '''
        print('Sharepoint sites:')
        for (site_name, site_id) in self.sharepoint_site_ids.items():
            print(f'Site: {site_name} | ID: {site_id}')
        stop_flag = False
        while stop_flag == False:
            user_choice = input('\nEnter name of site to browse: ')
            try: 
                library = self.site_libs[user_choice]
                stop_flag = True
            except:
                print('Error loading specified sharepoint site. Try again: ')
        print(f'\n{"=*"*16} ITEMS {"*="*16}')
        for item in library.get_child_folders():
            print(f'{item} | ID: {item.object_id}')
        stop_flag = False
        while stop_flag == False:
            user_choice = input('\nEnter folder (using ID) to list files: ')
            try:
                folder = library.get_item(user_choice)
                if folder.is_folder:
                    stop_flag = True
                    print(f'\n{"=*"*16} ITEMS {"*="*16}')
                    for item in folder.get_items():
                        print(f'{item} | ID: {item.object_id}')
                        # reset flag to True if any folders are found, to continue browing through sub-folders
                        if item.is_folder:
                            stop_flag = False
                else:
                    print('Object selected is not a folder. Try again.')
            except:
                print('Error loading specified folder. Try again.')
    
    def get_sharepoint_file_by_id(self, object_id):
        ''' 
        Get a sharepoint file by looking for its object_id within each site listed in settings config file 
        (this way it's not necessary to specify which site a file is located on, just need its object_id)
        '''
        for lib in self.site_libs.values():
            try: 
                return lib.get_item(object_id)
            except:
                pass
        # send error message and return None if file wasn't found in any of the site_libs
        print(error_msg := f'ERROR: failed to get sharepoint object with ID: {object_id}')
        self._account.send_email(recipients=self._account.failure_email_info['recipients'],
                                             subject='FAILURE: loading sharepoint file', 
                                             msg_body=error_msg) 
        return None
    
    def download_sharepoint_file_by_id(self, object_id):
        return self.get_sharepoint_file_by_id(object_id).download()

    def send_email(self, recipients, subject, msg_body, attachments=[]):
        message = self.account_connection.mailbox().new_message()
        message.to.add(recipients) 
        message.subject = subject
        message.body = msg_body
        if attachments:
            for attachment_path in attachments:
                message.attachments.add(attachment_path)
        message.send()
        print('Email successfully sent!')
        
class M365ExcelFileHandler:
    def __init__(self, file_object_id: str, load_data: bool, load_sheets: list = [], ignore_sheets: list = [], dl=False):
        '''
        - Extracts data from excel workbooks with M365 API and loads into pandas dataframes 
        - Dataframes are stored in self.data, with keys set as sheet names 
        - Errors will send a notification email (for loading sheets from workbook ... if there is an error on fetching the file, then get_sharepoint_file_by_id() will send notification instead)
        
        params:
            - file_object_id: 
                    - object_id from sharepoint item (find using MSAccount.interactive_view_sharepoint_data())
            - data_handling_method: (for collecting information and raw data from file)
                    - 'api' (any case): data is only accessed through the MS Graph API, no data files are locally downloaded. This tends to be slower than just downloading the files and gathering data from there
                    - 'dl': download data locally for processing file (but actual manipulation/updating of file on O365 will be through API calls, rather than just re-uploading the file) [NEED TO TEST SPEED OF RE-UPLOAD VS API CALLS FOR SIGNIFICANT CHANGES]
            - load_sheets: 
                    - list of sheets to load (skips sheets that are present in workbook but not specified)
            - ignore_sheets:
                    - list of sheets to ignore, loads every other sheet present in the workbook
            ** load_sheets and ignore_sheets parameters are mutually exclusive (i.e., one must be empty if the other is populated)
            ** if neither load_sheets or ignore_sheets parameters are populated, then default will load all sheets

        '''
        print(f'Loading file: {file_object_id}', end='...', flush=True)
        print(f'loading data = {load_data}', end='...', flush=True)
        self._account = MSAccount() # get account connection 
        self._file_id = file_object_id
        self._wb = self._account.get_sharepoint_file_by_id(file_object_id)
        if self._wb == None: # will be None if error fetching file
            raise Exception(f'Could not load workbook for object_id: {file_object_id}')
        else:
            self._wb = WorkBook(self._wb) # get file as WorkBook obj from O365 module, for using API calls specific to excel files

        # Get the valid worksheet names from file
        self._worksheet_names_valid = [str(s).replace('Worksheet: ', '') for idx, s in enumerate(self._wb.get_worksheets())]

        self._worksheet_names_load = [] # init empty list to store worksheet names to load
        if len(load_sheets) > 0 and len(ignore_sheets) > 0:
            raise Exception('Either "load_sheets" or "ignore_sheets" parameter can be used, but not both!')
        elif len(load_sheets) > 0:
            # case when load_sheets specified
            print(f'loading sheet{"s" if len(load_sheets) > 1 else ""}: {str(*load_sheets)}', end='...', flush=True)
            [self._worksheet_names_load.append(sheet) if sheet in self._worksheet_names_valid else print(f'ERROR: worksheet "{sheet}" is not contained in file: {file_object_id}. Skipping...') for sheet in load_sheets]
        elif len(ignore_sheets) > 0:
            # case when ignore_sheets specified
            print(f'ignoring sheet{"s" if len(ignore_sheets) > 1 else ""}: {str(*ignore_sheets)}', end='...', flush=True)
            [self._worksheet_names_load.append(sheet) if sheet not in ignore_sheets else None for sheet in self._worksheet_names_valid ]
        else:
            # default case (nothing provided for load_sheets or ignore_sheets
            [self._worksheet_names_load.append(sheet) for sheet in self._worksheet_names_valid]

        print('\n')
        # collect data for sheets
        if dl == False:
            self.sheet_data = {sname: self._get_sheet_data_api(sname, _get_df=load_data) for sname in self._worksheet_names_load}
        else:
            self.sheet_data = {sname: self._get_sheet_data_dl(sname) for sname in self._worksheet_names_load}
                               
        print('File loaded!')

    def close_wb_session(self):
        '''
        Close the workbook session with the MS Graph API (creating a session defaults to persistant session with the python O365 module)
        '''
        self._wb.session.close_session()
    
    def _get_sheet_data_dl(self, sheet_name):
        '''
        Get worksheet data by downloading it instead of using API - faster for large files, or maybe all files
        store self.downloaded_workbook
        '''
        print("TEST: LOADING DATA VIA DOWNLOAD!!")
        if not hasattr(self, 'downloaded_workbook_file'):
            dl_file_path = self._account.download_sharepoint_file_by_id(self._file_id)
            self.downloaded_workbook_file = pd.ExcelFile(dl_file_path)
        try:
            df = self.downloaded_workbook_file.parse(sheet_name)
        except:
            raise Exception(f'ERROR: could not get "{sheet_name}" from file id: {self._file_id}')
        
        return_data = object() # empty container to store return data
        return_data.name = sheet_name
        return_data.df = df
        #TODO return_data.used_range = 
        #TODO return_data.max_row = 
        #TODO return_data.max_col = 
        #TODO return_data.column_to_name_map =
        return return_data
        
    def _get_sheet_data_api(self, sheet_name: str, _get_df: bool = False):
        '''
        Sheet variables: 
            - sheet: O365.excel.WorkBook.WorkSheet object
            - sheet.name: worksheet name
            - sheet.used_range: range of data within the sheet
            - sheet.max_row: maxiumum row containing data within the sheet
            - sheet.max_col: maxiumum column containing data within the sheet
            - sheet.column_to_name_map: mapping of column letters to names (i.e., {'A': 'Column 1', 'B': 'Column 2'}
            - sheet.df **OPTIONAL**: sheet data represented as a Pandas DataFrame
        TODO: 
            - handle leading empty columns
            - column to datatype mapping???
            - handle sheets with weird headers (multiple rows, etc)???
        '''
        def test_scan_for_dt(sheet, nrows=25):
            scan_range = f'A1:{sheet.max_col}{nrows}'
            scan_data = sheet.get_range(scan_range)
            scan_values = scan_data.values
            scan_text = scan_data.text
            output = {'dt_cols': [], 'dt_problem_cols': []}
            for col_idx, col_name in enumerate(scan_values[0]):
                print(f'\nChecking column: {col_idx}')
                col_dt_count = 0
                col_dt_problem_count = 0 
                for row_idx in range(1, len(scan_values)):
                    print(f'Checking row: {row_idx}')
                    # if the "value" representation of the cell is different from the "text" representation
                    # AND if the value is an integer, then it might be a datetime value
                    test_value = scan_values[row_idx][col_idx]
                    test_text = scan_text[row_idx][col_idx]
                    try: 
                        test_text_value = float(test_text)
                    except:
                        ### DO SOMETHING HERE???????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
                        pass
                    print('Value & Text equal:', str(test_value) == test_text)
                    if str(test_value) == test_text:
                        print('...breaking...')
                        break
                    elif test_value == float(test_text):
                        ### DO SOMETHING HERE???????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
                        pass
                    else:
                        # check if the value can be converted into a date
                        from xlrd import xldate_as_datetime
                        dt_val = xldate_as_datetime(test_value, datemode=False)
                        if not isinstance(dt_val, datetime):
                            # stop enumerating rows if any non-datetime values are found 
                            break 
                        elif (dt_val.year < datetime.now().year - 100 or dt_val.year > datetime.now().year + 100):
                            print(f'Invalid found: {test_value}, {test_text}, {dt_val}')
                            # track if any invalid datetime values are found (with invalid meaning that the values convert to datetime correctly but are out of the +-100 year of current year range, so probably not actually a datetime value)
                            # and keep track of those columns separately from datetime columns (so that values are fetched as the .text version for this column)
                            col_dt_problem_count += 1  
                        else:
                            print(dt_val)
                            print('test for valid dt', not (dt_val.year < datetime.now().year - 100 or dt_val.year > datetime.now().year + 100), f'value: {dt_val.year} (range: {datetime.now().year - 100} - {datetime.now().year + 100})')  
                            col_dt_count += 1
                if col_dt_count == nrows - 1:
                    print(f'Found datetime column on sheet: {sheet.name}, column: {col_idx}')
                    output['dt_cols'].append(col_idx) # add column index to output (will only get to this point if all values checked in column are valid datetime)
                elif col_dt_problem_count == nrows -1:
                    print(f'Found PROBLEMATIC datetime column on sheet: {sheet.name}, column: {col_idx}')
                    output['dt_problem_cols'].append(col_idx) 
            return output
                    
        sheet = self._wb.get_worksheet(sheet_name)
        sheet.name = sheet_name
        sheet.used_range, sheet.max_row, sheet.max_col = self._sheet_get_used_range_fixed(sheet)
        if sheet.used_range != None:
            _column_headers = sheet.get_range(f'A1:{sheet.max_col}1').values[0]
            sheet.column_to_name_map = {self._convert_numeric_col_to_alphabetic(idx):col_name for (idx, col_name) in enumerate(_column_headers, start=1) if col_name != ''}
            sheet.dt_cols = test_scan_for_dt(sheet)
            if _get_df:
               # sheet.df = pd.DataFrame(sheet.get_range(sheet.used_range).values)
                sheet.df = self._load_excel_sheet_to_df(sheet)
        # return every variable in locals() except those that are preceeded by a "_" character
        return sheet
      
    def _sheet_get_used_range_fixed(self, sheet: Type['O365.excel.WorkSheet']):
        ###### TODO contribute to O365 project to fix 'valuesOnly' parameter and filtering queries???
        url = sheet.build_url(sheet._endpoints.get('get_used_range'))
        url += '(valuesOnly=true)?$select=address'
        response = sheet.session.get(url)
        # get range address (extracted from range object), and formatted to remove the sheet name (i,e. "Sheet1!A1:Z343" turns into just "A1:Z343")
        full_range_str = sheet.range_constructor(parent=sheet, **{sheet._cloud_data_key: response.json()}).address #.split("!")[-1]  
        #print('Loading', full_range_str)
        trunc_range_str = full_range_str.split("!")[-1] 
        max_range_str = trunc_range_str.split(':')[-1] # get the sheet coordinate extents (i.e., for "A1:Z335", would extract "Z335")
        max_row = re.findall(r'\d+', max_range_str) # returns a list
        max_col = re.findall(r'[a-zA-z]+', max_range_str)
        if not (len(max_row)==1 and len(max_col) == 1): # check that range extents are valid (should only be a single group of numbers/alphabetic chars for each variable)
            print(f'Error loading sheet "{sheet.name}". Skipping...')
            full_range_str = trunc_range_str = max_range_str = max_row = max_col = None
            #  try:
            #max_range_str, max_row, max_col = self._find_last_row_brute_method(self._file_id, sheet) 
          #  except:
         #       raise Exception('ERROR: sheet range is not valid:', full_range_str)
        else:
           max_row = int(max_row[0])
           max_col = max_col[0]
       # print('Loaded...')
        return trunc_range_str, max_row, max_col

    def _find_last_row_brute_method(self, file_object_id: str, sheet: Type['O365.excel.WorkSheet']):
        '''
        Get the last row through repeatedly requesting one row at a time from the MS Graph API
        *** THIS METHOD IS SLOW DUE TO REPEATED API REQUESTS FOR EACH SHEET, USE AS A LAST RESORT ONLY (should only be necessary for massive sheets that break the API) ***
        First, look at one row at a time, and exponentially increase the row number looking at, until an empty row is found
        Second, once an empty row is found, then iteratively check mid-point rows between row_search_low (the highest nonempty row found at each iteration), 
            and row_search_high (the upper bound for searching), until the number  of rows in the search bound is less than 1000.
        Once the search range is narrowed to <1000 rows, then query that entire range and search locally to find the last row (this saves some additional
        queries which slightly speeds up each function call by 0.5 - 1 second)

        params:
            - file_object_id: 
                    - used to save structural data about file to json format (structure: file_id.json --> {sheetnames_dict: {struct_data_dict: 
                                                                                                                                - {'max_row': last known max row number (currently this is the only k:v pair stored)}}}
            - sheet: 
                    - the O365.excel.WorkSheet object
        
        Important:
            - default to start searching at row 150 to give a bias to smaller datasets (which if < 150 rows, will immediately query and search those rows directly, saving need for additional queries)
            - otherwise start at previously recorded last row number for workbook.sheet loaded from json file

        TODO: Add double check to confirm last row when found. Check n+5, n+10 or something similar to confirm those are still empty. Otherwise could accidentally populate an intentional gap left between rows.
        '''
        # helper functions for loading and writing json of 
        def _load_json(file_object_id: str, sheet_name: str):        
            if os.path.exists(file_path := os.path.join('settings/json_data', f'{file_object_id}.json')):
                with open(file_path, 'r') as file:
                    dat = json.load(file)
                    print(f'Loaded json for file: {file_object_id}')
                    return dat.get(sheet_name, {})
            else:
                return {}

        def _write_json(file_object_id: str, sheet_name: str, out_data: dict):
            with open(os.path.join('settings/json_data', f'{file_object_id}.json'), "w") as outfile:
                json_out_data = json.load(outfile)
                # update dict keys that have been modified           
                for (k, v) in out_data.items():
                    json_out_data[k] = v           
                json.dump(json_out_data, outfile)
        
        print('Loading sheet with backup brute method...')
        json_data = _load_json(file_object_id, sheet.name)
        
        # start searching at the 'max_row' from json data, plus 150 (if it doesn't exist, will start at 150 instead)
        # additional 150 rows added as a buffer to be certain that the API request will include all relevant rows
        row_search_low = json_data.get('max_row', 0)
        row_search_high = row_search_low + 150
        step_count = 0
        empty_bound_found = False
        col_search_bound = self._convert_numeric_col_to_alphabetic(500) # search with 500 columns
        col_low_tracker = 0 # var to track the lowest column that contains data
        col_high_tracker = 0 # var to keep track of the highest column that contains data 

        while True:
            step_count += 1
            print(step_count)
            
            # set max row for excel365
            row_search_high = [1048576 if row_search_high > 1048576 else row_search_high][0] 
            
            # Query the row to search
            range_query = f'A{row_search_high}:{col_search_bound}{row_search_high}'
       
            check_row = sheet.get_range(range_query).values[0]
           
            empty_row = True # flag to track whether the row is empty
            for idx, cell in enumerate(check_row, start=1):
                if cell != '':
                    if idx > col_high_tracker:
                        col_high_tracker = idx
                    if idx < col_low_tracker:
                        col_low_tracker = idx
                    empty_row = False
                    # if nonempty cells are found and have not found an empty row yet, jump ahead to double the current search row
                    ## TODO add a growth rate instead...slow growth while still speeding up to not overshoot initial target by too much?
                    if empty_bound_found == False:
                        row_search_low = row_search_high
                        row_search_high = row_search_high * 5
                    #if empty rows have been found, then only jump ahead by the diff between high and low vals
                    else:
                        row_search_low_new = row_search_high
                        row_search_high = ((row_search_high - row_search_low)/2).__ceil__() + row_search_high
                        row_search_low = row_search_low_new
                    break
            # if no data found in the row
            if empty_row:
                # If an empty row has been found, and the difference between search points is less than 1000, then get all rows in this range /
                # to find last row slightly quicker (approx 0.5 - 1 second) due to reduced API queries needed
                if row_search_high - row_search_low < 1000:
                    final_check_bound = f'A{row_search_low}:{col_search_bound}{row_search_high}'
                    final_check_query = WorkBook_WorkSheet_obj.get_range(final_check_bound).values
                    final_empty_row_count = 0 
                    for row_idx, row in enumerate(final_check_query, start=1):
                        for val_idx, val in enumerate(row, start=1):
                            if val != '':
                                if val_idx > col_high_tracker:
                                    col_high_tracker = val_idx
                                if val_idx < col_low_tracker:
                                    col_low_tracker = val_idx
                                final_empty_row_count = 0 
                                break # immediately break and begin checking next row if any nonempty val found
                            elif val_idx == len(row) and val == '':
                                final_empty_row_count += 1
                                print(final_empty_row_count)
                                if final_empty_row_count == 1:
                                    final_first_empty_row = row_idx
                                if final_empty_row_count >= 5: # 
                                   # print('Find last row took steps (api queries):', step_count)
                                    max_row = final_first_empty_row-1+row_search_low
                                    min_col = self._convert_numeric_col_to_alphabetic(col_low_tracker)
                                    max_col = self._convert_numeric_col_to_alphabetic(col_high_tracker)
                                    _write_json(file_object_id, sheet.name, {'max_row': max_row})
                                    return f'{min_col}1:{max_col}{max_row}', max_row, max_col
                else:
                    empty_bound_found = True
                    # since an empty bound has been found, then backtrack high bound by half the difference bwtween high and low bounds /
                    # to gradually narrow the search band for the last nonempty row
                    row_search_high = ((row_search_high - row_search_low)/2).__floor__() + row_search_low
    
    def _load_excel_sheet_to_df(self, sheet: Type['O365.excel.WorkSheet']):
        '''
        - Loads a M365 sheet into a dataframe by querying data from API
        - For very large sheets (exceeding 3,500,000 cells), then queries are split up, then collected and combined sequentially

        ERROR: does not correctly convert datetime values
        '''
        # Get the last non-empty column letter 
        # BASED FIRST ROW DATA ONLY - generated from _get_column_mapping() which checks 5000 columns in 1st row
        if len(sheet.column_to_name_map) > 0:
            last_col_letter = list(sheet.column_to_name_map.keys())[-1]
        else: 
            last_col_letter = 'A'

        ''' 
        MS API claims queries are limited to 5,000,000 cells, so split queries up as necessary for very large data sets
        In reality, query limit seems to be 3,500,000 cells, so using that instead
        '''
        # calculate row limit (determined by number of columns)
        row_query_limit = (3500000 / len(list(sheet.column_to_name_map.keys()))).__floor__()
        last_row_number = sheet.max_row

        data_vals = [] # list to store data values, in case multiple queries need to be combined together
        if last_row_number > row_query_limit:
            query_ranges = [(i, i+row_query_limit-1) if i+row_query_limit-1 < last_row_number else (i, last_row_number) for i in range(1, last_row_number+1, row_query_limit)]
           # print('ERROR too many queries!!', query_ranges)
            for (ql, qh) in query_ranges:
                qrange = f'A{ql}:{last_col_letter}{qh}'
                #print('TESTTESTTESTTEST', qrange)
                data_vals.extend(sheet.get_range(qrange).values)
        else:
            data_vals.extend(sheet.get_range(f'A1:{last_col_letter}{last_row_number}').values)
            
        # Return pandas dataframe from query to data, using 'last_col_letter' and 'last_row_number' as limits
        return pd.DataFrame(data_vals[1:], columns=data_vals[0])
        
    # def _get_column_mapping_api(self, sheet: Type['O365.excel.WorkSheet']):
    #     '''
    #     Get column mapping (col number to alphabetic, and alphabetic to col number) for by checking for data in the first row
    #     Checks 5000 columns...maybe excessive
    #     '''
    #     col_to_name_map = {self._convert_numeric_col_to_alphabetic(idx):col_name for (idx, col_name) in enumerate(sheet.get_range(f'A1:{self._convert_numeric_col_to_alphabetic(5000)}1').values[0], start=1) if col_name != ''}
    #     return col_to_name_map
     
    def _convert_numeric_col_to_alphabetic(self, col_number):
        '''
        Function to convert numeric column number (starting from 1) to Excel naming convention for columns.
        For example, 4 converts to 'D', 27 converts to 'AA', or 261 converts to 'JA', etc.

        input: int
        output: string

        The logic with this function is to find the number of times the column number is wholly divisible by 26, and the remainder of the division.
        The remainder will be equal to the last character in the output string (or appended to the left of prior steps). Then on the following loop, 
        set the 'col_number' value equal to the number of times the number was divisible on the last step. If this value being checked is less than 26, 
        then it will be the final step of the loop.
        For example: starting with 56
                        - First, since 56 is greater than 26, check how many times it is divisible by 26 
                            - Find that 56 is divisible by 26: 2.1538 times
                            - So 56 is wholly divisible by 26: 2 times
                              and the remainder is: 56 - (2*26) = 4
                            - With the remainder calculated, the last character for the output string is 4 : "D"
                        - Then on the next loop, the max divisor (2) is used for finding the next character.
                            - Since 2 is less than 26, this will be the final loop iteration
                            - Simply find that 2 converted to alphabetic is "B"
                            - Append "B" to left of the existing string "D": 
                                    returns "BD"        
        '''
        to_alpha = lambda x: chr(x+64) # function to convert a 1-26 number (int) into a corresponding alphabetic character
        out_str = ''
        while col_number > 0:
            if col_number > 26:
                # add a very small negative "bias" when calculating maximum divisor to prevent numbers 
                # that are perfectly divisible by 26 from having a zero remainder
                max_divisor_by_26 = ((col_number / 26)-0.001).__floor__()
                remainder = col_number - (max_divisor_by_26 * 26)
                out_str = to_alpha(remainder) + out_str
                col_number = max_divisor_by_26
            else:
                out_str = to_alpha(col_number) + out_str
                col_number = 0 # stop the loop
        return out_str