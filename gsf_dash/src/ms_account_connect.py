import pandas as pd
import json
import os
import re
import functools
from pathlib import Path
from typing import Type
from datetime import datetime
from O365 import Account, FileSystemTokenBackend
from O365.excel import WorkBook
from .utils import load_setting

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
        for item in library.get_items():
            print(f'{item} | ID: {item.object_id}')
        stop_flag = False
        while stop_flag == False:
            if (user_choice := input('\nEnter folder (using ID) to list files, or "q" to quit: ')).lower() == 'q':
                return None
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
        self.send_email(recipients=self.failure_email_info['recipients'],
                                             subject='FAILURE: loading sharepoint file', 
                                             msg_body=error_msg) 
        return None
    
    def download_sharepoint_file_by_id(self, object_id: str, to_path: Path, **kwargs) -> Path|None:
        file_obj = self.get_sharepoint_file_by_id(object_id)
        dl_success = file_obj.download(to_path=to_path, **kwargs) # returns True if success, False if failure
        if dl_success:
            # if 'name' argument was passed to file_obj.download(), then the file will have that name
            # otherwise, use the file's actual name from the download source
            if not (file_name := kwargs.get('name')):
                file_name = file_obj.name
            file_path = to_path / file_name

            # check that the downloaded file is a valid file 
            # for very very low chance of file being corrupted, etc
            if os.path.isfile(file_path):
                return file_path
            else: 
                file_path.unlink() # delete bad file
                return None
        else:
            return None

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
    def __init__(self, file_object_id: str, include_sheets: list = [], ignore_sheets: list = [], load_data: bool = False, data_get_method: str = "DL", concat_sheets: bool = False, load_sheet_kwargs: dict = {}):
        '''
        - Extracts data from excel workbooks with M365 API and loads into pandas dataframes 
        - Dataframes are stored in self.data, with keys set as sheet names 
        - Errors will send a notification email (for loading sheets from workbook ... if there is an error on fetching the file, then get_sharepoint_file_by_id() will send notification instead)
        
        params:
            - file_object_id: 
                    - object_id from sharepoint item (find using MSAccount.interactive_view_sharepoint_data())
            - include_sheets: 
                    - list of sheets to load (skips sheets that are present in workbook but not specified)
            - ignore_sheets:
                    - list of sheets to ignore, loads every other sheet present in the workbook
                    - OPTIONAL: include "SUBSTRING[string]" to filter out sheet names that contain a substring
            - load_data: 
                    - True: load the sheet data into 
            - data_get_method: (for collecting information and raw data from file)
                    - 'API' (any case): data is only accessed through the MS Graph API, no data files are locally downloaded. This tends to be slower than just downloading the files and gathering data from there
                    - 'DL': download data locally for processing file (but actual manipulation/updating of file on O365 will be through API calls??)
            
            - load_last_n_rows: bool: load the most recent "n" rows from the file if an int is specified; otherwise, loads all rows
            ** include_sheets and ignore_sheets parameters are mutually exclusive (i.e., one must be empty if the other is populated)
            ** if neither include_sheets or ignore_sheets parameters are populated, then default will load all sheets
        '''
        print(f'Loading file: {file_object_id}', end='...', flush=True)
        print(f'loading data = {load_data}', end='...', flush=True)
        self._account = MSAccount() # get account connection 
        self._file_id = file_object_id
        self._file_obj = self._account.get_sharepoint_file_by_id(file_object_id)
        if self._file_obj == None: # will be None if error fetching file
            raise Exception(f'Could not load workbook for object_id: {file_object_id}')
        else:
            self._wb = WorkBook(self._file_obj) # get file as WorkBook obj from O365 module, for using API calls specific to excel files

        # Get the valid worksheet names from file
        self._worksheet_names_valid = [str(s).replace('Worksheet: ', '') for idx, s in enumerate(self._wb.get_worksheets())]

        self._worksheet_names_load = [] # init empty list to store worksheet names to load
        if len(include_sheets) > 0 and len(ignore_sheets) > 0:
            raise Exception('Either "include_sheets" or "ignore_sheets" parameter can be used, but not both!')
        elif len(include_sheets) > 0:
            # case when include_sheets specified
            print(f'loading sheet{"s" if len(include_sheets) > 1 else ""}: {str(*include_sheets)}', end='...', flush=True)
            [self._worksheet_names_load.append(sheet) if sheet in self._worksheet_names_valid else print(f'ERROR: worksheet "{sheet}" is not contained in file: {file_object_id}. Skipping...') for sheet in include_sheets]
        elif len(ignore_sheets) > 0:
            # case when ignore_sheets specified
            print(f'ignoring sheet{"s" if len(ignore_sheets) > 1 else ""}: {*ignore_sheets,}', end='...', flush=True) ## KEEP COMMA AFTER UNPACKED LIST IN F-STRING!! REMOVING IT WILL RESULT IN A SYNTAX ERROR
            #[self._worksheet_names_load.append(sheet) if sheet not in ignore_sheets else None for sheet in self._worksheet_names_valid ]
            [self._worksheet_names_load.append(sheet) for sheet in self._worksheet_names_valid ] # populate a list of all sheets to load, remove the sheets to ignore in next lines
            for sheet_name in self._worksheet_names_load.copy():
                for ignore_name in ignore_sheets:
                    substring_search = re.search(r'SUBSTRING\[(.+)\]', ignore_name)
                    if substring_search:
                        search_str = substring_search.group(1)
                        if search_str in sheet_name:
                            self._worksheet_names_load.remove(sheet_name)
                            break
                    else:
                        if sheet_name == ignore_name:
                            self._worksheet_names_load.remove(sheet_name)
                            break
        else:
            # default case (nothing provided for include_sheets or ignore_sheets
            [self._worksheet_names_load.append(sheet) for sheet in self._worksheet_names_valid]

        print('\n')

        # download data file if data_get_method = "DL"
        # download at this step versus when parsing data for each sheet
        if load_data and data_get_method == 'DL':
            dl_file_path = self._account.download_sharepoint_file_by_id(self._file_id, to_path=Path(f'data_sources/tmp/'))
            if dl_file_path: # will be None if error downloading
                self._downloaded_ExcelFile = pd.ExcelFile(dl_file_path.as_posix())
                dl_file_path.unlink() # delete file after it's loaded
            else:
                raise Exception(f'Download error for file_id: {self._file_id}!')

        # collect data for sheets
        self.sheets_data = {sname: self.get_sheet_data(sname, load_data, data_get_method, **load_sheet_kwargs) for sname in self._worksheet_names_load}

        # concatenate sheets into a single dataframe if concat_sheets == True and load_data == True
        if load_data and concat_sheets:
            self.concat_df = functools.reduce(lambda df1, df2: pd.concat([df1, df2], ignore_index=True), [sheet.df for sheet in self.sheets_data.values()])
        else:
            self.concat_df = None

        # Close the workbook session with the MS Graph API (creating a session defaults to persistant session with the python O365 module)
        self._wb.session.close_session()
        
        print('File loaded!')

    def get_sheet_data(self, sheet_name: str, load_data: bool, data_get_method: str, **kwargs):
        '''
        Params:
            - sheet_name: str: the specific sheet name to load, must match sheet name in Excel file
            - data_get_method: "API" or "DL", defaults to DL
            - load_data: True or False: whether to load the data into a Pandas dataframe; otherwise, will just get general info about the sheet used range
            
        Sheet variables to return: 
            - sheet: O365.excel.WorkBook.WorkSheet object
            - sheet.name: worksheet name
            - sheet.used_range: range of data within the sheet
            - sheet.max_row: maxiumum row containing data within the sheet
            - sheet.max_col: maxiumum column containing data within the sheet
            - sheet.column_to_name_map: mapping of column letters to names (i.e., {'A': 'Column 1', 'B': 'Column 2'}
            - sheet.df: sheet data represented as a Pandas DataFrame (will be None if load_data = False)
        TODO: 
            - handle leading empty columns
            - column to datatype mapping???
            - handle sheets with weird headers (multiple rows, etc)???
        '''             
        sheet = self._wb.get_worksheet(sheet_name)
        sheet.name = sheet_name
        sheet.used_range, sheet.max_row, sheet.max_col, sheet.min_row, sheet.min_col = self._sheet_get_used_range_fixed(sheet)
        if load_data:
            if data_get_method.upper() == 'DL':
                sheet.df = self._load_excel_sheet_to_df_download(sheet, **kwargs)
            elif data_get_method.upper() == 'API':
                sheet.df = self._load_excel_sheet_to_df_api(sheet, **kwargs)
            else:
                raise Exception(f'Invalid "get_method" specified for loading sheet: {sheet_name}!')
        else:
            sheet.df = None # set df attribute to None if not loading the sheet data
        return sheet

    def _load_excel_sheet_to_df_download(self, sheet: Type['O365.excel.WorkSheet'], **kwargs) -> pd.DataFrame:
        df = self._downloaded_ExcelFile.parse(sheet.name, **kwargs)
        print(f'loaded df for {sheet.name}...')
        return df
    
    def _load_excel_sheet_to_df_api(self, sheet: Type['O365.excel.WorkSheet'], start_row: int|None = None, end_row: int|None = None) -> pd.DataFrame:
        '''
        - NOTE: VERY SLOW, PRETTY MUCH USELESS VERSUS DOWNLOADING AND PROCESSING FILE LOCALLY
        - Loads a M365 sheet into a dataframe by querying data from API
        - For very large sheets (exceeding 3,500,000 cells), then queries are split up, then collected and combined sequentially

        params:
            - sheet: O365.excel.WorkSheet object that has been processed to have sheet.name, sheet.column_to_name_map, sheet.max_row, sheet.max_col, sheet.min_row, sheet.min_col, and sheet.dt_cols properties)
            - start_row (optional): the start row to extract data from (defaults to first row: 2 by Excel naming convention (with row 1 being header row that is always loaded))
            - end_row (optional): the end row to extract data from (defaults to the last row of data)
        ERROR: does not correctly convert datetime values
        '''
        if sheet.used_range != None:        
            _column_headers = sheet.get_range(f'{sheet.min_col}{sheet.min_row}:{sheet.max_col}{sheet.min_row}').values[0]
            sheet.column_to_name_map = {self._convert_numeric_col_to_alphabetic(idx):col_name for (idx, col_name) in enumerate(_column_headers, start=1) if col_name != ''}
            
        # get column headers from row 1 (actually sheet.min_row, but should be row #1 most of the time) in excel file
        _column_headers = sheet.get_range(f'{sheet.min_col}{sheet.min_row}:{sheet.max_col}{sheet.min_row}').values[0]

        if not start_row:
            start_row = int(sheet.min_row) + 1
        if not end_row:
            end_row = int(sheet.max_row)
        
        ''' 
        MS API documentation claims that queries are limited to 5,000,000 cells, so split queries up as necessary for very large data sets
        In reality, query limit seems to be 3,500,000 cells, so using that instead
        '''
        # calculate row limit (determined by number of columns)
        row_query_limit = (3500000 / len(list(sheet.column_to_name_map.keys()))).__floor__()
        
        data_vals = [] # list to store data values, in case multiple queries need to be combined together
        if end_row > row_query_limit:
            query_ranges = [(i, i+row_query_limit-1) if i+row_query_limit-1 < end_row else (i, end_row) for i in range(start_row, end_row+1, row_query_limit)]
            print('Too many queries, splitting up API data requests!!', query_ranges)
            for (ql, qh) in query_ranges:
                qrange = f'{sheet.min_col}{ql}:{sheet.max_col}{qh}'
                data_vals.extend(sheet.get_range(qrange).values)
        else:
            data_vals.extend(sheet.get_range(f'{sheet.min_col}{start_row}:{sheet.max_col}{end_row}').values)

        # Return pandas dataframe from query to data
        df = pd.DataFrame(data_vals, columns=_column_headers)

        # Find and convert any datetime columns before returning dataframe
        # starting from row 1 (after column headers), every row through row 25 must convert into a valid date field 
        # with year between 2017 and 2055, to be considered a valid datetime column
        non_dt_cols = []
        dt_cols = []
        if not len(df) > 25:
            print(f'Sheet {sheet.name} too short, skipping dt conversion for now...')
        else:
            for row_idx in range(1,26):
                for col_idx, cell in enumerate(df.iloc[row_idx,:]):
                    if col_idx in non_dt_cols:
                        pass # pass on subsequent rows if a non-datetime column has been identified on a prior checked row
                    # check if cell is an integer, then check if it can be converted into valid date with year being between 2017 and 2055
                    if type(cell) == int:
                        try:
                            if pd.to_datetime(cell, unit='d', origin='1899-12-30').year in (range(2017,2056)):
                                if col_idx not in dt_cols:
                                    dt_cols.append(col_idx)
                        except:
                            if col_idx not in non_dt_cols:
                                non_dt_cols.append(col_idx)
                            if col_idx in dt_cols:
                                dt_cols.pop(col_idx)
                    else:
                        if col_idx not in non_dt_cols:
                            non_dt_cols.append(col_idx)
                        if col_idx in dt_cols:
                            dt_cols.pop(col_idx)
            
            for col in dt_cols:
                # use .apply rather than applying to_datetime to the entire column at once
                # do this to handle each row separately & pass on rows that don't conform
                # (otherwise errors could prevent from applying to entire column at once)
                # df.iloc[:, col] = df.iloc[:, col].apply(lambda x: convert_dt_or_pass(x))
                df.iloc[:, col] = pd.to_datetime(df.iloc[:, col], unit='d', origin='1899-12-30', errors='coerce')
                
        return df
        
    def _sheet_get_used_range_fixed(self, sheet: Type['O365.excel.WorkSheet']):
        ###### TODO contribute to O365 project to fix 'valuesOnly' parameter and filtering queries???
        url = sheet.build_url(sheet._endpoints.get('get_used_range'))
        url += '(valuesOnly=true)?$select=address'
        response = sheet.session.get(url)
        # get range address (extracted from range object), and formatted to remove the sheet name (i,e. "Sheet1!A1:Z343" turns into just "A1:Z343")
        full_range_str = sheet.range_constructor(parent=sheet, **{sheet._cloud_data_key: response.json()}).address #.split("!")[-1]  
        #print('Loading', full_range_str)
        trunc_range_str = full_range_str.split("!")[-1] 
        
        min_range_str = trunc_range_str.split(':')[0] # get the sheet coordinate start (i.e., for "A1:Z335", would extract "A1")
        min_row = re.findall(r'\d+', min_range_str)[0] # returns as a list, get first item, should be the only one
        min_col = re.findall(r'[a-zA-z]+', min_range_str)[0] 
        
        max_range_str = trunc_range_str.split(':')[-1] # get the sheet coordinate extents (i.e., for "A1:Z335", would extract "Z335")
        max_row = re.findall(r'\d+', max_range_str)[0] # returns as a list, get first item, should be the only one
        max_col = re.findall(r'[a-zA-z]+', max_range_str)[0]
        
        return trunc_range_str, max_row, max_col, min_row, min_col
    
    def _find_last_row_brute_method(self, file_object_id: str, sheet: Type['O365.excel.WorkSheet']):
        '''
        Get the last row through repeatedly requesting one row at a time from the MS Graph API
        *** THIS METHOD IS SLOW DUE TO REPEATED API REQUESTS FOR EACH SHEET, USE AS A LAST RESORT ONLY (should only be necessary for massive sheets that otherwise break the API) ***
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

class EmailHandler:
    data_attachment_types = ['xlsx', 'xls', 'xlsm']
    
    def __init__(self):
        self.account = MSAccount().account_connection
    
    def send_email(self, recipients, subject, msg_body, attachments=[]):
        message = self.account.mailbox().new_message()
        message.to.add(recipients) 
        message.subject = subject
        message.body = msg_body
        if attachments:
            for attachment_path in attachments:
                message.attachments.add(attachment_path)
        message.send()
        print('Email sent successfully!')
  
    def get_latest_email_attachment_from_folder(self, folder_name, save_filename, dl_attachment=True):
        '''
        This method retrieves the most-recent email attachment from a specified email folder. If the email contains an attachment, then 
        if the attachment is on the permission list (self.data_attachment_types), then it is downloaded to ./data_sources/ and the file path is then returned.
        
        If the attachment is an email file (and attachment_type will be a 'file'), then self.extract_attachemt_from_email_file will be called, 
        and the first valid attachment will be saved to ./data_sources/ and its file path returned.
        
        NOTE: a potential shortcoming of this method is that it will return only the first valid attachment of the most-recent email. Therefore it must be modified if there 
        is ever a need to extract multiple attachments from an email, or needs to look through multiple emails, etc
        '''
        fldr = self.account.mailbox().get_folder(folder_name=folder_name)
        latest_message = fldr.get_message(query="", download_attachments=True) 
        attachments = latest_message.attachments
        if len(attachments) > 0:
            for a in attachments:
                a_type = a.attachment_type
                # check if the attachment type is an 'item' (aka hopefully a .eml file) or a 'file' (aka hopefully an excel file, but could be a pdf or anything else)
                # there is potential for error, where an 'item' is a different attachment such as a meeting invite but self.extract_attachment_from_email_file should 
                # still be able to extract files attached to those
                if a_type == 'item':
                    print('Email (.eml) attachment found...', end=' ', flush=True)
                    print(f'Saving as {a.name}.eml')
                    tmpfile_path = f'data_sources/tmp/{a.name}.eml'
                    latest_message.attachments.save_as_eml(a, to_path=tmpfile_path)
                    return self.extract_attachment_from_email_file(tmpfile_path, save_filename)
                elif a_type == 'file':
                    print('Non-email attachment found')
                    print(a.name)
                    if a.name.split('.')[-1] in self.data_attachment_types:
                        print('GOOD ATTACHMENT TYPE!')
                        a.save(location=f'data_sources/', custom_name=save_filename)
                        return f'data_sources/{a.name}'
                    else:
                        print('BAD ATTACHMENT TYPE! Skipping...')
                        pass
                        
        else:
            print('Did not find any file attachments!')
            
    def extract_attachment_from_email_file(self, email_filename, save_filename):
        from email import message_from_file
        msg = message_from_file(open(email_filename))
        attachments=msg.get_payload()
        for attachment in attachments:
            fnam = attachment.get_filename()
            f_extension = fnam.split('.')[-1]
            if fnam != None and f_extension in self.data_attachment_types: 
                print(f'Saving {fnam}...as {save_filename}.{f_extension}')
                save_fnam= f'data_sources/{save_filename}.{f_extension}'
                with open(save_fnam, 'wb') as f:
                    f.write(attachment.get_payload(decode=True))
                print(f'Successfully saved to {save_fnam}!')
                return save_fnam
            else:
                print(f'Skipping {fnam}, not a good filetype!')