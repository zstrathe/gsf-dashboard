import pandas as pd
import functools
import re
#import sqlite3
import sqlalchemy
import math
from dateutil.rrule import rrule, DAILY 
from datetime import datetime, date
from pathlib import Path
from os.path import getsize, isfile
from office365.sharepoint.client_context import ClientContext
from . import load_setting, EmailHandler
from .ms_account_connect import MSAccount, M365ExcelFileHandler
#from .db_utils import init_db_table, insert_replace_row_db_table, get_db_table_columns
from .db_utils import * #get_db_table_columns, init_db_table, get_primary_keys, delete_existing_rows_ponds_data

class Dataloader:
    _db_engine = sqlalchemy.create_engine("sqlite:///db/gsf_data.db", echo=False) # initialize sqlalchemy engine / sqlite database
    account = MSAccount() # NEW CLASS FOR HANDLING MS 365 API INTERACTIONS / NEED TO UPDATE CODE TO USE THIS INSTEAD OF 'office365' MODULE
    
    def __init__(self, select_date):
        self.select_date = pd.to_datetime(select_date).normalize() # Normalize select_date to remove potential time data and prevent possible key errors when selecting date range from data
        self.ponds_list = ['0101', '0201', '0301', '0401', '0501', '0601', '0701', '0801', '0901', '1001', '1101', '1201', 
                      '0102', '0202', '0302', '0402', '0502', '0602', '0702', '0802', '0902', '1002', '1102', '1202',
                      '0103', '0203', '0303', '0403', '0503', '0603', '0703', '0803', '0903', '1003', '1103', '1203',
                      '0104', '0204', '0304', '0404', '0504', '0604', '0704', '0804', '0904', '1004', '1104', '1204',
                      '0106', '0206', '0306', '0406', '0506', '0606', '0706', '0806', '0906', '1006',
                      '0108', '0208', '0308', '0408', '0508', '0608', '0708', '0808', '0908', '1008']
        self.sharepoint_connections = {} # initialize dict to store sharepoint connection for each unique site, to re-use it when downloading multiple files
        #scorecard_datafile = self.download_data('scorecard_data_info')
        #scorecard_dataframe = self.load_scorecard_data(scorecard_datafile)
        #epa_datafile1 = self.download_data('epa_data_info1')
        #epa_datafile2 = self.download_data('epa_data_info2')
        #epa_data_dict = self.load_epa_data([epa_datafile1, epa_datafile2])
        #active_dict = self.generate_active_dict(scorecard_dataframe, num_days_prior=5)
        #processing_datafile = self.download_data('processing_data_info')
        #processing_dataframe = self.load_processing_data(processing_datafile)
        #self.outdata = {'scorecard_dataframe': scorecard_dataframe, 'epa_data_dict': epa_data_dict, 'active_dict': active_dict, 'processing_dataframe': processing_dataframe}

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

    def download_daily_data_file(self, specify_date: str = '') -> pd.ExcelFile | None:
        '''
        Find and download the "Daily Data" .xlsx file by date (yyyy-mm-dd format)
        This method looks through "Daily Data" sharepoint directory organized by: YEAR --> MM_MONTH (ex: '04-April') --> FILE ("yyyymmdd Daily Data.xlsx")

        params:
        --------
        - specify_date: - date to get daily data for (defaults to self.select_date if not specified)
                        - must be in "yyyy-mm-dd" format
            
        RETURNS -> pd.ExcelFile object (if successful download) or None (if failure)
        '''
        specify_date = self.select_date if specify_date == '' else datetime.strptime(specify_date, "%Y-%m-%d")
        folder_id = load_setting('daily_data_info')['folder_id']
        for year_dir in self.account.get_sharepoint_file_by_id(folder_id).get_items():
            # first look for the "year" directory
            if year_dir.name == str(specify_date.year): 
                # look within the year subdirectory
                for month_dir in year_dir.get_items():
                    if month_dir.name.lower() == specify_date.strftime('%m_%B').lower(): 
                        for daily_file in month_dir.get_items():
                            # search for filename with regex: case-insensitive: ["yyyymmdd" (1 or more whitespace chars) "daily data" (zero or more whitespace chars) ".xlsx"]
                            # use re.search to find 3 groups in filename: date formatted as "yyyymmdd", "daily data" (case insensitive), ".xlsx" (must be at end)
                            file_search = re.search(r"(?i)({})\s+(Daily Data).*(\.xlsx$)".format(specify_date.strftime('%y%m%d')), daily_file.name)
                            if file_search:
                                print('FOUND FILE:', daily_file.name)
                                dl_path = Path(f'data_sources/tmp/')
                                daily_file.download(to_path=dl_path)
                                dl_path = dl_path / daily_file.name
                                if dl_path.is_file():
                                    excel_file = pd.ExcelFile(dl_path.as_posix()) # load excel file
                                    dl_path.unlink() # delete file after loading
                                    return excel_file
                                else:
                                    print(f'ERROR DOWNLOADING DAILY DATA FILE FOR {datetime.strftime(specify_date, "%m/%d/%Y")}!')
                                    dl_path.unlink() # remove whatever file may be present (corrupted/partial download maybe??)
                                    return None
                        print(f'COULD NOT FIND DAILY DATA FILE FOR {datetime.strftime(specify_date, "%m/%d/%Y")}!')
                        return None

    def rebuild_daily_data_db(self, start_date: str, end_date: str, db_name: str, table_name: str = 'ponds_data') -> None:
        ''' 
        Method to fully re-build the daily data database from "from_date" to the  "to_date"
        if test_db is specified, then write to that database instead of the normal db file
        start_date: specify by "YYYY-MM-DD" - earliest possible date with data is 2017-03-15
        end_date: specify by "YYYY-MM-DD"
        '''
        db_engine = sqlalchemy.create_engine(f"sqlite:///db/{db_name}.db", echo=False) # initialize sqlalchemy engine / sqlite database
        init_db_table(db_engine, table_name) # init 'ponds_data' table (if using default param)
        init_db_table(db_engine, f'{table_name}_calculated') # init 'ponds_data_calculated' table
        init_db_table(db_engine, f'{table_name}_expenses') # init 'ponds_data_expenses' table
        
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date, "%Y-%m-%d")

        removed_cols_list = []
        for d in rrule(DAILY, dtstart=start_date, until=end_date):
            print('\n\nLoading', datetime.strftime(d, "%m/%d/%Y"))
            [removed_cols_list.append(removed_col) for removed_col in self.load_daily_data(db_engine=db_engine, specify_date=d, return_removed_cols=True) if removed_col not in removed_cols_list ]
        print('Finished building DB!!')
        print('Removed columns:', removed_cols_list)

    def load_daily_data_prev_n_days(self, prev_num_days_to_load: int, specify_date: datetime = None, **kwargs):
        # load class 'select_date' if custom parameter not provided
        if not specify_date:
            specify_date = self.select_date
        start_date = specify_date - timedelta(days=prev_num_days_to_load)
        for d in rrule(DAILY, dtstart=start_date, until=specify_date):
            self.load_daily_data(specify_date=d, **kwargs)
    
    def load_daily_data(self, specify_date: datetime, db_engine: sqlalchemy.Engine = None, table_name='ponds_data', return_removed_cols=False):  # daily_data_excel_file_path: Path
        '''
        Load the daily data file path, must be an excel file!
        Data is indexed by pond_name in multiple sheets within the excel file
        This function combines data between all sheets into single dataframe and loads into database 

        TODO: load "Comments" and "Split Innoculum" data from scorecard file
        '''
        # load class _db_engine if custom enginee parameter not provided
        if not db_engine:
            db_engine = self._db_engine
            
        excel_file = self.download_daily_data_file(specify_date=specify_date.strftime('%Y-%m-%d'))
        if excel_file == None: # file doesn't exist or download error
            # if no daily data is found for specified date, then insert blank rows into database table (one for each Pond)
            excel_dataframes = {'empty data': pd.DataFrame(self.ponds_list, columns=['PondID'])}
        else:
            print(f'Loading daily data for date: {specify_date}')
            excel_dataframes = {sheet_name: excel_file.parse(sheet_name, converters={'Pond':str,'Pond ID':str}) for sheet_name in excel_file.sheet_names} # load sheet and parse Pond label columns as strings to preserve leading zeros (otherwise i.e. 0901 would turn into 901)

        # extracted allowed columns from the DB table
        # Use these columns in the database of daily data for each pond
        # will need to rebuild the database if altering columns
        allowed_cols = get_db_table_columns(db_engine, table_name)
        used_cols = {} # keep track of used columns and the sheet that they appear on (for checking whether there is duplicate data or inaccurate data (such as same column name between sheets with different values...shouldn't happen and might indicate bad data)
        sheets_data = {}
        all_removed_cols = []
        # extract data from each sheet create a composite dataframe for each pond
        #for sheet_name in excel_sheets.sheet_names:
        for idx, (sheet_name, df) in enumerate(excel_dataframes.items()):
            print('Processing', sheet_name, '...')
            ''' Column name cleaning '''
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')] # drop columns without a header, assumed to be empty or unimportant
            # remove information in parentheses from column names (some contain UOM info like "(g/L)")...maybe extract and store in a separate table for column UOM info...but just remove for now
            # additionally, keep everything if "(filter)" is in the name since that is a way of differentiating some measurements...may potentiall need to add more exceptions and make this more robust
            tag_strip_exemptions = ('(filter)', '(filtered)') # exemptions for column tags to exempt from removing / keep these in the column name if they exist
            df.columns = [re.sub(r'\([^)]*\)', '', colname) if not any([substr in colname.lower() for substr in tag_strip_exemptions]) else colname for colname in df.columns] 
            df.columns = df.columns.str.strip() # strip leading or trailing whitespace from column names
            df = df.rename(columns={'Notes': f'Notes-{sheet_name}', 'Comments': f'Comments-{sheet_name}', 'Pond': 'PondID', 'Pond ID': 'PondID', 'AFDW (Filter)': 'Filter AFDW'}) # append sheet_name to Notes and Comments fields since those will be distinct per each sheet, and rename "Pond ID" to "Pond" if necessary

            ''' Skip sheet if there isn't a "PondID" column present
                    - Do this after cleaning column names due to renaming "Pond" and "Pond ID" columns to "PondID"
                    - might have issues if any sheets don't have the headers starting on the first row
            '''
            if not 'PondID' in df.columns:
                print('Not a valid data sheet?? Skipping...')
                continue

            ''' Clean up "PondID" column values '''
            df = df.dropna(subset='PondID') # drop rows with NaN values in the 'PondID' column
            df['PondID'] = df['PondID'].astype(str) # set Pond to string since some fields may be numerical, but need to perform string operations on this field
            df['Media Only'] = df['PondID'].apply(lambda label: True if "media only" in label.lower() else False) # extract "Media Only" field from the Pond label (for ex, might be: "1204 (Media Only)" in the source spreadsheet file)
            df['PondID'] = df['PondID'].apply(lambda val: (re.sub(r'\(media only\)', '', val, flags=re.IGNORECASE)).strip()) # remove the "Media Only" tag from pond labels if it's present
            
            ''' Fill in rows with missing PondID's '''
            [df.append({'PondID': pond_name}, ignore_index=True) for pond_name in self.ponds_list if pond_name not in df['PondID']] # Check that each Pond is listed, add it as a blank row if not

            ''' Miscellaneous Column Cleaning '''
            df['Column'] = df['PondID'].str[-1] # get column number as a string (last digit of pond name)
            df['Date'] = specify_date.strftime('%Y-%m-%d') # overwrite date column if it exists already, convert from datetime to string representation "yyyy-mm-dd"
            if 'Time Sampled' in df: # format "Time Sampled" column as a string if present (due to potential typos causing issues with some values reading as Datetime and some as strings)
                #print('BEFORE', df['Time Sampled'])
                df['Time Sampled'] = df['Time Sampled'].apply(lambda x: str(x) if not pd.isna(x) else None)
                #print('AFTER', df['Time Sampled'])
            
            ''' Set "PondID" as index column of each sheet (required for properly merging sheets) and sort the df by PondID '''
            df = df.set_index('PondID').sort_values(by='PondID')
            
            # for scorecard sheet, drop everything but the "Comments-Scorecard" column, since every other data point in this sheet is coming from another sheet via excel formula
            if sheet_name == 'Scorecard':
                df = df[['Comments-Scorecard']]
            else:
                ''' Drop columns not in allowed_cols list '''
                removed_cols = df.columns.difference(allowed_cols)
                [all_removed_cols.append(col) for col in removed_cols if col not in all_removed_cols]
                df = df[df.columns.intersection(allowed_cols)]
             
                # Handle duplicate columns and drop duplicates, but raise an exception when mismatching data is found for the same column name
                for col in df.columns:
                    #print('Checking column', col, '... col already seen?', col in used_cols)
                    if col not in used_cols:
                        used_cols[col] = sheet_name
                    else:
                        other_ = sheets_data[used_cols[col]][col][lambda x: (x!=0)&(x.isna() == False)] # get other column data (as a Pandas Series object) with 0's and NaN values removed for easier comparison
                        current_ = df[col][lambda x: (x!=0)&(x.isna() == False)] # get current column data with 0's and NaN values removed for easier comparison
    
                        ''' test different method of checking column equivalence between sheets, not requiring exact match between sheets
                        # loop through the current column index and check for equivalence of index: value pairs (should be pond names: value)
                        # do this instead of df.equals() in case of unequal length indexes between sheets (in case of maybe a singleton item of data on one sheet), if the paired columns are equal then it should be fine
                        nonmatch_flag=False
                        for idx_val in current_.index: 
                            if current_.loc[idx_val] != other_.loc[idx_val]:
                                nonmatch_flag=True
                                break
                        if nonmatch_flag == False:
                            print('Columns match!', used_cols[col], sheet_name, col)
                        else:
                            print('Column mismatch!', used_cols[col], sheet_name, col)
                        '''
                        
                        if other_.empty: # if the other column is completely empty, drop the other column and set the used_column to the current one being checked (sheet_name of the column used)
                            sheets_data[used_cols[col]] = sheets_data[used_cols[col]].drop([col], axis=1)
                            used_cols[col] = sheet_name # set the current sheet name for the "used_cols" dict which tracks which sheet a column of data is being sourced from
                        elif other_.equals(current_) or current_.empty: # if both columns are equal or current column is empty, then drop the current column
                            df = df.drop([col], axis=1)
                        else:
                            print(f'ERROR: mismatching column data between sheets: {used_cols[col]} & {sheet_name}, column: {col}')
                            print('removing shorter column...')
                            if len(other_) >= len(current_): 
                                # if the other column data is longer or same as current col, then drop the current col
                                # this will drop the current col if both sets of data are the same length (but mismatching)
                                # unfortunately there's no way to determine which data would be "correct" if they don't match but are same length, so just assume the first column processed was the correct
                                df = df.drop([col], axis=1)
                            else: # else drop the "other" column
                                sheets_data[used_cols[col]] = sheets_data[used_cols[col]].drop([col], axis=1)
                                used_cols[col] = sheet_name  # set the current sheet name for the "used_cols" dict which tracks which sheet a column of data is being sourced from
                            #raise Exception(f'ERROR: mismatching column data between sheets: {used_cols[col]} & {sheet_name}, column: {col}')
            sheets_data[sheet_name] = df

        print('All cols removed from file:', all_removed_cols)
        
        # compute a column length check value in case of bugs from handling and dropping duplicate columns 
        # raise an exception in case of an error
        all_columns = []
        [[all_columns.append(label) for label in list(df.columns)] for df in sheets_data.values()]
        unique_columns = []
        [unique_columns.append(label) for label in all_columns if label not in unique_columns]
        column_length_check_ = len(unique_columns)
        
        # use functools.reduce to iteratively merge sheets, keeping all unique columns from each sheet (outer join), join on the 'PondID' column as primary key
        joined_df = functools.reduce(lambda sheet1, sheet2: pd.merge(sheet1, sheet2, on='PondID', how='outer'), sheets_data.values())

        #print('testetstest joined_df cols', len(joined_df.columns))
        if column_length_check_ != len(joined_df.columns):
            raise Exception('ERROR WITH MERGING DAILY DATA SHEETS!')
        
        joined_df = joined_df.reset_index() #.set_index(['Date', 'PondID']) # reset index and create a multi-index (a compound primary key or whatever it's called for the SQL db)
        
        # Delete pre-existing duplicate rows from database table (so that using pd.to_sql() with 'append' mode will not add duplicated rows when re-loading data)
        print('Deleting existing rows from table...')
        delete_existing_rows_ponds_data(db_engine, table_name, joined_df)
        
        # Use DataFrame.to_sql() to insert data into database table
        joined_df.to_sql(name=table_name, con=db_engine, if_exists='append', index=False)
        print(f'Updated DB for {datetime.strftime(specify_date, "%m/%d/%Y")}!')

        # call function to perform calculations from daily data (stored in 'ponds_data_calculated' table)
        self.daily_data_calculations(check_date=specify_date, db_engine=db_engine)
        self.chemical_cost_calculations(check_date=specify_date, db_engine=db_engine)
        
        if return_removed_cols:
            return all_removed_cols
        else:
            return None
          
    def daily_data_calculations(self, check_date: datetime, db_engine: sqlalchemy.Engine | None = None, data_table_name: str = 'ponds_data', out_table_name: str = 'ponds_data_calculated') -> None:
        '''
        Method to generate calculated fields from daily data that has been loaded
          - INPUT DATA: 
              - List of:
                  Dict with keys by pond name and data (i.e., {'0101': pd.DataFrame, '0201': pd.DataFrame, ...})
        THESE CALCULATIONS SHOULD ONLY RELY ON ONE DAY OF DATA (i.e., cannot calculate growth, time-series averages, etc)

        OUTPUT:
        ---------
        database columns appended to "table_name" (default: 'ponds_data_calculated')
             - 'Date': date
             - 'PondID': pond_id 
             - 'calc_mass': mass (in kg) calculated from density and depth of pond
             - 'calc_mass_nanno_corrected': calculated mass multiplied by measured "% Nanno" measurement 
             - 'harvestable_depth_inches': harvestable depth (in inches) based on TARGET_TOPOFF_DEPTH, TARGET_DENSITY_AFTER_TOPOFF, MIN_HARVEST_DENSITY variables defined below
             - 'harvestable_gallons': harvestable_depth_inches converted into gallons 
             - 'harvestable_mass': harvestable_depth_inches converted into mass (in kg)
             - 'harvestable_mass_corrected': harvestable_mass multiplied by "% Nanno" measurement
        '''
        # helper function to convert density & depth to mass (in kilograms)
        def afdw_depth_to_mass(afdw: int|float, depth: int|float, pond_id: str):
            '''
            params: 
            - afdw: density in g/L
            - depth: depth in inches
            - pond_id: id number of pond (i.e., '0401')
            '''
            depth_to_liters = 35000 * 3.78541 # save conversion factor for depth (in inches) to liters
            # for the 6 and 8 columns, double the depth to liters conversion because they are twice the size
            if pond_id[-2:] == '06' or pond_id[-2:] == '08': 
                depth_to_liters *= 2
            # calculate and return total mass (kg)
            return round((afdw * depth_to_liters * depth) / 1000, 2)

        # load the default class database engine if not specified (should only need to be specified for testing)
        if not db_engine: 
            db_engine = self._db_engine

        check_date = check_date.strftime("%Y-%m-%d") # convert datetime to string for sql query
        data_table = load_table(db_engine, data_table_name)
        
        # variables for calculating harvestable mass 
        # based on min target density to begin harvesting, target density after harvesting and topping off with water, and target depth after topping off with water
        TARGET_TOPOFF_DEPTH = 13
        TARGET_DENSITY_AFTER_TOPOFF = 0.4
        MIN_HARVEST_DENSITY = 0.5

        calcs_df = pd.DataFrame()
        
        with db_engine.begin() as conn:
            afdw_depth_data = conn.execute(sqlalchemy.select(data_table.c["PondID", "Filter AFDW", "Depth", "% Nanno"]).where(data_table.c["Date"] == check_date)).fetchall()

        for (pond_id, afdw, depth, pct_nanno) in afdw_depth_data:
            
            if any(x in (0, None) for x in (afdw, depth)):
           # if None in (afdw, depth):
                print(f'Incomplete data, skipping {pond_id} for {check_date}')
                # add new blank row for pond_id
                calcs_df = calcs_df.append({'PondID': pond_id}, ignore_index=True)
            else:
                estimated_mass = afdw_depth_to_mass(afdw, depth, pond_id)
                
                # calculate harvestable depth of pond (in inches), rounding down to nearest 1/8 inch
                harvestable_depth_inches = math.floor((((depth * afdw) - (TARGET_TOPOFF_DEPTH * TARGET_DENSITY_AFTER_TOPOFF)) / afdw)*8)/8
                if harvestable_depth_inches < 0:
                    harvestable_depth_inches = 0
                    
                # calculate harvestable volume (in gallons) based on harvestable depth and conversion factor (35,000) to gallons. Double for the '06' and '08' column ponds since they are double size
                harvestable_gallons = harvestable_depth_inches * 35000 
                if pond_id[-2:] == '06' or pond_id[-2:] == '08':
                    harvestable_gallons *= 2

                # calculate harvestable mass using the harvestable_depth_inches calculation
                harvestable_mass = afdw_depth_to_mass(afdw, harvestable_depth_inches, pond_id)

                if type(pct_nanno) in (float, int, complex):
                    estimated_mass_nanno_corrected = round(estimated_mass * (pct_nanno/100),2)
                    harvestable_mass_nanno_corrected = round(harvestable_mass * (pct_nanno/100), 2)
                else:
                    estimated_mass_nanno_corrected = 0
                    harvestable_mass_nanno_corrected = 0

                # use dict comprehension to filter: to set <=0 values equal to None in numeric fields
                append_dict = {col_name: (value if (type(value) not in (int, float, complex) or value > 0) else None) for (col_name, value) in 
                                            {'PondID': pond_id, 
                                            'calc_mass': estimated_mass, 
                                            'calc_mass_nanno_corrected': estimated_mass_nanno_corrected, 
                                            'harvestable_depth_inches': harvestable_depth_inches, 
                                            'harvestable_gallons': harvestable_gallons, 
                                            'harvestable_mass': harvestable_mass,
                                            'harvestable_mass_nanno_corrected': harvestable_mass_nanno_corrected}.items()}
                calcs_df = calcs_df.append(append_dict, ignore_index=True)

                # # append row of calculated data to the output dataframe
                # calcs_df2 = pd.DataFrame({'PondID': pond_id, 
                #                           'calc_mass': estimated_mass, 
                #                           'calc_mass_nanno_corrected': estimated_mass_nanno_corrected, 
                #                           'harvestable_depth_inches': harvestable_depth_inches, 
                #                           'harvestable_gallons': harvestable_gallons, 
                #                           'harvestable_mass': harvestable_mass,
                #                           'harvestable_mass_nanno_corrected': harvestable_mass_nanno_corrected}, index=[0])
                # print(calcs_df2)
                # calcs_df = calcs_df.join(calcs_df2, on='PondID') # join df containing all PondID's with calculated data
                #calcs_df.loc[calcs_df['PondID'] == pond_id] = {'PondID': pond_id, 'calc_mass': estimated_mass, 'harvestable_depth_inches': harvestable_depth_inches, 'harvestable_gallons': harvestable_gallons, 'harvestable_mass': harvestable_mass}
                #calcs_df = calcs_df.append({'PondID': pond_id, 'calc_mass': estimated_mass, 'harvestable_depth_inches': harvestable_depth_inches, 'harvestable_gallons': harvestable_gallons, 'harvestable_mass': harvestable_mass}, ignore_index=True)

        # add date column to calcs_df to use for composite key (date, pond_id), use insert() method to put col at beginning 
        calcs_df.insert(loc=0, column='Date', value=check_date)
  
        # delete the existing rows with composite key of (Date, PondID), since using df.to_sql() will just append rows without any option to update
        delete_existing_rows_ponds_data(db_engine=db_engine, table_name=out_table_name, update_data_df=calcs_df)
        
        # add df to sql db
        # TODO: add foreign key to ref to "ponds_data" db???
        calcs_df.to_sql(name=out_table_name, con=db_engine, if_exists='append', index=False)

    def chemical_cost_calculations(self, check_date: datetime, db_engine: sqlalchemy.Engine | None = None, data_table_name = 'ponds_data', out_table_name: str = 'ponds_data_expenses') -> None:
        '''
        Method to calculate estimated costs from consumed chemicals and store in db_table
        '''
        # load the default class database engine if not specified (should only need to be specified for testing)
        if not db_engine: 
            db_engine = self._db_engine

        check_date = check_date.strftime("%Y-%m-%d") # convert datetime to string for sql query
        data_table = load_table(db_engine, data_table_name)

        # hardcode costs - TEMPORARY!!
        chem_costs = {
            'uan-32': {'uom': 'gal', 'cost': 2.08, 'data_column': 'Volume UAN-32 Added', 'out_column': 'uan32_cost'},
            'fertilizer-10-34': {'uom': 'gal', 'cost': 4.25, 'data_column': 'Volume 10-34 Added', 'out_column': 'fert1034_cost'},
            'bleach': {'uom': 'gal', 'cost': 3.15, 'data_column': 'Volume Bleach Added', 'out_column': 'bleach_cost'},
            #'co2': {'uom': 'lb', 'cost': 0.15, 'data_column': '', 'out_column': 'co2_cost'},  DON'T YET HAVE A METHOD OF SPLITTING CO2 COSTS PER POND / IMPLEMENT LATER
            'trace': {'uom': 'gal', 'cost': 1, 'data_column': 'Volume Trace Added', 'out_column': 'trace_cost'},
            'iron': {'uom': 'gal', 'cost': 1, 'data_column': 'Volume Iron Added', 'out_column': 'iron_cost'},
            'cal-hypo': {'uom': 'kg', 'cost': 0.78, 'data_column': 'kg Cal Hypo Added', 'out_column': 'cal_hypo_cost'},
            'benzalkonium': {'uom': 'gal', 'cost': 1, 'data_column': 'Volume Benzalkonium Added', 'out_column': 'benzalkonium_cost'}
                }

        # query the chemical usage amounts from db
        data_col_names = ['Date', 'PondID']
        [data_col_names.append(chem_costs[c]['data_column']) for c in chem_costs.keys()]
        with db_engine.begin() as conn:
            query_data = conn.execute(sqlalchemy.select(*[data_table.c[col] for col in data_col_names]).where(data_table.c["Date"] == check_date)).fetchall()
        data_df = pd.DataFrame(query_data, columns=data_col_names).fillna(0)
        
        expenses_col_names = [chem_costs[c]['out_column'] for c in chem_costs.keys()]
        cost_df = pd.DataFrame(None, columns=expenses_col_names)
        cost_df[['Date', 'PondID']] = data_df[['Date', 'PondID']]
        for (chem_name_key, vals) in chem_costs.items():
            data_column = vals['data_column']
            exp_column = vals['out_column']
            cost_df[exp_column] = data_df[data_df['PondID'] == cost_df['PondID']][data_column] * vals['cost']
                    
        # delete the existing rows with composite key of (Date, PondID), since using df.to_sql() will just append rows without any option to update
        delete_existing_rows_ponds_data(db_engine=db_engine, table_name=out_table_name, update_data_df=cost_df)

        # append df to database table
        cost_df.to_sql(name=out_table_name, con=db_engine, if_exists='append', index=False)

    def calculate_growth(pond_data_df, select_date, num_days, remove_outliers, data_count_threshold=2, weighted_stats_for_outliers=True):
        '''
        NEW **** MOVED TO DATALOADER, UTILIZE DATABASE TO QUERY/CALCULATE THIS?? ****
        
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
            ''' helper function for df.apply to label row as the next available data since a pond was last harvested. checks 3 days prior'''
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
