from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import functools
import re
import sqlalchemy
import math
from dateutil.rrule import rrule, DAILY 
from datetime import datetime, date
from pathlib import Path
from . import load_setting 
from .ms_account_connect import MSAccount, M365ExcelFileHandler
from .db_utils import * 
from .utils import redirect_logging_to_file

class Dataloader:
    def __init__(self, run_date: str|None, db_engine_name: str = 'gsf_data'):
        '''
        params:
            - run_date: 
                - date formatted as str (yyyy-mm-dd) or None
                - if not None, will process as a DAILY db update 
            - db_engine: sql alchemy engine name (str)
        '''
        self.db_engine_name = db_engine_name
        self.db_engine = sqlalchemy.create_engine(f"sqlite:///db/{db_engine_name}.db", echo=False) # initialize sqlalchemy engine / sqlite database
        
        if run_date:
            self.run_date = pd.to_datetime(run_date).normalize() # Normalize run_date to remove potential time data and prevent possible key errors when selecting date range from data
            self.main_run_each_proc(run_date_start=run_date, run_date_end=run_date) # run method to load data on a daily basis

    def main_run_each_proc(self, run_date_start: datetime, run_date_end: datetime) -> None:
        '''
        Method to tie together daily processes to run for downloading/loading/processing data and loading into a db table
        '''
        param_kwargs = {'db_engine': self.db_engine, 
                        'run_date_start': run_date_start, 
                        'run_date_end': run_date_end}

        # init queue of subclasses of DBColumnsBase, which should all be ETL processes to extract & load data for one or more columns in the DB
        # TODO: currently each process calls a method to update the corresponding rows in the DB table specified, need to collect data (by table) from each process and minimize DB update calls???
        data_etl_queue = DBColumnsBase.__subclasses__()
        completed_etl_name_list = [] # init empty list to store string names of classes successfully run
        reset_queue_counter = {p.__name__: 0 for p in data_etl_queue} # init a dict to count how many times a specific process cannot be run due to a dependency (raise Exception if > 1)
        
        while data_etl_queue:
            data_etl_class_obj = data_etl_queue.pop(0) # get first item in queue & remove it
            class_vars = vars(data_etl_class_obj)
            if 'DEPENDENCY_CLASSES' in class_vars:
                dep_classes = class_vars['DEPENDENCY_CLASSES'] # list of class names (strings)
                reset_queue_flag = False
                for c in dep_classes:
                    if not c in completed_etl_name_list:
                        print('Detected dependency class needed, re-appending to end of processing queue!')
                        reset_queue_counter[data_etl_class_obj.__name__] += 1
                        if reset_queue_counter[data_etl_class_obj.__name__] > 1:
                            raise Exception(f'ERROR: could not run process name: [{data_etl_class_obj.__name__}] for run date: [{run_date.strftime("%Y-%m-%d")}], due to dependency process: [{c}] not existing??')
                        data_etl_queue.append(data_etl_class_obj)
                        reset_queue_flag = True
                        break
                if reset_queue_flag:
                    continue
            print('Running:', data_etl_class_obj.__name__) 
            data_etl_class_obj(**param_kwargs)
            completed_etl_name_list.append(data_etl_class_obj.__name__)
       
        print('Finished with daily data updates to DB!')

    def rebuild_db(self, start_date: str, end_date: str) -> None:
        ''' 
        Method to fully re-build the daily data database from "start_date" to the  "end_date"
       
        start_date: specify by "YYYY-MM-DD" - earliest possible date with data is 2017-03-15
        end_date: specify by "YYYY-MM-DD"
        '''
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date, "%Y-%m-%d")

        # initialize database tables from CREATE TABLE statements stored in ./settings/db_schemas/
        # could just let pd.to_sql() create the tables, but this allows additional flexibility with setting contstraints, etc.
        def init_tables():
            init_db_table(self.db_engine, 'ponds_data') 
            init_db_table(self.db_engine, 'ponds_data_calculated') 
            init_db_table(self.db_engine, 'ponds_data_expenses') 
            init_db_table(self.db_engine, 'epa_data') 
            init_db_table(self.db_engine, 'agg_daily')

        # use decorator from .utils.py to redirect stdout to a log file, only while building the DB
        @redirect_logging_to_file(log_file_directory=Path(f'db/logs/'), 
                                            log_file_name=f'{date.today().strftime("%y%m%d")}_rebuild_db_{self.db_engine_name}_{start_date.strftime("%y%m%d")}_{end_date.strftime("%y%m%d")}.log')
        def rebuild_db_run(start_date, end_date):
            #removed_cols_list = []
            # for d in rrule(DAILY, dtstart=start_date, until=end_date):
                # print('\n\nLoading', datetime.strftime(d, "%m/%d/%Y"))
                ##########[removed_cols_list.append(removed_col) for removed_col in self.load_daily_data(db_engine=db_engine, specify_date=d, return_removed_cols=True) if removed_col not in removed_cols_list ]
                ## TODO: re-implement tracking of all columns removed from files processed
            self.main_run_each_proc(run_date_start=start_date, run_date_end=end_date)
            print('Finished building DB!!')
            #print('Removed columns:', removed_cols_list)
        
        init_tables()
        rebuild_db_run(start_date, end_date)

class DBColumnsBase(ABC):
    ponds_list = ['0101', '0201', '0301', '0401', '0501', '0601', '0701', '0801', '0901', '1001', '1101', '1201', 
                  '0102', '0202', '0302', '0402', '0502', '0602', '0702', '0802', '0902', '1002', '1102', '1202',
                  '0103', '0203', '0303', '0403', '0503', '0603', '0703', '0803', '0903', '1003', '1103', '1203',
                  '0104', '0204', '0304', '0404', '0504', '0604', '0704', '0804', '0904', '1004', '1104', '1204',
                  '0106', '0206', '0306', '0406', '0506', '0606', '0706', '0806', '0906', '1006',
                  '0108', '0208', '0308', '0408', '0508', '0608', '0708', '0808', '0908', '1008']
    account = MSAccount() # CLASS FOR HANDLING MS 365 API INTERACTIONS 
    
    def __init__(self, db_engine: sqlalchemy.Engine, run_date_start: datetime, run_date_end: datetime, run: bool = True):
        '''
        params:
            db_engine: sqlalchemy Engine
            run_date_start: datetime
            run_date_end: datetime
            run: FOR TESTING - True by default
        '''
        if not hasattr(self, 'OUT_TABLE_NAME'):
            raise Exception(f"ERROR: attempted to create a DB Table class without first setting OUT_TABLE_NAME for {self.__class__}!")

        self.db_engine = db_engine    
        self.run_date_start = run_date_start
        self.run_date_end = run_date_end
        run_range_days = (self.run_date_end - self.run_date_start).days
        if run_range_days < self.MIN_LOOKBACK_DAYS:
            self.run_date_start = self.run_date_start - pd.Timedelta(days=self.MIN_LOOKBACK_DAYS-run_range_days)
            
        if run:
            self.run()

    # set MIN_LOOKBACK_DAYS property
    # this is the minimum days that will be included in each data class update 
    # i.e., when running once daily, run_date_start == run_date_end, so update for additional dates
    @property
    def MIN_LOOKBACK_DAYS(self) -> int:
        if not hasattr(self, "_MIN_LOOKBACK_DAYS"):
            self._MIN_LOOKBACK_DAYS = 5
        return self._MIN_LOOKBACK_DAYS

    @MIN_LOOKBACK_DAYS.setter
    def MIN_LOOKBACK_DAYS(self, val: int) -> None:
        self._MIN_LOOKBACK_DAYS = val
        
    @abstractmethod
    def run():
        pass
        '''
        Extract, transform, and load data into database table (self.OUT_TABLE_NAME)
        '''

class DailyDataLoad(DBColumnsBase):
    ''' 
    Load primary ponds data into 'ponds_data' table 
    '''
    OUT_TABLE_NAME = 'ponds_data'
    
    def __init__(self, *args, **kwargs):
        self.OUT_TABLE_NAME = 'ponds_data'
        
        # init as base class after setting OUT_TABLE_NAME
        super().__init__(*args, **kwargs) 

    def run(self):
        daily_data_dfs = []
        # Start with a "base_df" to ensure that a row every Date & PondID combination is included in the db, even if no data is found
        date_range = pd.date_range(self.run_date_start, self.run_date_end) # .map(lambda x: x.strftime('%Y-%m-%d')
        lp_date, lp_pondid = pd.core.reshape.util.cartesian_product([date_range, self.ponds_list])
        base_df = pd.DataFrame(list(zip(lp_date, lp_pondid)), columns=['Date', 'PondID'])
        daily_data_dfs.append(base_df)

        # Call method to process and load into db the daily data files 
        daily_data_dfs.append(self._load_daily_data_date_range(dt_start=self.run_date_start, dt_end=self.run_date_end))
        
        # Combine "base" and "daily data" dfs
        joined_daily_df = functools.reduce(lambda df1, df2: pd.merge(df1, df2, on=['Date','PondID'], how='outer'), daily_data_dfs)

        # update the db (calling a function from .db_utils.py that deletes the existing db table row (based on Date&PondID cols) and appends the new row from the update_data_df)
        update_table_rows_from_df(db_engine=self.db_engine, db_table_name=self.OUT_TABLE_NAME, update_data_df=joined_daily_df)

    def _load_daily_data_date_range(self, dt_start: datetime, dt_end: datetime):
        day_dfs = []
        for idx, d in enumerate(rrule(DAILY, dtstart=dt_start, until=dt_end)):
            day_dfs.append(self._load_daily_data_file(specify_date=d))
        out_df = pd.concat(day_dfs, axis=0, join='outer') 
        return out_df

    def _load_daily_data_file(self, specify_date: datetime, return_removed_cols=False):  # daily_data_excel_file_path: Path
        '''
        Load the daily data file path, must be an excel file!
        Data is indexed by pond_name in multiple sheets within the excel file
        This function combines data between all sheets into single dataframe and loads into database 
        '''  
        excel_file = self._download_daily_data_file(specify_date=specify_date.strftime('%Y-%m-%d'))
        if excel_file == None: # file doesn't exist or download error
            # if no daily data is found for specified date, then insert blank rows into database table (one for each Pond)
            excel_dataframes = {'empty data': pd.DataFrame(self.ponds_list, columns=['PondID'])}
        else:
            print(f'Loading daily data for date: {specify_date}')
            excel_dataframes = {sheet_name: excel_file.parse(sheet_name, converters={'Pond':str,'Pond ID':str}) for sheet_name in excel_file.sheet_names} # load sheet and parse Pond label columns as strings to preserve leading zeros (otherwise i.e. 0901 would turn into 901)

        # extracted allowed columns from the DB table
        # Use these columns in the database of daily data for each pond
        # will need to rebuild the database if altering columns
        allowed_cols = get_db_table_columns(self.db_engine, self.OUT_TABLE_NAME)
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
            df['Date'] = specify_date #.strftime('%Y-%m-%d') # overwrite date column if it exists already, convert from datetime to string representation "yyyy-mm-dd"
            if 'Time Sampled' in df: # format "Time Sampled" column as a string if present (due to potential typos causing issues with some values reading as Datetime and some as strings)
                df['Time Sampled'] = df['Time Sampled'].apply(lambda x: str(x) if not pd.isna(x) else None)
            if 'Filter AFDW' in df:
                df['Filter AFDW'] = df['Filter AFDW'].replace(0, None) # replace zero values in AFDW column with None, as zero values may cause issues with other calculated fields 
            
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

        if column_length_check_ != len(joined_df.columns):
            raise Exception('ERROR WITH MERGING DAILY DATA SHEETS!')
        
        joined_df = joined_df.reset_index() #.set_index(['Date', 'PondID']) # reset index and create a multi-index (a compound primary key or whatever it's called for the SQL db)
        return joined_df
        
    def _download_daily_data_file(self, specify_date: str = '') -> pd.ExcelFile | None:
        '''
        Find and download the "Daily Data" .xlsx file by date (yyyy-mm-dd format)
        This method looks through "Daily Data" sharepoint directory organized by: YEAR --> MM_MONTH (ex: '04-April') --> FILE ("yyyymmdd Daily Data.xlsx")

        params:
        --------
        - specify_date: - date to get daily data for (defaults to self.run_date if not specified)
                        - must be in "yyyy-mm-dd" format
            
        RETURNS -> pd.ExcelFile object (if successful download) or None (if failure)
        '''
        specify_date = self.run_date if specify_date == '' else datetime.strptime(specify_date, "%Y-%m-%d")
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
                                dl_path_dir = Path(f'data_sources/tmp/')
                                dl_file_path = dl_path_dir / daily_file.name
                                download_attempts = 5
                                while True:
                                    daily_file.download(to_path=dl_path_dir)
                                    if dl_file_path.is_file():
                                        excel_file = pd.ExcelFile(dl_file_path.as_posix()) # load excel file
                                        dl_file_path.unlink() # delete file after loading
                                        return excel_file
                                    else:
                                        download_attempts -= 1
                                        if download_attempts > 0:
                                            print('Waiting 5 seconds to try download again...')
                                            time.sleep(5)
                                            continue
                                        else:
                                            raise Exception(f'ERROR: located but could not download daily data file for {datetime.strftime(specify_date, "%m/%d/%Y")}!')
                                        
                        print(f'COULD NOT FIND DAILY DATA FILE FOR {datetime.strftime(specify_date, "%m/%d/%Y")}!')
                        return None

class ScorecardDataLoad(DBColumnsBase):
    ''' 
    Load scorecard data into 'ponds_data' table 
    '''
    OUT_TABLE_NAME = 'ponds_data'
    
    def __init__(self, *args, **kwargs):
    
        # init as base class 
        super().__init__(*args, **kwargs)

    def run(self):
         # Get data from scorecard file and add/update the database table for those entries ("Comments-Scorecard" & "Split Innoculum" [to indicate when a pond is harvested or split])
        # Reload data for past 5 days (in case any of it changed) 
        sc_df = self._load_scorecard_data(begin_date=self.run_date_start, end_date=self.run_date_end)

        # update the db 
        update_table_rows_from_df(db_engine=self.db_engine, db_table_name=self.OUT_TABLE_NAME, update_data_df=sc_df)
        
    def _load_scorecard_data(self, begin_date: datetime, end_date: datetime):
        # save loaded file as a class attribute in case of using multiple successive calls to this method
        if not hasattr(self, '_sc_df'):
            print('Loading scorecard file...')
            file_id = load_setting('file_ids').get('scorecard')
            sc_file = M365ExcelFileHandler(file_object_id=file_id, load_data=True, data_get_method='DL', ignore_sheets=['Analysis', 'Template', 'Notes', 'SUBSTRING[(old)]', 'SUBSTRING[HRP]'])
    
            include_columns = ['Date', 'Comments', 'Split Innoculum']
            all_df_list = []
            for pond_id in self.ponds_list:
                try: 
                    pond_df = sc_file.sheets_data.get(pond_id).df
                except:
                    print(f'No scorecard data available for PondID: {pond_id}. Skipping...')
                    continue
                pond_df.columns.values[0] = 'Date' # rename the first column to date, since it's always the pond id listed here on these sheets (but it's the date index col)
                pond_df['Date'] = pd.to_datetime(pond_df['Date'], errors='coerce') # datetime values should already exist, but this converts any errors into a NaT value with errors='coerce'
                pond_df = pond_df.dropna(subset=['Date']) # drop na values from Date column (after forcing error values into a NaT value)                                        
                #pond_df = pond_df[pond_df['Date'].between(begin_date, end_date)] # filter data by Date, get only data between begin_date and end_date
                #pond_df['Date'] = pond_df['Date'].apply(lambda x: x.strftime("%Y-%m-%d")) # convert Date into a string representation for SQL queries
                pond_df = pond_df[include_columns] # filter columns
                pond_df['PondID'] = pond_id # add PondID column for DB table indexing
                pond_df = pond_df.rename(columns={'Comments': 'Comments-ScorecardFile'}) # rename Comments column to be consistent with DB
                all_df_list.append(pond_df)
            ScorecardDataLoad._sc_df = pd.concat(all_df_list, axis=0) # combine dataframes for each PondID into one large df (combine by rows)
        
        out_df = self._sc_df.copy()
        out_df = out_df[out_df['Date'].between(begin_date, end_date)]
        return out_df

class CO2ConsumptionLoad(DBColumnsBase):
    '''
    Load co2 consumption data into 'agg_daily' table
    '''
    OUT_TABLE_NAME = 'agg_daily'
    
    def __init__(self, *args, **kwargs):
        # init as base class 
        super().__init__(*args, **kwargs) 

    def run(self):
        if not hasattr(self, '_co2_data'):
            CO2ConsumptionLoad._co2_data = self._load_data_file()

        # get copy of df from class attr and filter dates
        co2_df = self._co2_data.copy()
        co2_df = co2_df[co2_df['Date'].between(self.run_date_start, self.run_date_end)]
        
        # update the db 
        update_table_rows_from_df(db_engine=self.db_engine, db_table_name=self.OUT_TABLE_NAME, update_data_df=co2_df, pk_cols=['Date'])

    def _load_data_file(self) -> pd.DataFrame:
        print('Loading co2 consumption file...')
        file_id = load_setting('file_ids').get('co2_consumption')
        co2_df = M365ExcelFileHandler(file_object_id=file_id, load_data=True, data_get_method='DL', ignore_sheets=['2017', 'Troubleshooting'], concat_sheets=True).concat_df
        # filter columns to just Date and Daily Consumption (lbs)
        co2_df = co2_df[['Date', 'Daily Consumption (lbs)']].dropna()
        co2_df = co2_df.rename(columns={'Daily Consumption (lbs)': 'total_co2_consumption'})
        return co2_df  
        
class EPALoad(DBColumnsBase):
    OUT_TABLE_NAME = 'epa_data'
    OUT_COLUMNS = ['epa_val', 'epa_val_total_fa', 'epa_actual_measurement']
    DEPENDENCY_CLASSES = ['GetActiveStatus'] # define class dependency that needs to be run prior to an instance of this class
    MIN_LOOKBACK_DAYS = 15 # override default daily update lookback days (from 5 to 15)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def run(self):
        ## TODO: check if there's data available from M365 api to check if file has been updated?? then just store local file and re-download when necessary??
        if not hasattr(self, '_epa_df'):
            EPALoad._epa_df = self._download_and_merge_epa_data()
        epa_df = self._epa_df.copy()

        # filter epa_df based on lookback_days parameter value (lookback_days is days in addition to the run_date, so num of rows returned will be lookback_days+1
        epa_df = epa_df[epa_df['Date'].between(self.run_date_start, self.run_date_end)]

        # update the db (calling a function from .db_utils.py)
        update_table_rows_from_df(db_engine=self.db_engine, db_table_name=self.OUT_TABLE_NAME, update_data_df=epa_df)
        
    def _download_and_merge_epa_data(self) -> pd.DataFrame:
        print('Loading EPA data...')
        # download excel files and get downloaded file paths (pathlib.Path) as a list
        epa_data_file_paths = self._dl_epa_data()

        # process excel files (extract Date, PondID, 'epa_val' and 'epa_fa_val' as columns from file)
        tmp_list_of_dataframes = []
        [tmp_list_of_dataframes.append(self._process_excel_file(excel_filename)) for excel_filename in epa_data_file_paths] # process_excel_file() returns a pandas DataFrame 
        
        # concat dataframes (using functools.reduce so number of files can vary)
        merged_epa_df = functools.reduce(lambda df1, df2: pd.concat([df1, df2], ignore_index=True), tmp_list_of_dataframes)
        
        # group by Date and PondID to get averaged EPA values (when more than one value exists for a pond/date), then reset_index() to reset grouping back to a normal df
        merged_epa_df = merged_epa_df.groupby(by=['Date', 'PondID']).mean().reset_index().sort_values(by=['Date', 'PondID'])

        merged_epa_df['epa_actual_measurement'] = 'X' # add a column to keep track of "actual" epa_values (days when measured) versus days in-betweeen measurements that are filled in

        # construct a DF with combination of all possible Date and PondID columns for date range according to earliest and latest dates for EPA data
        epa_date_range = pd.date_range(start=merged_epa_df['Date'].min(), end=self.run_date_end)
        lp_date, lp_pondid = pd.core.reshape.util.cartesian_product([epa_date_range, self.ponds_list])
        dates_df = pd.DataFrame(list(zip(lp_date, lp_pondid)), columns=['Date', 'PondID'])

        # merge the dates_df with epa data df; using a left join so if any epa data is invalid (either date or PondID) then it will not be included in the merge, 
        # and days with missing data will just be empty (except for Date and PondID fields)
        out_df = pd.merge(dates_df, merged_epa_df, on=['Date', 'PondID'], how='left')

        # get list of all columns that should be forward filled by checking exclusion list
        fill_columns = [col for col in out_df.columns if col not in ('Date', 'PondID', 'epa_actual_measurement')]

        # query dates when ponds were completely emptied / harvested completely, so to use as hard "reset points" in epa_data when forward filling in for dates in between readings
        active_pond_dates_df = query_data_table_by_date_range(db_name_or_engine=self.db_engine, 
                                                                table_name='ponds_data_calculated', 
                                                                query_date_start=out_df['Date'].min(), # use the first date from the epa data / this will likely error out, but query should get first available data with "check_safe_date=True" 
                                                                query_date_end=self.run_date_end,
                                                                col_names=['active_status'],
                                                                check_safe_date=True) # Date and PondID automatically included

        out_df = pd.merge(out_df, active_pond_dates_df, on=['Date', 'PondID'], how='left')
        out_df.loc[out_df['active_status'] == False, fill_columns] = 'n.a.'
        
        # forward fill epa data to empty rows (dates between measurements) for active ponds
        for pond_id in out_df['PondID'].unique():
            # get temp copy of df filtered by PondID
            tmp_pond_id_df = out_df[out_df['PondID'] == pond_id].copy()
            tmp_pond_id_df[fill_columns] = tmp_pond_id_df[fill_columns].ffill()
            out_df.loc[out_df['PondID'] == pond_id, fill_columns] = tmp_pond_id_df[fill_columns]

        out_df = out_df.replace('n.a.', None)
        out_df = out_df.drop('active_status', axis=1)
        
        return out_df

    def _dl_epa_data(self) -> list[Path]:
        # get file IDs from settings file, get only the values from the dict that's returned
        epa_data_file_ids = {k:v for (k, v) in load_setting('file_ids').items() if 'epa_data' in k}

        epa_file_paths = []
        for (label, file_id) in epa_data_file_ids.items():
            tmp_file_path = self.account.download_sharepoint_file_by_id(object_id=file_id, name=f'{label}.xlsx', to_path=Path('data_sources/tmp/')) # returns file path, else returns None if failed
            if tmp_file_path is not None:
                epa_file_paths.append(tmp_file_path)
            else:
                raise Exception(f'ERROR: failure downloading EPA data for {label}: {file_id}!')
        return epa_file_paths

    def _process_excel_file(self, excel_filename) -> pd.DataFrame:
        # load epa_data spreadsheet  
        epa_df = pd.read_excel(excel_filename, sheet_name="Sheet1", header=None)

        # set up an empty dict to store kwargs for options of processing rows in each data file
        row_process_kwargs = {}
        
        # check which file is being loaded (2 different sources currently)
        if 'epa_data_primary.xlsx' in excel_filename.__repr__(): # checking if this is the primary data source 
            print('Processing primary epa file')
            # parse the first 4 columns to update header (since excel file col headers are on different rows/merged rows)
            epa_df.iloc[0:4] = epa_df.iloc[0:4].fillna(method='bfill', axis=0) # backfill empty header rows in the first 4 header rows, results in header info copied to first row for all cols
            epa_df.iloc[0] = epa_df.iloc[0].apply(lambda x: ' '.join(x.split())) # remove extra spaces from the strings in the first column
            epa_df.columns = epa_df.iloc[0] # set header row
            epa_df = epa_df.iloc[4:] # delete the now unnecessary/duplicate rows of header data
            epa_df = epa_df[['Sample type', 'EPA % AFDW', 'EPA % of Total FA']]
            
        elif 'epa_data_secondary.xlsx' in excel_filename.__repr__(): #chck if it's the secondary data source
            # not currently using a secondary source
            pass
        else:
            print('Filename wrong (expected "epa_data_primary.xlsx" or "epa_data_secondary.xlsx"...Wrong file provided?')
            return pd.DataFrame() # return empty dataframe, as next concatenation steps expect dfs as input

        # use .apply() to generate new df for output, applying self._process_epa_row() to each row, with row_process_kwargs containing optional args
        out_epa_df = pd.DataFrame()
        out_epa_df[['Date', 'PondID', 'epa_val', 'epa_val_total_fa']] = epa_df.apply(lambda x: self._process_epa_row(x[0], x[1], x[2], **row_process_kwargs), axis=1, result_type='expand')
        
        # drop n/a rows (rows containing N/A values will result from using epa_df.apply(), and self._process_epa_row() returning None for rows with invalid data)
        out_epa_df = out_epa_df.dropna() 

        return out_epa_df
        
    def _process_epa_row(self, sample_label, epa_val, epa_val_total_fa, debug_print=False, convert_val_from_decimal_to_percentage=False) -> tuple:  
        return_vals = {'date': None, 
                       'pond_id': None, 
                       'epa_val': None, 
                       'epa_val_total_fa': None}
        return_on_error = [None] * len(return_vals)
        if debug_print:
            print('Printing output of row processing for debugging!')
            print(sample_label, end=' | ')
    
        if type(sample_label) != str:
            if debug_print:
                print('ERROR: sample label not a string')
            return return_on_error
        
        # search for pond name in sample_label with regex (looking for 4 digits surrounded by nothing else or whitespace)
        # regex ref: https://stackoverflow.com/questions/45189706/regular-expression-matching-words-between-white-space
        pondname_search = re.search(r'(?<!\S)\d{4}(?!\S)', sample_label)
        if pondname_search:
            return_vals['pond_id'] = pondname_search.group()
        else:
            # check for pond name with alternate data source (epa_data2.xlsx), where some of the pond names are represented with only 3 digits, missing a leading 0 (e.g., 301 - means '0301') 
            pondname_search = re.search(r'(?<!\S)\d{3}(?!\S)', sample_label)
            if pondname_search:
                tmp_pond_id = pondname_search.group()
                return_vals['pond_id'] = '0' + tmp_pond_id
            else:
                if debug_print:
                    print('ERROR: no pond name found in sample label')
                return return_on_error

        # check if pond id number exists as a key in master ponds list, ignore this data line if not    
        if return_vals['pond_id'] not in self.ponds_list:
            print('EPA KEY ERROR:', return_vals['pond_id'])
            return return_on_error
        
        # search for date in sample_label with regex (looking for 6 digits surrounded by nothing else or whitespace)
        date_search = re.search(r'(?<!\S)\d{6}(?!\S)',sample_label)
        if date_search:
            return_vals['date'] = datetime.strptime(date_search.group(), "%y%m%d")
        else:
            if debug_print:
                print('ERROR: no date found in sample label')
            return return_on_error

        try:
            # return None if epa_val is zero (but not for epa_val_total_fa as it is secondary)
            if epa_val == 0:
                return return_on_error
            return_vals['epa_val'] = float(epa_val)
            return_vals['epa_val_total_fa'] = float(epa_val_total_fa)
        except:
            if debug_print:
                print('ERROR: "epa_val" or "epa_val_total_fa" is not a valid number')
            return return_on_error

        # convert epa val from decimal to percentage (i.e., 0.01 -> 1.00)
        if convert_val_from_decimal_to_percentage:
            return_vals['epa_val'] *= 100
            return_vals['epa_val_total_fa'] *= 100
        
        if debug_print:
            print('SUCCESS', return_vals)

        if None in return_vals.values():
            raise Exception("ERROR: epa data all good but 'None' still in return values, missing a value assignment in dict?")
        return return_vals.values()

class GetActiveStatus(DBColumnsBase):
    '''
    Get the "active status" of each pond. Normally, if it has a 'Filter AFDW' and 'Fo' measurement, it would be considered 'Active'
    However, to handle occasional lags in data reporting, this status query determines if a pond should be considered active by checking previous dates
    If any rows within the 'max_check_days' int var have 'Filter AFDW' and 'Fo' measurements, then they will still be considered 'Active'.
    If any row, in the 'Split Innoculum' column, is preceded by 'HC' for the previous date (or 'I' for the current date and 'H' 
    for the previous date [old way of reporting]), then that row is noted as 'Inactive'
    '''
    OUT_TABLE_NAME = 'ponds_data_calculated'
    DEPENDENCY_CLASSES = ['DailyDataLoad'] # define class dependency that needs to be run prior to an instance of this class
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def run(self):
        active_df = self._get_ponds_active_status(begin_date=self.run_date_start, end_date=self.run_date_end) # gets df with 'active_status' col

        # update the db (calling a function from .db_utils.py)
        update_table_rows_from_df(db_engine=self.db_engine, db_table_name=self.OUT_TABLE_NAME, update_data_df=active_df)
    
    def _get_ponds_active_status(self, begin_date: datetime, end_date: datetime) -> pd.DataFrame:
        max_check_days = 5 # maximum number of days to look back from a given date to determine if it has activity
        def _check_row(row_data):
            row_pond_id = row_data['PondID']
            row_date = row_data['Date']
            
            # filter the df to only rows being checked (current row, and the 5 days preceding it) for specific row_pond_id
            check_df = ref_df[(ref_df['PondID'] == row_pond_id) & (ref_df['Date'] >= row_date - pd.Timedelta(days=max_check_days)) & (ref_df['Date'] <= row_date)]

            # reverse sort the df by date (so newest to oldest)
            # do this instead of just .iloc[::-1] in case of the reference data being out of order
            check_df = check_df.sort_values(by='Date', ascending=False)
            # reset index so that values relative each row can be checked (next row values)
            check_df = check_df.reset_index(drop=True)
            
            # iterate through check_df rows in reverse order
            for row_idx, row_data in check_df.iterrows():
                # first check if there's any value in the 'Time Sampled' column
                # next, check if there is a value in either of the 'Filter AFDW' or 'Fo' columns (and that they are not zero)
                # if so, then consider the pond as active
                if not pd.isna(row_data['Time Sampled']):
                    if (not pd.isna(row_data['Filter AFDW']) and row_data['Filter AFDW'] != 0) or (not pd.isna(row_data['Fo']) and row_data['Fo'] != 0):
                        return True
                # else, if this is not the last item being iterated, check the next index value of 'Split Innoculum' column
                # for the prior day (but checking next index because of reversed order): "HC" value indicates complete harvest, 
                # meaning that the pond is inactive if iteration reached this point
                elif row_idx+1 != len(check_df) and check_df.iloc[row_idx+1]['Split Innoculum'] == "HC":
                    return False
                # also check for "H" on prior row and "I" in current row (discontinued way of noting full harvests)
                elif row_data['Split Innoculum'] == 'I' and row_idx+1 != len(check_df) and check_df.iloc[row_idx+1]['Split Innoculum'] == "H":
                    return False
            # if all rows in the 'check_df' are iterated through and didn't find any 'Fo' values
            # then assume inactive status, and return False
            return False
        
        query_begin_date = begin_date - pd.Timedelta(days=max_check_days)
        ref_df = query_data_table_by_date_range(db_name_or_engine=self.db_engine, table_name='ponds_data', query_date_start=begin_date-pd.Timedelta(days=max_check_days), query_date_end=end_date, col_names=['Time Sampled', 'Fo', 'Split Innoculum', 'Filter AFDW'])
        out_df = ref_df.loc[ref_df['Date'] >= begin_date, :].copy()
       
        for pond_id in out_df['PondID'].unique():
            pond_id_mask = out_df['PondID'] == pond_id
            out_df.loc[pond_id_mask, 'active_status'] = out_df.loc[pond_id_mask, :].apply(lambda x: _check_row(x), axis=1)
            
        return out_df[['Date', 'PondID', 'active_status']]

class MassVolumeHarvestableCalculations(DBColumnsBase):
    OUT_TABLE_NAME = 'ponds_data_calculated'
    DEPENDENCY_CLASSES = ['DailyDataLoad'] # define class dependency that needs to be run prior to an instance of this class
  
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def run(self):
        # Load data necessary for calculations from the ref data table
        # (query_data_table_by_date_range function returns a DataFrame)
        self.ref_data_df = query_data_table_by_date_range(db_name_or_engine=self.db_engine, 
                                                      table_name='ponds_data', 
                                                      query_date_start=self.run_date_start, 
                                                      query_date_end=self.run_date_end, 
                                                      col_names=["Filter AFDW", "Depth", "% Nanno", "Split Innoculum"]) # Date and PondID automatically included

        # Run calculations and collect in a df
        calc_df = self._base_calculations(self.ref_data_df)
        update_table_rows_from_df(db_engine=self.db_engine, db_table_name=self.OUT_TABLE_NAME, update_data_df=calc_df)
            
   # def base_calculations(self, check_date: datetime, db_engine: sqlalchemy.Engine | None = None, data_table_name: str = 'ponds_data', out_table_name: str = 'ponds_data_calculated') -> None:
    def _base_calculations(self, ref_df) -> pd.DataFrame:
        '''
        Method to generate calculated fields from daily data that has been loaded
          - INPUT DATA: 
              - ref_df: reference dataframe from sql query w/ Date, PondID, AFDW, % Nanno, and Depth columns

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
        
        # variables for calculating harvestable mass 
        # based on min target density to begin harvesting, target density after harvesting and topping off with water, and target depth after topping off with water
        TARGET_TOPOFF_DEPTH = 13
        TARGET_DENSITY_AFTER_TOPOFF = 0.4
        MIN_HARVEST_DENSITY = 0.5
        
        calcs_df = pd.DataFrame()
        for (date, pond_id, afdw, depth, pct_nanno) in self.ref_data_df.loc[:,['Date', 'PondID', 'Filter AFDW', 'Depth', '% Nanno']].values.tolist():    

            # initialize calculations dict with all values empty
            calcs_dict = {'calc_mass': None,
                          'calc_mass_nanno_corrected': None,
                          'harvestable_depth_inches': None,
                          'harvestable_gallons': None,
                          'harvestable_mass': None,
                          'harvestable_mass_nanno_corrected': None}
            
            if any(pd.isna(x) or x in (0, None) for x in (afdw, depth)):
                pass # skip calculations since data is missing
            else:
                calcs_dict['calc_mass'] = afdw_depth_to_mass(afdw, depth, pond_id)
                
                # calculate harvestable depth of pond (in inches), rounding down to nearest 1/8 inch
                calcs_dict['harvestable_depth_inches'] = math.floor((((depth * afdw) - (TARGET_TOPOFF_DEPTH * TARGET_DENSITY_AFTER_TOPOFF)) / afdw)*8)/8
                if calcs_dict['harvestable_depth_inches'] < 0:
                    calcs_dict['harvestable_depth_inches'] = 0
                    
                # calculate harvestable volume (in gallons) based on harvestable depth and conversion factor (35,000) to gallons. 
                calcs_dict['harvestable_gallons'] = calcs_dict['harvestable_depth_inches'] * 35000 
                # Double for the '06' and '08' column (last two digits of PondID) ponds since they are double size
                if pond_id[-2:] == '06' or pond_id[-2:] == '08':
                    calcs_dict['harvestable_gallons'] *= 2

                # calculate harvestable mass using the harvestable_depth_inches calculation
                calcs_dict['harvestable_mass'] = afdw_depth_to_mass(afdw, calcs_dict['harvestable_depth_inches'], pond_id)

                if type(pct_nanno) in (float, int, complex):
                    calcs_dict['calc_mass_nanno_corrected'] = round(calcs_dict['calc_mass'] * (pct_nanno/100),2)
                    calcs_dict['harvestable_mass_nanno_corrected'] = round(calcs_dict['harvestable_mass'] * (pct_nanno/100), 2)
                else:
                    calcs_dict['calc_mass_nanno_corrected'] = 0
                    calcs_dict['harvestable_mass_nanno_corrected'] = 0

            # use dict comprehension to filter: to set <=0 values equal to None in numeric fields
            append_dict = {col_name: (value if (type(value) not in (int, float, complex) or value > 0) else None) for (col_name, value) in 
                                        {**{'Date': date, 'PondID': pond_id}, **calcs_dict}.items()}
            calcs_df = calcs_df.append(append_dict, ignore_index=True)

        return calcs_df

class CalcMassGrowth(DBColumnsBase):
    OUT_TABLE_NAME = 'ponds_data_calculated'
    DEPENDENCY_CLASSES = ['DailyDataLoad', 'MassVolumeHarvestableCalculations', 'GetActiveStatus'] # define class dependency that needs to be run prior to an instance of this class
    MIN_LOOKBACK_DAYS = 7
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def run(self):
        pass
           
    def _calc_growth(self, start_date, end_date):
        # query data to calculate 'liters' (from depth) 
        # and 'normalized liters' by converting liters with conversion factor (ratio of 'Filter AFDW' value compared to 0.50)
        ref_df1 = query_data_table_by_date_range(db_name_or_engine=self.db_engine, table_name='ponds_data', query_date_start=start_date, query_date_end=end_date, col_names=['Filter AFDW', 'Depth'])
        inches_to_liters = 35000 * 3.78541 # save conversion factor for depth (in inches) to liters
        ref_df1['liters'] = ref_df1['Depth'].apply(lambda depth_val: inches_to_liters * depth_val)
        ref_df1['afdw_norm_factor'] = ref_df1['Filter AFDW'] / 0.50
        ref_df1['normalized_liters'] = ref_df1['liters'] * ref_df1['afdw_norm_factor']

        # query calculated fields further used in this calculation
        ref_df2 = query_data_table_by_date_range(db_name_or_engine=self.db_engine, table_name='ponds_data_calculated', query_date_start=start_date, query_date_end=end_date, col_names=['active_status', 'est_harvested', 'calc_mass_nanno_corrected'])

        # join the queried dataframes
        calc_df = pd.merge(ref_df1, ref_df2, on=['Date', 'PondID'], how='outer')

        # set index as Date so that df.rolling() and df.shift() functions will work with days (to handle gaps in ponds being active)
        calc_df = calc_df.set_index('Date') 
        for pond_id in calc_df['PondID'].unique():
            # first, set mask only on PondID field and backfill necessary data cols            
            mask = (calc_df['PondID'] == pond_id) # (calc_df['active_status'] == True
            calc_df.loc[mask, ['normalized_liters', 'calc_mass_nanno_corrected']] = calc_df.loc[mask, ['normalized_liters', 'calc_mass_nanno_corrected']].bfill(limit=5) # backfill calculated mass for missing days (weekends, etc)
            calc_df.loc[mask, ['normalized_liters', 'calc_mass_nanno_corrected']] = calc_df.loc[mask, ['normalized_liters', 'calc_mass_nanno_corrected']].dropna() # drop rows that are still N/A after backfill
            
            # second, update mask to include 'active' ponds only
            mask = (calc_df['PondID'] == pond_id) & (calc_df['active_status'] == True)
            calc_df.loc[mask, :] = calc_df.loc[mask, :].fillna(0) # fill in n/a values for .rolling().sum() to calculate rolling sums (otherwise NaN values cause it to not work)

            # calculate rolling harvested amount for past 7-days
            # since the 'calc_mass_nanno_corrected' includes the harvested amount for the day when harvested, then shift this rolling window by
            # 1 day and actually only get a 6-day rolling sum (this ensures that harvests are not being counted on the same day) 
            calc_df.loc[mask, 'rolling_7_day_harvested_mass'] = calc_df.loc[mask, 'est_harvested'].shift(1).rolling('6d').sum()
            
            # get the 'liters', 'normalized liters', and 'calc_mass' from 7-days ago for ease of calculation
            calc_df.loc[mask, 'liters_7_days_prev'] = calc_df.loc[mask, 'liters'].shift(7, freq='D')
            calc_df.loc[mask, 'normalized_liters_7_days_prev'] = calc_df.loc[mask, 'normalized_liters'].shift(7, freq='D')
            calc_df.loc[mask, 'mass_7_days_prev'] = calc_df.loc[mask, 'calc_mass_nanno_corrected'].shift(7, freq='D')
            
            # calculate the net 7-day mass change in both kilograms and grams
            calc_df.loc[mask, '7_day_mass_change_kg'] = (calc_df.loc[mask, 'rolling_7_day_harvested_mass'] + calc_df.loc[mask, 'calc_mass_nanno_corrected'] - calc_df.loc[mask, 'mass_7_days_prev'])
            calc_df.loc[mask, '7_day_mass_change_grams'] = calc_df.loc[mask, '7_day_mass_change_kg']*1000
            
            # calculate growth
            calc_df.loc[mask, '7d_growth'] = calc_df.loc[mask, '7_day_mass_change_grams'] / calc_df.loc[mask, 'liters_7_days_prev']
            calc_df.loc[mask, '7d_normalized_growth'] = calc_df.loc[mask, '7_day_mass_change_grams'] / calc_df.loc[mask, 'normalized_liters_7_days_prev']

        # reset index so that 'Date' is a column again
        calc_df = calc_df.reset_index()

        calc_df = calc_df[['Date', 'PondID', '7d_growth', '7d_normalized_growth']]
        
        # # update the db (calling a function from .db_utils.py)
        # update_table_rows_from_df(db_engine=self.db_engine, db_table_name=self.OUT_TABLE_NAME, update_data_df=growth_df)
        calc_df.to_excel('test_pond_growth_output.xlsx')

class CalcAggMassGrowth(DBColumnsBase):
    OUT_TABLE_NAME = 'ponds_data_calculated'
    DEPENDENCY_CLASSES = ['DailyDataLoad', 'MassVolumeHarvestableCalculations', 'GetActiveStatus'] # define class dependency that needs to be run prior to an instance of this class
    MIN_LOOKBACK_DAYS = 7
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def run(self):
        pass
           
    def _calc_growth(self, start_date, end_date):
        # query data to calculate 'liters' (from depth) 
        # and 'normalized liters' by converting liters with conversion factor (ratio of 'Filter AFDW' value compared to 0.50)
        ref_df1 = query_data_table_by_date_range(db_name_or_engine=self.db_engine, table_name='ponds_data', query_date_start=start_date, query_date_end=end_date, col_names=['Filter AFDW', 'Depth'])
        inches_to_liters = 35000 * 3.78541 # save conversion factor for depth (in inches) to liters
        ref_df1['liters'] = ref_df1['Depth'].apply(lambda depth_val: inches_to_liters * depth_val)
        ref_df1['afdw_norm_factor'] = ref_df1['Filter AFDW'] / 0.50
        ref_df1['normalized_liters'] = ref_df1['liters'] * ref_df1['afdw_norm_factor']
        
        # query calculated fields further used in this calculation
        ref_df2 = query_data_table_by_date_range(db_name_or_engine=self.db_engine, table_name='ponds_data_calculated', query_date_start=start_date, query_date_end=end_date, col_names=['active_status', 'est_harvested', 'calc_mass_nanno_corrected'])

        # join the queried dataframes
        calc_df = pd.merge(ref_df1, ref_df2, on=['Date', 'PondID'], how='outer')
        
        # set index as Date so that df.rolling() and df.shift() functions will work with days (to handle gaps in ponds being active)
        calc_df = calc_df.set_index('Date') 

        # iterate through PondIDs and backfill missing data in the 'normalized_liters' and 'calc_mass_nanno_corrected' columns
        for pond_id in calc_df['PondID'].unique():
            mask = (calc_df['PondID'] == pond_id) 
            calc_df.loc[mask, ['normalized_liters', 'calc_mass_nanno_corrected']] = calc_df.loc[mask, ['normalized_liters', 'calc_mass_nanno_corrected']].bfill(limit=5) # backfill calculated mass for missing days (weekends, etc)
        # drop any rows which now contain N/A values for either 'normalized_liters' or 'calc_mass_nanno_corrected'
        # most being dropped here are probably just 'inactive' but this should catch any other cases with data missing for whatever reason
        calc_df = calc_df.dropna(how='any', subset=['normalized_liters', 'calc_mass_nanno_corrected']) 

        # filter out inactive ponds
        # But do it AFTER backfilling data, to ensure that data isn't backfilled over gaps where a pond isn't active
        # (therefore, data will be backfilled into non-active days, but then that backfilled data is removed at this step as it isn't valid anyway)
        calc_df = calc_df[calc_df['active_status'] == True]
        
        # fill in any missing values (that weren't forward filled) with zero
        calc_df = calc_df.fillna(0)

        # subtotal all active ponds by 'Date'
        calc_df = calc_df.groupby(by=['Date']).sum().reset_index()
        # set 'Date' as index for .rolling() and .shift() to work with daily frequency
        calc_df = calc_df.set_index('Date') 
        
        # calculate rolling harvested amount for past 7-days
        # since the 'calc_mass_nanno_corrected' includes the harvested amount for the day when harvested, then shift this rolling window by
        # 1 day and actually only get a 6-day rolling sum (this ensures that harvests are not being counted on the same day) 
        calc_df['rolling_7_day_harvested_mass'] = calc_df['est_harvested'].shift(1).rolling('6d').sum()
        
        # get the 'liters', 'normalized liters', and 'calc_mass' from 7-days ago for ease of calculation
        calc_df['liters_7_days_prev'] = calc_df['liters'].shift(7, freq='D')
        calc_df['normalized_liters_7_days_prev'] = calc_df['normalized_liters'].shift(7, freq='D')
        calc_df['mass_7_days_prev'] = calc_df['calc_mass_nanno_corrected'].shift(7, freq='D')
        
        # calculate the net 7-day mass change in both kilograms and grams
        calc_df['7_day_mass_change_kg'] = calc_df['rolling_7_day_harvested_mass'] + calc_df['calc_mass_nanno_corrected'] - calc_df['mass_7_days_prev']
        calc_df['7_day_mass_change_grams'] = calc_df['7_day_mass_change_kg'] * 1000
        
        # calculate 7-day aggregrate growth (in grams/Liter/day)
        calc_df['7d_agg_growth'] = calc_df['7_day_mass_change_grams'] / calc_df['liters_7_days_prev']
        calc_df['7d_agg_normalized_growth'] = calc_df['7_day_mass_change_grams'] / calc_df['normalized_liters_7_days_prev']

        # reset index so that 'Date' is a column again
        calc_df = calc_df.reset_index()
        
        # calculate 7-day aggregate normalized (to equivalent of Liters if the AFDW was 0.50) growth (in grams/Liter/day)
        calc_df = calc_df[['Date', '7d_agg_growth', '7d_agg_normalized_growth']]
        
        # # update the db (calling a function from .db_utils.py)
        # update_table_rows_from_df(db_engine=self.db_engine, db_table_name=self.OUT_TABLE_NAME, update_data_df=growth_df)
        calc_df.to_excel('test_agg_growth_output.xlsx')
        

class CalcHarvestedSplit(DBColumnsBase):
    OUT_TABLE_NAME = 'ponds_data_calculated'
    DEPENDENCY_CLASSES = ['DailyDataLoad', 'MassVolumeHarvestableCalculations'] # define class dependency that needs to be run prior to an instance of this class
   
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def run(self):
        hs_df = self._calc_harvested_split(start_date=self.run_date_start, end_date=self.run_date_end) # gets df with 'active_status' col

        # update the db (calling a function from .db_utils.py)
        update_table_rows_from_df(db_engine=self.db_engine, db_table_name=self.OUT_TABLE_NAME, update_data_df=hs_df)
    
    def _calc_harvested_split(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        df_ref_data = query_data_table_by_date_range(db_name_or_engine=self.db_engine, table_name='ponds_data', query_date_start=start_date, query_date_end=end_date, col_names=['Split Innoculum'])
        df_data_calcs = query_data_table_by_date_range(db_name_or_engine=self.db_engine, table_name='ponds_data_calculated', query_date_start=start_date, query_date_end=end_date, col_names=None)
       
        output_df = df_data_calcs.copy()
        for pond_id in output_df['PondID'].unique():
            # get df mask with only values for specific PondID
            mask = output_df['PondID'] == pond_id
            # get mask index (will be non-sequential) corresponding to the overall df index (i.e., [4, 92, 180, 268, 356])
            mask_idx = list(output_df.loc[mask].index) 

            # fill missing values for calc_mass_nanno_corrected, to calculate mass changes between days, and get the next available value to calculate with
            # to ensure that values for days marked as "harvested", "split", or "inactive" in the Split Innoculum column from daily data, are not backfilled
            # first mark any values that are both missing and on day's marked as "H"/"S"/"I", as 'tmp_for_ffill'
            # then backfill all missing values 
            # then replace 'tmp_for_ffill' values with None, and forward fill those
            ## TODO: look into a more efficient way to do this??
            output_df.loc[mask, '_tmp_mass'] = output_df.loc[mask, :].apply(lambda df_row: 'tmp_for_ffill' if (pd.isna(df_row['calc_mass_nanno_corrected'])) & (df_ref_data[(df_ref_data['PondID'] == df_row['PondID']) & (df_ref_data['Date'] == df_row['Date'])]['Split Innoculum'].iloc[0] != None) else df_row['calc_mass_nanno_corrected'], axis=1)
            output_df.loc[mask, '_tmp_mass'] = output_df.loc[mask, '_tmp_mass'].bfill()
            output_df.loc[mask, '_tmp_mass'] = output_df.loc[mask, '_tmp_mass'].replace('tmp_for_ffill', None)
            output_df.loc[mask, '_tmp_mass'] = output_df.loc[mask, '_tmp_mass'].ffill()
            
            def _get_h_s_amount(df_row):
                df_row = df_row.copy()
                return_dict = {'H/S': None, 'est_harvested': None, 'est_split': None} # initialize return_dict, values default as None
                
                # get the position of the current row within the mask_idx (i.e., if mask_idx = [4, 92, 180, 268, 356] and current row index is 180, then cur_row_mask_idx = 3
                cur_row_mask_idx = mask_idx.index(df_row.name)

                ref_split_innoc_val_cur = df_ref_data[(df_ref_data['PondID'] == df_row['PondID']) & (df_ref_data['Date'] == df_row['Date'])]['Split Innoculum'].iloc[0]
                return_dict['H/S'] = ref_split_innoc_val_cur # add the split_innoculum value as a reference in calculation DB
                
                # if not on last row of data, and if the current row 'Split Innoculum' value == "S" or "H", then look ahead to the next row for change in mass to get est harvested amount
                if cur_row_mask_idx+1 != len(mask_idx): 
                    next_row_vals = output_df.loc[mask_idx[cur_row_mask_idx + 1], :] # returns a single-row pandas df
                    ref_split_innoc_val_next = df_ref_data[(df_ref_data['PondID'] == df_row['PondID']) & (df_ref_data['Date'] == next_row_vals['Date'])]['Split Innoculum'].iloc[0]
                    if ref_split_innoc_val_cur in ('H', 'S', 'HC'):
                        if ref_split_innoc_val_cur in ('H', 'HC'):
                            update_key = 'est_harvested'
                        else:
                            update_key = 'est_split'
                        
                        if ref_split_innoc_val_cur == 'HC' or ref_split_innoc_val_next == 'I': # if next row is noted "I" for 'inactive', then assume all mass was harvested and return current day mass
                            return_dict[update_key] = df_row['_tmp_mass']
                        else:
                            # if the '_tmp_mass' value is na, then assume change in mass cannot be calculated
                            if pd.isna(df_row['_tmp_mass']):
                                #print('Error with', df_row['PondID'], df_row['Date'])
                                pass
                            else:
                                next_day_mass_change = df_row['_tmp_mass'] - next_row_vals['_tmp_mass']
                                if next_day_mass_change > 0:
                                    return_dict[update_key] = next_day_mass_change
                return return_dict
                
            output_df.loc[mask, ['H/S', 'est_harvested', 'est_split']] = output_df.loc[mask, :].apply(lambda x: _get_h_s_amount(x), axis=1, result_type='expand')
        output_df = output_df.drop([col for col in output_df.columns if '_tmp_' in col], axis=1)
        return output_df

class GetPondHealthStatusCode(DBColumnsBase):
    OUT_TABLE_NAME = 'ponds_data_calculated'
    DEPENDENCY_CLASSES = ['DailyDataLoad', 'GetActiveStatus', 'EPALoad'] # define class dependency that needs to be run prior to an instance of this class
  
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def run(self):
        status_df = self._get_pond_status_color_code(begin_date=self.run_date_start, end_date=self.run_date_end)

        # update the db (calling a function from .db_utils.py)
        update_table_rows_from_df(db_engine=self.db_engine, db_table_name=self.OUT_TABLE_NAME, update_data_df=status_df)
        
    def _get_pond_status_color_code(self, begin_date: datetime, end_date: datetime) -> pd.DataFrame:
        '''
        status codes:
            - 0: inactive pond 
            - 1: grey: incomplete data (either 'afdw' or 'epa_val' are missing)
            - 2: red: afdw less than 0.25
            - 3: brown: afdw >= 0.25; epa_val < 2.5%
            - 4: yellow: (afdw >= 0.25 and afdw < 0.50) OR (epa_val >=2.5% and epa_val < 3%)
            - 5: light green: afdw >= 0.50 and < 0.80; epa_val > 3%
            - 6: dark green: afdw >= 0.80; epa_val > 3%
            - ERROR: should not happen, possible bug in code / missing conditions / etc
        '''
        
        afdw_df = query_data_table_by_date_range(db_name_or_engine=self.db_engine, table_name='ponds_data', query_date_start=begin_date, query_date_end=end_date, col_names=['Filter AFDW'])
        active_df = query_data_table_by_date_range(db_name_or_engine=self.db_engine, table_name='ponds_data_calculated', query_date_start=begin_date, query_date_end=end_date, col_names=['active_status'])
        epa_df = query_data_table_by_date_range(db_name_or_engine=self.db_engine, table_name='epa_data', query_date_start=begin_date, query_date_end=end_date, col_names=['epa_val'], raise_exception_on_error=False)
        df = functools.reduce(lambda df1, df2: pd.merge(df1, df2, on=['Date','PondID'], how='outer'), [df for df in [afdw_df, active_df, epa_df]])
       
        # conditions
        conditions = [
            df['active_status'] == False,
            (pd.isna(df['Filter AFDW'])) | (pd.isna(df['epa_val'])),
            df['Filter AFDW'] < 0.25,
            (df['Filter AFDW'] >= 0.25) & (df['epa_val'] < 2.5),
            ((df['Filter AFDW'] >= 0.25) & (df['Filter AFDW'] < 0.50)) | ((df['epa_val'] >= 2.5) & (df['epa_val'] < 3.0)),
            (df['Filter AFDW'] >= 0.50) & (df['Filter AFDW'] < 0.80) & (df['epa_val'] >= 3.0),
            (df['Filter AFDW'] >= 0.80) & (df['epa_val'] >= 3.0)]
        choices = [0, 1, 2, 3, 4, 5, 6]
        
        df['status_code'] = np.select(conditions, choices, default='ERROR')
        return df[['Date', 'PondID', 'status_code']]
        
class ChemicalUsageLoad(DBColumnsBase):
    OUT_TABLE_NAME = 'ponds_data_expenses'
    DEPENDENCY_CLASSES = ['DailyDataLoad'] # define class dependency that needs to be run prior to an instance of this class
   
    # hardcode costs - TEMPORARY??
    chem_costs = {
        'uan-32': {'uom': 'gal', 'cost': 2.08, 'data_column': 'Volume UAN-32 Added', 'out_column': 'uan32_cost'},
        'fertilizer-10-34': {'uom': 'gal', 'cost': 4.25, 'data_column': 'Volume 10-34 Added', 'out_column': 'fert1034_cost'},
        'bleach': {'uom': 'gal', 'cost': 3.15, 'data_column': 'Volume Bleach Added', 'out_column': 'bleach_cost'},
        #'co2': {'uom': 'lb', 'cost': 0.15, 'data_column': '', 'out_column': 'co2_cost'},  DON'T YET HAVE A METHOD OF SPLITTING CO2 COSTS PER POND / IMPLEMENT LATER
        'trace': {'uom': 'gal', 'cost': 0.41, 'data_column': 'Volume Trace Added', 'out_column': 'trace_cost'},
        'iron': {'uom': 'gal', 'cost': 0.60, 'data_column': 'Volume Iron Added', 'out_column': 'iron_cost'},
        'cal-hypo': {'uom': 'kg', 'cost': 3.17, 'data_column': 'kg Cal Hypo Added', 'out_column': 'cal_hypo_cost'},
        'benzalkonium': {'uom': 'gal', 'cost': 49.07, 'data_column': 'Volume Benzalkonium Added', 'out_column': 'benzalkonium_cost'}
            }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def run(self):
        calc_df = self._chemical_cost_calculations(start_date=self.run_date_start, end_date=self.run_date_end) 
        
        # update the db (calling a function from .db_utils.py)
        update_table_rows_from_df(db_engine=self.db_engine, db_table_name=self.OUT_TABLE_NAME, update_data_df=calc_df)
        
    def _chemical_cost_calculations(self, start_date: datetime, end_date: datetime) -> None:
        '''
        Method to calculate estimated costs from consumed chemicals and store in db_table
        '''
        # query the chemical usage amounts from db
        data_col_names = [v['data_column'] for v in self.chem_costs.values()]
        calc_df = query_data_table_by_date_range(db_name_or_engine=self.db_engine, table_name='ponds_data', query_date_start=start_date, query_date_end=end_date, col_names=data_col_names)
        calc_df = calc_df.fillna(0) # fill in zeroes for NaN chemical usage values

        # loop through chemicals in cost dict and add to "out_dict" according to the "out_column" value in cost dict
        out_cols = ['Date', 'PondID'] # init list to append with output column names to filter output df
        for idx, subdict in enumerate(self.chem_costs.values()):
            in_col_name = subdict['data_column']
            out_col_name = subdict['out_column']
            col_cost = subdict['cost']
            out_cols.append(out_col_name)
            calc_df[out_col_name] = calc_df[in_col_name].apply(lambda x: round(x * col_cost, 2))
        calc_df = calc_df.loc[:, out_cols] # filter df to only the output columns
        return calc_df

        
############# TODO: update growth calculation with method using db
#         def calculate_growth(pond_data_df, run_date, num_days, remove_outliers, data_count_threshold=2, weighted_stats_for_outliers=True):
#             '''
#             NEW **** MOVED TO DATALOADER, UTILIZE DATABASE TO QUERY/CALCULATE THIS?? ****
#             NOT UPDATED YET
            
#             pond_data_df: pandas dataframe
#                 - dataframe for individual pond
#             run_date: datetime.date
#                 - current date (required for growth relative to this date)
#             remove_outliers: bool
#                 - whether to remove outliers from the data before calculating growth (which are then zeroed and forward-filled)
#                 - outliers are calculated using the 10 most-recent non-zero data points
#                 - when this is set then the current day is excluded from calculating mean and std. dev (since it's more likely to be an outlier itself)
#             num_days: int
#                 - number of days to calculate growth for
#             data_count_threshold: int 
#                 - absolute threshold for days that must have data within the num_days growth period, otherwise growth will be 'n/a'
#             weighted_stats_for_outliers: bool
#                 - whether to use weighted statistics for finding outliers, applies 80% weight evenly to the most-recent 5 days, then 20% to the remainder
#             outlier_stddev_thresh: int
#                 - the standard deviation threshold for determining outliers (when greater or less than the threshold * standard deviation)
#                 - using a threshold of 2 standard deviations as default for more aggressive outlier detection (statistical standard is usally 3 std. dev.)
#             pond_test: str
#                 - FOR TESTING: string corresponds to pond_name to display df output between steps
#             '''
            
#             # the standard deviation threshold for determining outliers (when greater or less than the threshold * standard deviation)
#             # using a threshold of 2.25 standard deviations as default for more aggressive outlier detection (statistical standard is usally 3 std. dev.)
#             outlier_stddev_thresh = 2.25 
            
#             pond_test=None # set to pond_name string for printing output at intermediate calculation steps
            
#             # Select last 20 days of data for columns 'AFDW', 'Depth', and 'Split Innoculum' (higher num in case of missing data, etc)
#             pond_growth_df = pond_data_df.loc[run_date - pd.Timedelta(days=20):run_date][['AFDW (filter)', 'Depth', 'Split Innoculum']]
#             # Get calculated mass column from AFDW and Depth
#             pond_growth_df['mass'] = afdw_depth_to_mass(pond_growth_df['AFDW (filter)'], pond_growth_df['Depth'], pond_name)
            
#             def next_since_harvest_check(row):
#                 ''' helper function for df.apply to label row as the next available data since a pond was last harvested. checks 3 days prior'''
#                 if row['mass'] != 0:
#                     if row['tmp harvest prev 1'] == 'Y': # if the previous day was harvested ('H' in 'Split Innoculum' col)
#                         return 'Y'
#                     elif row['tmp data prev 1'] == 'N' and row['tmp harvest prev 2'] == 'Y': # if 
#                         return 'Y'
#                     elif row['tmp data prev 1'] == 'N' and row['tmp data prev 2'] == 'N' and row['tmp harvest prev 3'] == 'Y':
#                         return 'Y'
#                     else:
#                         return ''
#                 else:
#                     return ''
    
#             # find if row is the next data since harvest (i.e., if there is a few days delay in data since it was harvested), to ensure negative change from harvest is always zeroed out for growth calcs
#             pond_growth_df['tmp data prev 1'] = pond_growth_df['mass'].shift(1).apply(lambda val: 'Y' if val != 0 else 'N')
#             pond_growth_df['tmp data prev 2'] = pond_growth_df['mass'].shift(2).apply(lambda val: 'Y' if val != 0 else 'N')
#             pond_growth_df['tmp harvest prev 1'] = pond_growth_df['Split Innoculum'].shift(1).apply(lambda val: 'Y' if val == 'H' or val == 'S' else 'N')
#             pond_growth_df['tmp harvest prev 2'] = pond_growth_df['Split Innoculum'].shift(2).apply(lambda val: 'Y' if val == 'H' or val == 'S' else 'N')
#             pond_growth_df['tmp harvest prev 3'] = pond_growth_df['Split Innoculum'].shift(3).apply(lambda val: 'Y' if val == 'H' or val == 'S' else 'N')
#             pond_growth_df['next data since harvest'] = pond_growth_df.apply(lambda row: next_since_harvest_check(row), axis=1) 
#             del pond_growth_df['tmp data prev 1']
#             del pond_growth_df['tmp data prev 2'] 
#             del pond_growth_df['tmp harvest prev 1'] 
#             del pond_growth_df['tmp harvest prev 2'] 
#             del pond_growth_df['tmp harvest prev 3']
    
#             if remove_outliers: # calculate and drop outliers
#                 # get a new df for outlier detection and drop the last/ most recent row, 
#                 # assuming it is highly likely an outlier (due to data entry error, etc), so exclude that value from calc of mean and std dev for detecting outliers
#                 mass_nonzero_ = pond_growth_df.drop([run_date], axis=0) 
#                 mass_nonzero_ = mass_nonzero_[mass_nonzero_['mass'] != 0]['mass'] # get a pandas series with only nonzero values for calculated mass
#                 # limit selection for outlier detection to the last 10 non-zero values
#                 mass_nonzero_ = mass_nonzero_.iloc[-10:]
                
#                 if pond_name == pond_test:
#                     print('std dev: ', mass_nonzero_.std())
#                     print('mean: ', mass_nonzero_.mean())
#                     print('TEST LENGTH QUANTILESDF', len(mass_nonzero_), mass_nonzero_)
                
#                 if weighted_stats_for_outliers and len(mass_nonzero_) >= 5: # use weighted stats if specified, UNLESS there are less than 5 non-zero "mass" datapoints in total
#                     from statsmodels.stats.weightstats import DescrStatsW
                    
#                     # init weights for weighted outlier detection (list should sum to 1 total)
#                     outlier_weights_ = [0.16]*5 # first 5 values weighted to 0.8 total
#                     if len(mass_nonzero_) > 5: # add more weights if necessary (between 1 and 5 more)
#                         outlier_weights_ += [((1-sum(outlier_weights_)) / (len(mass_nonzero_)-len(outlier_weights_)))] * (len(mass_nonzero_)-len(outlier_weights_))
                    
#                     weighted_stats = DescrStatsW(mass_nonzero_.iloc[::-1], weights=outlier_weights_, ddof=0)
    
#                     if pond_name == pond_test:
#                         print(f'Weighted mean: {weighted_stats.mean}, Weighted Std Dev: {weighted_stats.std}, weighted outlier range (+- {outlier_stddev_thresh} std dev): <{weighted_stats.mean-(outlier_stddev_thresh*weighted_stats.std)}, >{weighted_stats.mean+(outlier_stddev_thresh*weighted_stats.std)}')
#                         print(f'Non-Weighted mean: {mass_nonzero_.mean()}, Non-Weighted Std Dev: {mass_nonzero_.std()}, non-weighted outlier range (+- {outlier_stddev_thresh} std dev): <{mass_nonzero_.mean()-(outlier_stddev_thresh*mass_nonzero_.std())}, >{mass_nonzero_.mean()+(outlier_stddev_thresh*mass_nonzero_.std())}')
#                         print('test df', test_mean := mass_nonzero_[-5:])
                    
#                     mass_nonzero_mean_ = weighted_stats.mean
#                     mass_nonzero_std_ = weighted_stats.std
#                 else: # using regular statistics / non-weighted
#                     mass_nonzero_mean_ = mass_nonzero_.mean()
#                     mass_nonzero_std_ = mass_nonzero_.std()
    
#                 if pond_name == pond_test:
#                     print(f'TESTING FOR POND {pond_name}')
#                     print('outliers\n', pond_growth_df[(pond_growth_df['mass']-mass_nonzero_mean_).apply(lambda x: -x if x < 0 else x) >= (outlier_stddev_thresh*mass_nonzero_std_)], 'testing outliers')
                
#                 #outliers_df = pond_growth_df[np.abs(pond_growth_df['mass']-mass_nonzero_mean_) >= (outlier_stddev_thresh*mass_nonzero_std_)]
#                 outliers_df = pond_growth_df[(pond_growth_df['mass']-mass_nonzero_mean_).apply(lambda x: -x if x < 0 else x) >= (outlier_stddev_thresh*mass_nonzero_std_)]
#                 pond_growth_df = pond_growth_df.assign(outlier=pond_growth_df.index.isin(outliers_df.index)) # assign 'outliers' column equal to True if data point is an outlier
    
#                 if pond_name == pond_test:
#                     print('Starting data\n')
#                     from IPython.display import display
#                     display(pond_growth_df)
    
#                 # set the mass of outliers to 0
#                 pond_growth_df['mass'] = pond_growth_df.apply(lambda row: 0 if row['outlier'] == True else row['mass'], axis=1)
    
#                 if pond_name == pond_test:
#                     print('\nAfter removing outliers\n')
#                     display(pond_growth_df)
            
#             # check if enough data exists for calculating growth, and return 'n/a' early if not
#             growth_period_data_count = pond_growth_df.loc[run_date - pd.Timedelta(days=num_days-1):run_date][pond_growth_df['mass'] != 0]['mass'].count()
#             if growth_period_data_count < data_count_threshold:
#                 return 'n/a'
            
#             # forward fill zero-values
#             pond_growth_df['mass'] = pond_growth_df['mass'].replace(0, float('nan')).fillna(method='ffill') # replace 0 with nan for fillna() to work
    
#             if pond_name == pond_test:
#                 print('\nAfter forward-filling zero vals\n')
#                 display(pond_growth_df)
    
#             # calculate estimated harvest amount, not used elsewhere but could be useful for some aggregate tracking
#             pond_growth_df['tmp prev mass'] = pond_growth_df['mass'].shift(1)
#             pond_growth_df['tmp next mass'] = pond_growth_df['mass'].shift(-1)
#             pond_growth_df['harvested estimate'] = pond_growth_df.apply(lambda row: row['tmp prev mass'] - row['mass'] if row['next data since harvest'] == 'Y' else 0, axis=1)
    
#             pond_growth_df['day chng'] = pond_growth_df.apply(lambda row: row['mass'] - row['tmp prev mass'] if row['next data since harvest'] != 'Y' else 0, axis=1)
#             del pond_growth_df['tmp prev mass'] 
#             del pond_growth_df['tmp next mass'] 
    
#             if pond_name == pond_test:
#                 print('\nAfter calculating estimated harvest and daily change in mass')
#                 display(pond_growth_df)
    
#             # calculate growth
#             try:
#                 daily_growth_rate_ = int(pond_growth_df.loc[run_date - pd.Timedelta(days=num_days-1):run_date]['day chng'].sum() / num_days)
#                 if pond_name == pond_test:
#                     display(daily_growth_rate_)
#                 return daily_growth_rate_
#             except:
#                 return 'n/a: Error'

#             ## TODO: outlier detection (pond 0402 - 1/16/23 as example of outlier to remove)
#             # import numpy as np
# #                         quantiles_df = pond_growth_df[pond_growth_df['mass'] != 0]['mass']
# #                         print('std dev: ', quantiles_df.std())
# #                         print('mean: ', quantiles_df.mean())
# #                         print(quantiles_df[np.abs(quantiles_df-quantiles_df.mean()) <= (3*quantiles_df.std())])

# #                         quantile_low = quantiles_df.quantile(0.01)
# #                         quantile_high = quantiles_df.quantile(0.99)
# #                         iqr = quantile_high - quantile_low
# #                         low_bound = quantile_low - (1.5*iqr)
# #                         high_bound = quantile_high + (1.5*iqr)
# #                         print('low quantile: ', quantile_low, '| high quantile: ', quantile_high)
# #                         print('low bound: ', low_bound, '| high bound: ', high_bound)
           
## TODO PROCESSING AND SFDATA
    # def load_processing_data(self, excel_filename):
    #     print('Loading processing data')
    #     df = pd.read_excel(excel_filename, sheet_name='SF Harvest')
    #     df = df.rename(columns={df.columns[0]:'date'})
    #     df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.normalize() # convert date column from string *use .normalize method to remove potential time data
    #     df = df.set_index(df['date'].name) # set date column as index
    #     df = df.loc[:, ~df.columns.str.contains('^Unnamed')] # drop columns without a header, assumed to be empty or unimportant
    #     print('Processing data loaded!')
    #     return df
    
    # def load_sfdata(self, excel_filename):
    #     print('Loading SF Data')
    #     df = pd.read_excel(excel_filename, sheet_name='Customer Log')
    #     df = df.rename(columns={df.columns[0]:'date'})
    #     df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.normalize() # convert date column from string *use .normalize method to remove potential time data
    #     df = df.set_index(df['date'].name) # set date column as index
    #     df = df.loc[:, ~df.columns.str.contains('^Unnamed')] # drop columns without a header, assumed to be empty or unimportant
    #     print('SF data loaded!')
    #     return df 
    
