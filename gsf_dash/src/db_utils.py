import os
import sqlalchemy
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype
from dateutil.rrule import rrule, DAILY 
from datetime import datetime, date, timedelta
from pathlib import Path
from . import load_setting, EmailHandler
from .ms_account_connect import MSAccount, M365ExcelFileHandler

def get_db_table_columns(db_engine: sqlalchemy.Engine, table_name) -> list:
    table = load_table(db_engine, table_name)
    inspector = sqlalchemy.inspect(table)
    return [col.name for col in inspector.columns]

def check_if_table_exists(db_engine: sqlalchemy.Engine, table_name: str) -> bool:
    return sqlalchemy.inspect(db_engine).has_table(table_name) # boolean value depending on table_name existence in db

def init_db_table(db_engine: sqlalchemy.Engine, table_name: str) -> None:
    table_schemas_path = Path('./settings/db_schemas')
    table_schemas_dir = os.listdir(table_schemas_path)
    if f'{table_name}.create_table' not in table_schemas_dir:
        print(f'ERROR: table schema for {table_name} not found! Add sqlite table create statement to .settings/db_schemas/ folder as .create_table file')
        return

    # load the CREATE TABLE statement from file
    with open(table_schemas_path / f'{table_name}.create_table', 'r') as file:
        table_schema = file.read()

    table_already_exists = check_if_table_exists(db_engine, table_name)
    
    with db_engine.begin() as conn:
        if table_already_exists:
            if confirm_drop_table := input(f'{table_name} table already exists! Type "Yes" to confirm dropping table, otherwise press any other key') == 'Yes':
                tmp_table = sqlalchemy.Table(table_name, sqlalchemy.MetaData(), autoload_with=db_engine)
                tmp_table.drop(db_engine)
            else:
                print('Doing nothing because table already exists')
                return
                
        # create the table by executing CREATE TABLE statement
        conn.execute(sqlalchemy.text(table_schema))
        print('Successfully created table!')

def get_primary_keys(db_table: sqlalchemy.Table) -> list:
    inspector = sqlalchemy.inspect(db_table)
    # Inspector.primary_key -> Column objects that are primary keys for table -> column.name
    return [k.name for k in inspector.primary_key]

def update_table_rows_from_df(db_engine: sqlalchemy.Engine, db_table_name: str, update_data_df: pd.DataFrame, pk_cols: list = ['Date', 'PondID']) -> bool:
    '''
    Function to update database table rows from a source pandas DataFrame

    params: 
        - db_engine: sqlalchemy.Engine
        - db_table_name: string of the table name in the database to update rows in
        - update_data_df: dataframe of the data to be used for updating
        - pk_cols: list of columns to use as primary keys (defaults to ["Date", "PondID"])
    NOTE: 
        - pk_cols (primary key column names) must be in the update_data_df, as well as already existing in the database table
        - update_data_df columns must be already existing in the database table
        - if table update is unsuccessful, the database transaction will be rolled back!

    returns: bool (True on success, False on failure)
    '''
    if len(update_data_df) == 0:
        print('No data, skipping DB update...')
        return False
    
    # convert Date column in update dataframe to string
    update_data_df = convert_df_date_cols(update_data_df, option='TO_STR')

    # create temp table in database to update the target table from
    update_data_df.to_sql(name='__temp_table', con=db_engine, if_exists='replace', index=False)
    
    # construct a statement to first insert primary key pairs into the table if they don't already exist (ignore if they do exist using INSERT OR IGNORE for sqlite)
    pk_vals = ', '.join(['(' + ', '.join([f'"{i}"' for i in sublist]) + ')' for sublist in update_data_df[pk_cols].values.tolist()])
    '''
    pk_vals formatted as string example for (Date, PondID): "('2023-09-01', '0401'), ('2023-09-02', '0401'), ... "
    '''
    sql_insert_keys_stmt = f'INSERT OR IGNORE INTO {db_table_name} ({", ".join(pk_cols) if len(pk_cols) > 1 else pk_cols[0]}) VALUES {pk_vals};'
    sql_insert_keys_stmt = sqlalchemy.text(sql_insert_keys_stmt)
    
    # check that all columns being updated, as well as the primary key column arguments, are present in the database table. raise an exception if not
    db_table_col_names = get_db_table_columns(db_engine, db_table_name)
    if not all([col in db_table_col_names for col in list(update_data_df.columns)+pk_cols]): # add pk_cols to check list just in case columns were provided that do not exist in db table
        raise Exception(f'ERROR: columns to update are not present in table: {db_table_name}!\nColumns: {update_data_df.columns.tolist()}')
    
    # construct sql UPDATE FROM string
    # This works with Sqlite, but may need replaced if migrating to another database type
    update_col_names = [f'"{col_name}"' for col_name in update_data_df.columns if col_name not in pk_cols] # put quotes around column names so sql queries dont break for columns that include spaces in the name
    sql_update_stmt = (f'UPDATE {db_table_name} '
                + f'SET ({", ".join([col_name for col_name in update_col_names])}) = ({", ".join([f"__temp_table.{col_name}" for col_name in update_col_names])}) ' 
                + 'FROM __temp_table '
                + f'WHERE {" AND ".join([f"__temp_table.{key_col} = {db_table_name}.{key_col}" for key_col in pk_cols])};')
    sql_update_stmt = sqlalchemy.text(sql_update_stmt)
    
    # execute sql statement
    # if resulting number of modified rows does not equal the length of the update_df, then rollback transaction 
    with db_engine.begin() as conn:
        insert_rows_result = conn.execute(sql_insert_keys_stmt)
        print('New rows inserted:', insert_rows_result.rowcount)
        
        update_result = conn.execute(sql_update_stmt)
        if update_result.rowcount == len(update_data_df):
            print('Rows successfully updated:', update_result.rowcount)
            return True
        else:
            print(f'Error updating table rows, number of row updates ({update_result.rowcount}) does not match source dataframe length ({len(update_data_df)})! Rolling back db transaction...')
            conn.rollback()
            return False
 
def load_table(db_engine: sqlalchemy.Engine, table_name: str) -> sqlalchemy.Table:
    metadata = sqlalchemy.MetaData()
    return sqlalchemy.Table(table_name, metadata, autoload_with=db_engine)
    
def query_data_table_by_date(db_name_or_engine: str|sqlalchemy.Engine, table_name: str, query_date: datetime, col_names: list | None = None) -> pd.DataFrame:
    '''
    Helper function to query data from a database table for a specified date

    params
    ---------
    db_name: database name as string (not including path or file extension!)
    table_name: table name as a string
    query_date: datetime
    col_names: list of column names (strings) to return or None
        - if None -> returns all columns

    Returns
    --------
    pd.DataFrame
    '''
    # check if db_name_or_engine is a sqlalchemy.Engine instance...otherwise load the engine
    if not isinstance(db_name_or_engine, sqlalchemy.Engine):
        db_engine = sqlalchemy.create_engine(f"sqlite:///db/{db_name_or_engine}.db", echo=False)
    else:
        db_engine = db_name_or_engine

    query_date = query_date.strftime("%Y-%m-%d")
    if not check_if_table_exists(db_engine, table_name):
        print(f'ERROR: Could not load {table_name} from database: {db_name_or_engine}!')
    else:
        table_obj = load_table(db_engine, table_name)
        with db_engine.begin() as conn:
            if col_names:
                output = conn.execute(sqlalchemy.select(*[table_obj.c[col] for col in col_names]).where(table_obj.c["Date"] == query_date)).fetchall()
            else:
                output = conn.execute(sqlalchemy.select(table_obj).where(table_obj.c["Date"] == query_date)).fetchall()
        out_df = pd.DataFrame(output, columns=col_names)
        out_df = convert_df_date_cols(out_df, option='TO_DT') # convert the Date column from string (db representation) into datetime format
        return out_df

def query_data_table_by_date_range(db_name_or_engine: str|sqlalchemy.Engine, table_name: str, query_date_start: datetime, query_date_end: datetime, col_names: list | None = None, raise_exception_on_error: bool = True, check_safe_date: bool = False) -> pd.DataFrame:
    '''
    Helper function to query data from a database table for a specified date

    params
    ---------
    db_name: database name as string (not including path or file extension!)
    table_name: table name as a string
    query_date_start: datetime
    query_date_end: datetime
    col_names: list of column names (strings) to return or None
        - if None -> returns all columns
    raise_exception_on_error: bool :  if query fails, default (True) is to raise Exception; however if set to False this parameter will allow None to be returned instead
    check_safe_date: bool: if True, check for the first available date in the db table and override the query_date_start param; otherwise if False (default) do not override 
    
    Returns
    --------
    pd.DataFrame (including "Date" and "PondID" fields by default, so they do not need to be specified in col_names parameter!
    '''
    # check if db_name_or_engine is a sqlalchemy.Engine instance...otherwise load the engine
    if not isinstance(db_name_or_engine, sqlalchemy.Engine):
        db_engine = sqlalchemy.create_engine(f"sqlite:///db/{db_name_or_engine}.db", echo=False)
    else:
        db_engine = db_name_or_engine

    query_date_start_dt = query_date_start
    query_date_end_dt = query_date_end
    query_date_start = query_date_start.strftime("%Y-%m-%d")
    query_date_end = query_date_end.strftime("%Y-%m-%d")
    
    if not check_if_table_exists(db_engine, table_name):
        print(f'ERROR: Could not load {table_name} from database: {db_name_or_engine}!')
    else:
        table_obj = load_table(db_engine, table_name)

        # check if PondID column exists in table, if so, add to query output columns
        pond_id_flag = False
        for col in table_obj.columns:
            if '.PondID' in str(col):
                pond_id_flag = True
                break
        
        # since 'Date' and 'PondID' (if in table) columns do not need to be specified in params, add them to the col_names parameter 
        # if the col_names parameter is None (default), then all columns in the table are returned, so no need to append these
        if col_names != None:
            if pond_id_flag:
                col_names.insert(0,'PondID')
            col_names.insert(0,'Date')
  
        with db_engine.begin() as conn:
            
            # when check_safe_date = True, get date from first row of db table, and if query_date_start is earlier than that date, then set query_date_start to the first row date
            if check_safe_date:
                #first_avail_date = conn.execute(sqlalchemy.text("SELECT Date FROM :table_name LIMIT 1;"), {'table_name': table_name}).fetchall() 
                first_avail_date = conn.execute(sqlalchemy.select(table_obj.c["Date"])).first()[0]
                if datetime.strptime(first_avail_date, "%Y-%m-%d") > query_date_start_dt:
                    query_date_start = first_avail_date

                last_avail_date = conn.execute(sqlalchemy.select(table_obj.c["Date"]).order_by(table_obj.c["Date"].desc())).first()[0]               
                if datetime.strptime(last_avail_date, "%Y-%m-%d") < query_date_end_dt:
                    query_date_end = last_avail_date

            # execute query / different queries depending on whether filtering on column names or getting all columns
            if col_names:
                output = conn.execute(sqlalchemy.select(*[table_obj.c[col] for col in col_names]).where((table_obj.c["Date"] >= query_date_start) & (table_obj.c["Date"] <= query_date_end))).fetchall()
            else:
                output = conn.execute(sqlalchemy.select(table_obj).where((table_obj.c["Date"] >= query_date_start) & (table_obj.c["Date"] <= query_date_end))).fetchall()
            
            # raise Exception if output empty (unless when raise_exception_on_error = False)
            if len(output) == 0:
                if raise_exception_on_error:
                    raise Exception(f'ERROR: could not query data from db for db_name: {db_name_or_engine}, table_name: {table_name}, query_date_start: {query_date_start}, query_date_end: {query_date_end}, col_names: {col_names}')
                else:
                    # return empty df (with col_names specified plus "Date" and "PondID") if query resulted in no data and raise_exception_on_error = False
                    return pd.DataFrame(columns=col_names)
        out_df = pd.DataFrame(output, columns=col_names)
        out_df = convert_df_date_cols(out_df, option='TO_DT') # convert the Date column from string (db representation) into datetime format
        return out_df

def convert_df_date_cols(df, option: str, dt_format: str = '%Y-%m-%d') -> pd.DataFrame:
    '''
    Helper function to convert the Date column in a pandas dataframe

    param:
    -option:
        - 'TO_STR': convert datetime column into string representation (for storing into db)
        - 'TO_DT': convert column into datetime representation (for data manipulation outside of db)

    It would be easier to just use a db that can handle datetime values...
    '''
    for col_name in df.columns:
        ### TODO: use regex to find 'date', not surrounded by any other alphabetic characers (so not just part of another word)
        if 'date' in col_name.lower():
            # when converting datetime to string
            # for inputting data into DB
            if option == 'TO_STR':
                # to ensure that all datetime columns are properly converted, check if each col is of 'datetime64' type
                if is_datetime64_any_dtype(df.dtypes[col_name]):
                    df[col_name] = df[col_name].dt.strftime(dt_format)
                else:
                    raise Exception(f"ERROR: tried to convert {col_name} from datetime to string, but it is not in datetime format!\n('date' substring in column name will cause automatic conversion attempt')")
            
            # when converting string to datetime
            # for extracting date from db (stored as strings)
            elif option == 'TO_DT':
                df[col_name] = pd.to_datetime(df[col_name], errors='coerce')

            else:
                raise Exception('ERROR: no datetime conversion "option" parameter specified!')
        
    return df
        

def check_active_query(db_engine: sqlalchemy.Engine, pond_id: str, check_date: str, num_prior_days_to_check: int = 5) -> None:
    check_date = datetime.strptime(check_date, '%Y-%m-%d')
    earliest_check_date = check_date - timedelta(days=num_prior_days_to_check)
    check_date_range = rrule(DAILY, dtstart=earliest_check_date, until=check_date)
    print(check_date_range)
    ponds_data_table = load_table(db_engine, 'ponds_data')
    temp_counter = 0
    with db_engine.begin() as conn:
        #for d in rrule(DAILY, dtstart=check_date, until=check_date - timedelta(days=num_prior_days_to_check)):   
        for d in check_date_range[::-1]:
            date_str = d.strftime('%Y-%m-%d')
          #  fo_data = conn.execute(sqlalchemy.text('SELECT "Fo" FROM ponds_data WHERE ponds_data."Date" = :date'), date=date_str).fetchall()
            fo_data = conn.execute(sqlalchemy.select(ponds_data_table.c["Fo"]).where((ponds_data_table.c["Date"] == date_str) & (ponds_data_table.c["PondID"] == pond_id))).fetchall()
            if fo_data[0][0] != None:
                return True
            temp_counter += 1
            print(f'TESTTESTTEST {temp_counter}:', fo_data)