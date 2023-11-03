import os
import sqlalchemy
import pandas as pd
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

def delete_existing_rows_ponds_data(db_engine: sqlalchemy.Engine, table_name: str, update_data_df: pd.DataFrame) -> None:
    if not check_if_table_exists(db_engine, table_name):
        return # return early if the table doesn't exists / do nothing
    table = sqlalchemy.Table(table_name, sqlalchemy.MetaData(), autoload_with=db_engine)
    data_rows = update_data_df.to_dict(orient='records')
    with db_engine.begin() as conn:
        for row in data_rows:
            conn.execute(table.delete().where(sqlalchemy.and_(table.c.Date == row['Date'], table.c.PondID == row['PondID'])))
    print('Deleted duplicate rows!')

def update_table_rows_from_df(db_engine: sqlalchemy.Engine, table_name: str, update_data_df: pd.DataFrame) -> None:
    print('Deleting existing rows from table...')
    delete_existing_rows_ponds_data(db_engine=db_engine, table_name=table_name, update_data_df=update_data_df)
    #Use DataFrame.to_sql() to insert data into database table
    update_data_df.to_sql(name=table_name, con=db_engine, if_exists='append', index=False)
    print('Updated DB!')

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
            return pd.DataFrame(output, columns=col_names)

def query_data_table_by_date_range(db_name_or_engine: str|sqlalchemy.Engine, table_name: str, query_date_start: datetime, query_date_end: datetime, col_names: list | None = None) -> pd.DataFrame:
    '''
    Helper function to query data from a database table for a specified date

    params
    ---------
    db_name: database name as string (not including path or file extension!)
    table_name: table name as a string
    query_date_start: datetime
    query_date_end: datetime
    col_names: list of column names (strings) to return or None
        - if None -> returns all columns??

    Returns
    --------
    pd.DataFrame
    '''
    # check if db_name_or_engine is a sqlalchemy.Engine instance...otherwise load the engine
    if not isinstance(db_name_or_engine, sqlalchemy.Engine):
        db_engine = sqlalchemy.create_engine(f"sqlite:///db/{db_name_or_engine}.db", echo=False)
    else:
        db_engine = db_name_or_engine
        
    query_date_start = query_date_start.strftime("%Y-%m-%d")
    query_date_end = query_date_end.strftime("%Y-%m-%d")

    if col_names:
        # add Date and PondID to col_names so that they are returned in the query (if any col names are provided at all - all columns are returned if not, so no need to append these)
        col_names.insert(0,'PondID')
        col_names.insert(0,'Date')
    
    if not check_if_table_exists(db_engine, table_name):
        print(f'ERROR: Could not load {table_name} from database: {db_name_or_engine}!')
    else:
        table_obj = load_table(db_engine, table_name)
        with db_engine.begin() as conn:
            if col_names:
                output = conn.execute(sqlalchemy.select(*[table_obj.c[col] for col in col_names]).where((table_obj.c["Date"] >= query_date_start) & (table_obj.c["Date"] <= query_date_end))).fetchall()
            else:
                output = conn.execute(sqlalchemy.select(table_obj).where((table_obj.c["Date"] >= query_date_start) & (table_obj.c["Date"] <= query_date_end))).fetchall()
            if len(output) == 0:
                raise Exception(f'ERROR: could not query data from db for db_name: {db_name_or_engine}, table_name: {table_name}, query_date_start: {query_date_start}, query_date_end: {query_date_end}, col_names: {col_names}')
            return pd.DataFrame(output, columns=col_names)

##### unused
# def query_data_table_by_pond_and_date(db_name: str, table_name: str, pond_id: str, date: datetime, col_names: list|None = None) -> list:
#     db_engine = sqlalchemy.create_engine(f"sqlite:///db/{db_name}.db", echo=False)
#     query_date = date.strftime("%Y-%m-%d")
#     if not check_if_table_exists(db_engine, table_name):
#         print(f'ERROR: Could not load {table_name} from database: {db_name}!')
#     else:
#         table_obj = load_table(db_engine, table_name)
#         with db_engine.begin() as conn:
#             output = conn.execute(sqlalchemy.select(*[table_obj.c[col] for col in col_names]).where((table_obj.c["PondID"] == pond_id) & (table_obj.c["Date"] == query_date))).fetchall()
#             print(output)


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