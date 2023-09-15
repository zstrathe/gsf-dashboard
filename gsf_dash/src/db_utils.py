import os
import sqlalchemy
from dateutil.rrule import rrule, DAILY 
from datetime import datetime, date
from pathlib import Path
from . import load_setting, EmailHandler
from .ms_account_connect import MSAccount, M365ExcelFileHandler

def get_db_table_columns(db_engine: sqlalchemy.Engine, table_name) -> list:
    table = sqlalchemy.Table(table_name, sqlalchemy.MetaData(), autoload_with=db_engine)
    inspector = sqlalchemy.inspect(table)
    return [col.name for col in inspector.columns]

def init_db_table(db_engine: sqlalchemy.Engine, table_name: str) -> None:
    table_schemas_path = Path('./settings/db_schemas')
    table_schemas_dir = os.listdir(table_schemas_path)
    if f'{table_name}.table_init' not in table_schemas_dir:
        print(f'ERROR: table schema for {table_name} not found! Add sqlite table create statement to .settings/db_schemas/ folder as .table_init file')
        return

    # load the CREATE TABLE statement from file
    with open(table_schemas_path / f'{table_name}.create_table', 'r') as file:
        table_schema = file.read()

    table_already_exists = sqlalchemy.inspect(db_engine).has_table(table_name) # boolean value depending on table_name existence in db
    
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

def get_primary_keys(db_table: sqlalchemy.Table):
    inspector = sqlalchemy.inspect(db_table)
    # Inspector.primary_key -> Column objects that are primary keys for table -> column.name
    return [k.name for k in inspector.primary_key]

def delete_existing_rows_ponds_data(db_engine: sqlalchemy.Engine, table_name: str, data_rows: list) -> None:
    table = sqlalchemy.Table(table_name, sqlalchemy.MetaData(), autoload_with=db_engine)
    with db_engine.begin() as conn:
        for row in data_rows:
            conn.execute(table.delete().where(sqlalchemy.and_(table.c.Date == row['Date'], table.c.PondID == row['PondID'])))
    print('Deleted duplicate rows!')