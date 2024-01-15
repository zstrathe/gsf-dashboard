# Green Stream Farms Dashboard 
A module for ETL & ELT processes to extract data from various sources (mostly Excel files stored on MS Sharepoint), and to generate daily "dashboard" reports from the stored data.

## Components:
## automation/dl_gen_email.sh:
  - Bash/shell script for running automated as a cron job on Linux systems
    - Extracts the current date from system and runs gsf_dash/__main__.py with date as a param

## gsf_dash/__main__.py: 
  - Python script for command line execution of module
    - Utilizes try/except blocks for execution of individual components in order to capture 'traceback' data, and sends an email notification with traceback data
    - If execution of all components is successful, emails output files to a distribution list
      - params:
        - --date (-d) : Date to run module (in 'yyyy-mm-dd' string format)
        - --test_run (-t) : OPTIONAL : if set to true, then runs module as a test (sends output data to a list of test addresses)

## gsf_dash/src/dataloader.py:
  - Python script for running a combination of ETL and ELT processes to extract data and store in a relational database  
    - Utilizes base class/subclass structure for individual processes
    - Dataloader class: main class to be instantiated:
      - params: 
        - run_date: 'yyyy-mm-dd' string OR None
          - If a run_date provided, then runs every subclass of DBColumnsBase, utilizing a queue and checking for dependencies (and re-shuffling the queue as needed)
        - db_engine: string for database name (used to create an instance of a sqlalchemy.Engine)
    - DBColumnsBase: base class for each ETL/ELT process
      - each subclass should have constants:
        - OUT_TABLE_NAME: name of database table that data will be stored in
        - DEPENDENCY_CLASSES: list of processes/classes that should be run first (i.e., processes that this process is dependent on data from, before it can be run)
      - params: 
        - db_engine: string of engine name
        - run_date_start: datetime.datetime object for the begin of date range to collect data for
        - run_date_end: datetime.datetime object for the end of date range to collect data for
      - properties: 
        - MIN_LOOKBACK_DAYS: integer : minimum number of days to "look back." Default is set to 5, so gathers data for the current date and previous 5 days. Overrides the run_date_start param if the date range provided for run_date_start and run_date_end are less than MIN_LOOKBACK_DAYS) 

## gsf_dash/src/db_utils.py:
  - Utility functions for interacting with database (currently in SQLite format for easy portability, but may need to change to PostreSQL for potential cloud deployment/migration)
    - init_db_table():
      - function for initializing a database table, so that data tables are explicitly defined
      - First checks if table exists, then gets user confirmation to drop and overwrite if it does already exist
        - utilizes settings/db_schemas/{table_name}.create_table file which includes CREATE TABLE IF NOT EXISTS statement defining column names, datatypes, primary key columns
    - update_table_rows_from_df(): 
      - function for adding/updating rows in table
      - automatically converts dates into string (for storing as text in SQLite db)
      - 1st step: inspects the data payload (a Pandas DataFrame) for missing dates in the date range, and adds in necessary rows with Null values
      - 2nd step: checks if rows all exist in the table for every primary key pair (Date and PondID for tables that include individual pond data, or just Date for aggregate data), and adds a blank row for the primary key pair as necessary (all in a single update statement)
      - 3rd step: checks that all columns in the data payload are already present in the database table, and raises an exception if not
      - 4th step: uses pandas df.to_sql() function to insert a '__temp_table' into the DB with the data payload dataframe
      - 5th step: executes a row insert statement for primary key pairs (as needed) into the target table
      - 6th step: executes an UPDATE query to copy data from the '__temp_table' into the target table
      - 7th step: checks if the SQL result rowcount is equal to the data payload rowcount, and rollsback the SQL transaction if they are not equal 
    - query_data_table_by_date_range():
      - function for querying data from the DB
      - checks for Date and PondID columns so they do not need to be specified as params and are always included in output of query
      - automatically converts Date column from string (db storage format) to datetime
      - includes optional params for: 
        - column names: if a list provided, only queries those columns (plus Date | Date & PondID by default), otherwise default queries all columns in table
        - raise_exception_on_error: if no output is returned from SQL query, whether to raise an Exception or just return None
        - check_safe_date (to override param if necessary so that query doesn't fail if outside of date range in db)

## gsf_dash/src/ms_account_connect.py:
  - For authenticating and interacting with the Microsoft365 API, utilizing O365 Python module (https://github.com/O365/python-o365)
    - MSAccount() class:
      - main class for authentication, downloading files, etc.
      - easiest auth method (using basic 'User' auth in Azure settings), is user authentication, utilizing stored data from browser 
        - requires manually running MSAccount(auth_manual=True) to initially setup and store the authentication token
        - need to setup a 'client_id' (from app registration info) and 'client_secret' (from a 'Client secret' setup in App registrations -> Certificates & secrets)
        - Client secret will expire and needs to be setup again (maximum Azure allows is 24 months)
      - also have functionality setup for authentication with an encrypted certificate (generate and upload to Azure under App registrations -> Certificated & secrets)
      - interative_view_sharepoint_data():
        - helper function for getting object_id's from sharepoint folders/items, because it's an easy way to store settings for getting files
        - interactively step through directories 
    - M365ExcelFileHandler() class:
      - class for extracting data from Excel workbooks, using the M365 API
      - intended functionality is to interact/update data within the M365 Workbook objects, and it gathers info from the API regarding the last row, last column, etc. 
      - for simply downloading and extracting data from an excel file, it's easier to just use the MSAccount() class to download, then load the file with Pandas
    - EmailHandler() class:
      - class for handling sending emails, as well a function for extracting file attachments from specified emails

## gsf_dash/src/utils.py:
  - Helper functions for misc purposes
    - load_setting():
      - load setting from /settings/settings.cfg file
      - uses ConfigParser() to read the file and returns the specified_setting as a dict
    - generate_multipage_pdf():
      - for generating Matplotlib figures (provided as a list) into a multipage pdf
    - redirect_logging_to_file()
      - decorator function to redirect stdout (using contextlib.redirect_stdout) to a file
      - intended use is for logging useful data to file (such output of initializing database or updating database) 

## gsf_dash/src/generate_overview_abstractclasses:
  - Generates reports, utilizing abstract base classes to generate the structure of the reports, with default properties that can be adjusted for subclass instances	
    - BaseGridReport() class:
      - base class for plotting data on a 'grid' layout (corresponding to the physical layout of the ponds)
      - required functions for each subclass:
        - load_data(): function to load data for the report
        - plot_each_pond() function to plot each pond on the report
        - plot_annotations() function to plot various annotations on the report (to this generally should be utilizing figure coordinates to plot things)
    - ExpenseGridReport() class:
      - report utilizing BaseGridReport as a base class
      - reporting calculated (estimated) harvest mass and calculated expenses per pond
      - reporting aggregate data for harvest mass & expenses
    - BaseTableReport() class:
      - base class for plotting data in table(s)
      - has property for 'inter_table_spacing' if multiple tables are provided
      - required functions for each subclass:
        - load_data(): load data and return in dict format as {'title': table_title_string, 'df': pandas DataFrame} for each individual table to plot
    - PotentialHarvestsReport(): 
      - IN PROGRESS: currently the running version of "Potential Harvests" report ... need to update to utilize the BaseTableReport() class
		
## gsf_dash/src/graph_reports.py:
  - Generates reports, utilzing abstract base class, for time-series graphs
    - BaseTimeSeriesGraphReport():
      - IN PROGRESS: base class for time-series graph reports
		
## gsf_dash/src/generate_overview.py:
  - Currently generating reports for 'Pond Health Overview' and 'Pond EPA Overview'		
    - IN PROGRESS: converting these to reports utilizing abstract base classes 	
