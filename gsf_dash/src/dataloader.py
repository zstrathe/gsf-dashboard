from abc import ABC, abstractmethod
import functools
import math
import time
import re
from datetime import datetime
from datetime import date as date_util
from pathlib import Path
from dateutil.rrule import rrule, DAILY
import pandas as pd
import numpy as np
import sqlalchemy

from .ms_account_connect import MSAccount, M365ExcelFileHandler
from .db_utils import (
    init_db_table,
    query_data_table_by_date_range,
    check_if_table_exists,
    get_db_table_columns,
    update_table_rows_from_df,
)
from .utils import load_setting, redirect_logging_to_file


class Dataloader:
    def __init__(self, run_date: str | None, db_engine_name: str = "gsf_data"):
        """
        params:
            - run_date:
                - date formatted as str (yyyy-mm-dd) or None
                - if not None, will process as a DAILY db update
            - db_engine: sql alchemy engine name (str)
        """
        self.db_engine_name = db_engine_name
        # initialize sqlalchemy engine / sqlite database
        self.db_engine = sqlalchemy.create_engine(
            f"sqlite:///db/{db_engine_name}.db", echo=False
        )

        if run_date:
            # Normalize run_date to remove potential time data and prevent possible key errors when selecting date range from data
            self.run_date = pd.to_datetime(run_date).normalize()
            # run method to load data on a daily basis
            self.main_run_each_proc(run_date_start=run_date, run_date_end=run_date)

    def main_run_each_proc(
        self, run_date_start: datetime, run_date_end: datetime
    ) -> None:
        """
        Method to tie together daily processes to run for downloading/loading/processing data and loading into a db table
        """
        param_kwargs = {
            "db_engine": self.db_engine,
            "run_date_start": run_date_start,
            "run_date_end": run_date_end,
        }

        # init queue of subclasses of DBColumnsBase, which should all be ETL processes to extract & load data for one or more columns in the DB
        # MAYBE TODO: currently each process calls a method to update the corresponding rows in the DB table specified, need to collect data (by table) from each process and minimize DB update calls???
        data_etl_queue = DBColumnsBase.__subclasses__()
        # init empty list to store string names of classes successfully run
        completed_etl_name_list = []
        # init a dict to count how many times a specific process cannot be run due to a dependency (raise Exception if > 1)
        reset_queue_counter = {p.__name__: 0 for p in data_etl_queue}
        # define limit for how many times a process can attempt to run before a 'Failure' state is reached
        # using a very conservative\high limit of the length of all other items in the queue
        # (higher limit shouldn't be an issue because the process will just quickly fail over and over)
        reset_queue_limit = len(data_etl_queue) - 1

        while data_etl_queue:
            # get first item in queue & remove it
            data_etl_class_obj = data_etl_queue.pop(0)
            class_vars = vars(data_etl_class_obj)
            if (
                "DEPENDENCY_CLASSES" in class_vars
                and class_vars["DEPENDENCY_CLASSES"] is not None
            ):
                # list of class names (strings)
                dep_classes = class_vars["DEPENDENCY_CLASSES"]
                reset_queue_flag = False
                for c in dep_classes:
                    if c not in completed_etl_name_list:
                        print(
                            f"Detected dependency class needed, cannot run: {data_etl_class_obj.__name__}. Re-appending to end of processing queue!"
                        )
                        reset_queue_counter[data_etl_class_obj.__name__] += 1
                        # raise exception when reset_queue_limit exceeded
                        if (
                            reset_queue_counter[data_etl_class_obj.__name__]
                            > reset_queue_limit
                        ):
                            raise Exception(
                                f'ERROR: could not run process name: [{data_etl_class_obj.__name__}] for run dates: [{run_date_start.strftime("%Y-%m-%d")} - {run_date_end.strftime("%Y-%m-%d")}], due to dependency process: [{c}] not existing??'
                            )
                        data_etl_queue.append(data_etl_class_obj)
                        reset_queue_flag = True
                        break
                if reset_queue_flag:
                    continue
            print("\n*** Running:", data_etl_class_obj.__name__, "***")
            # init/run the data class object
            data_etl_class_obj(**param_kwargs)
            completed_etl_name_list.append(data_etl_class_obj.__name__)

        print("Finished with daily data updates to DB!")

    def rebuild_db(self, start_date: str, end_date: str) -> None:
        """
        Method to fully re-build the daily data database from "start_date" to the  "end_date"

        start_date: specify by "YYYY-MM-DD" - earliest possible date with data is 2017-03-15
        end_date: specify by "YYYY-MM-DD"
        """
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date, "%Y-%m-%d")

        # initialize database tables from CREATE TABLE statements stored in ./settings/db_schemas/
        # could just let pd.to_sql() create the tables, but this allows additional flexibility with setting contstraints, etc.
        def init_tables():
            db_schema_files = [
                f.name.split(".")[0]
                for f in Path().glob("settings/db_schemas/*.create_table")
            ]
            for table_name in db_schema_files:
                init_db_table(self.db_engine, table_name)

        # use decorator from .utils.py to redirect stdout to a log file, only while building the DB
        @redirect_logging_to_file(
            log_file_directory=Path("db/logs/"),
            log_file_name=f'{date_util.today().strftime("%y%m%d")}_rebuild_db_{self.db_engine_name}_{start_date.strftime("%y%m%d")}_{end_date.strftime("%y%m%d")}.log',
        )
        def rebuild_db_run(start_date, end_date):
            # removed_cols_list = []
            # for d in rrule(DAILY, dtstart=start_date, until=end_date):
            # print('\n\nLoading', datetime.strftime(d, "%m/%d/%Y"))
            ##########[removed_cols_list.append(removed_col) for removed_col in self.load_daily_data(db_engine=db_engine, specify_date=d, return_removed_cols=True) if removed_col not in removed_cols_list ]
            ## TODO: re-implement tracking of all columns removed from files processed
            self.main_run_each_proc(run_date_start=start_date, run_date_end=end_date)
            print("Finished building DB!!")
            # print('Removed columns:', removed_cols_list)

        init_tables()
        rebuild_db_run(start_date, end_date)

    def rebuild_data_class(self, class_name: str) -> None:
        """
        TODO:
            - get dependent classes from data class
            - get existing data date range and use for rebuilding
                - add params for specifying dates
                - if existing data doesn't exist, throw exception unless dates specified
        """

        try:
            data_class_obj = globals().get(class_name)
        except KeyError:
            raise ValueError(
                f"Could not get class name: {class_name}! Wrong name provided?"
            )
        param_kwargs = {
            "db_engine": self.db_engine,
            "run_date_start": datetime(2023, 9, 1),
            "run_date_end": datetime(2023, 12, 28),
        }
        # init/run the data class object
        data_class_obj(**param_kwargs)

    def _update_existing_db_new_tables(self) -> None:
        db_schema_files = [
            f.name.split(".")[0]
            for f in Path().glob("settings/db_schemas/*.create_table")
        ]
        for table_name in db_schema_files:
            # True if table exists, False otherwise
            if not check_if_table_exists(self.db_engine, table_name):
                init_db_table(self.db_engine, table_name)


class DBColumnsBase(ABC):
    # turn off formatting for black formatter to avoid list with line for each item
    # fmt: off
    ponds_list = [
        "0101", "0201", "0301", "0401", "0501", "0601", "0701", "0801", "0901", "1001", "1101", "1201",
        "0102", "0202", "0302", "0402", "0502", "0602", "0702", "0802", "0902", "1002", "1102", "1202",
        "0103", "0203", "0303", "0403", "0503", "0603", "0703", "0803", "0903", "1003", "1103", "1203",
        "0104", "0204", "0304", "0404", "0504", "0604", "0704", "0804", "0904", "1004", "1104", "1204",
        "0106", "0206", "0306", "0406", "0506", "0606", "0706", "0806", "0906", "1006", 
        "0108", "0208", "0308", "0408", "0508", "0608", "0708", "0808", "0908", "1008"
    ]
    # fmt: on

    # CLASS FOR HANDLING MS 365 API INTERACTIONS
    account = MSAccount()

    def __init__(
        self,
        db_engine: sqlalchemy.Engine,
        run_date_start: datetime,
        run_date_end: datetime,
        run: bool = True,
    ):
        """
        params:
            db_engine: sqlalchemy Engine
            run_date_start: datetime
                - if range between run_date_start and run_date_end is < self.MIN_LOOKBACK_DAYS (default=5),
                  then run_date_start is adjusted so that the date range is equal to self.MIN_LOOKBACK_DAYS
            run_date_end: datetime
            run: FOR TESTING - True by default
        """
        if not hasattr(self, "OUT_TABLE_NAME"):
            raise Exception(
                f"ERROR: attempted to create a DB Table class without first setting OUT_TABLE_NAME for {self.__class__}!"
            )

        self.db_engine = db_engine
        self.run_date_start = run_date_start
        self.run_date_end = run_date_end
        run_range_days = (self.run_date_end - self.run_date_start).days
        if run_range_days < self.MIN_LOOKBACK_DAYS:
            self.run_date_start = self.run_date_start - pd.Timedelta(
                days=self.MIN_LOOKBACK_DAYS - run_range_days
            )

        if run:
            self.run()

    # set MIN_LOOKBACK_DAYS property
    # this is the minimum days that will be included in each data class update
    # i.e., when running once daily, run_date_start == run_date_end, so update for additional dates
    @property
    def MIN_LOOKBACK_DAYS(self) -> int:
        if not hasattr(self, "_MIN_LOOKBACK_DAYS"):
            self._MIN_LOOKBACK_DAYS = 10
        return self._MIN_LOOKBACK_DAYS

    @MIN_LOOKBACK_DAYS.setter
    def MIN_LOOKBACK_DAYS(self, val: int) -> None:
        self._MIN_LOOKBACK_DAYS = val

    @abstractmethod
    def run(self) -> None:
        """
        Extract, transform, and load data (Or just Extract and Load) into database table (self.OUT_TABLE_NAME)
        """
        pass

    def _dl_sharepoint_files(self, file_identifier_str: list[str] | str) -> dict:
        """
        params:
            file_identifier_str_list:
                - list of strings corresponding to key in settings.cfg file under 'file_ids' category
                - OR a single string (which is converted to a list of length 1)

        output:
            dictionary:
                - key: file idenfifier string (same as key in settings.cfg file)
                - value: pathlib.Path object corresponding to downloaded file
        """
        # convert to a list if only a single string param provided
        if isinstance(file_identifier_str, str):
            file_identifier_str = [file_identifier_str]

        # get file IDs from settings file, get only the values from the dict that's returned
        file_ids = {
            k: v
            for (k, v) in load_setting("file_ids").items()
            if k in file_identifier_str
        }

        out_file_paths = {}
        for label, file_id in file_ids.items():
            # returns file path, else returns None if failed
            tmp_file_path = self.account.download_sharepoint_file_by_id(
                object_id=file_id,
                name=f"{label}.xlsx",
                to_path=Path("data_sources/tmp/"),
            )
            if tmp_file_path is not None:
                out_file_paths[label] = tmp_file_path
            else:
                raise Exception(
                    f"ERROR: failure downloading data for {label}: {file_id}!"
                )
        return out_file_paths


class DailyDataLoad(DBColumnsBase):
    """
    Load primary ponds data into 'ponds_data' table
    """

    OUT_TABLE_NAME = "ponds_data"
    DEPENDENCY_CLASSES = None

    def __init__(self, *args, **kwargs):
        self.OUT_TABLE_NAME = "ponds_data"

        # init as base class after setting OUT_TABLE_NAME
        super().__init__(*args, **kwargs)

    def run(self):
        daily_data_dfs = []

        # Call method to process and load into db the daily data files
        daily_data_dfs.append(
            self._load_daily_data_date_range(
                dt_start=self.run_date_start, dt_end=self.run_date_end
            )
        )

        # Combine "base" and "daily data" dfs
        joined_daily_df = functools.reduce(
            lambda df1, df2: pd.merge(df1, df2, on=["Date", "PondID"], how="outer"),
            daily_data_dfs,
        )

        # update the db (calling a function from .db_utils.py that deletes the existing db table row (based on Date&PondID cols) and appends the new row from the update_data_df)
        update_table_rows_from_df(
            db_engine=self.db_engine,
            db_table_name=self.OUT_TABLE_NAME,
            update_data_df=joined_daily_df,
        )

    def _load_daily_data_date_range(self, dt_start: datetime, dt_end: datetime):
        day_dfs = []
        removed_cols_full = []
        for idx, d in enumerate(rrule(DAILY, dtstart=dt_start, until=dt_end)):
            d_df, removed_cols_list = self._load_daily_data_file(specify_date=d)
            day_dfs.append(d_df)
            [
                removed_cols_full.append(col_name)
                for col_name in removed_cols_list
                if col_name not in removed_cols_full
            ]
        out_df = pd.concat(day_dfs, axis=0, join="outer")
        print(
            f'\nAll columns ignored from Daily Data files for date range {dt_start.strftime("%-m/%-d/%Y")} to {dt_end.strftime("%-m/%-d/%Y")}:\n{removed_cols_full}\n'
        )
        return out_df

    def _load_daily_data_file(
        self, specify_date: datetime, return_removed_cols=False
    ) -> (pd.DataFrame, list):
        """
        Load the daily data file path, must be an excel file!
        Data is indexed by pond_name in multiple sheets within the excel file
        This function combines data between all sheets into single dataframe and loads into database
        """
        excel_file = self._download_daily_data_file(
            specify_date=specify_date.strftime("%Y-%m-%d")
        )
        # file doesn't exist or download error
        if excel_file is None:
            # if no daily data is found for specified date, then insert blank rows into database table (one for each Pond)
            excel_dataframes = {
                "empty data": pd.DataFrame(self.ponds_list, columns=["PondID"])
            }
        else:
            print(f"Loading daily data for date: {specify_date}")
            # load sheet and parse Pond label columns as strings to preserve leading zeros (otherwise i.e. 0901 would turn into 901)
            excel_dataframes = {
                sheet_name: excel_file.parse(
                    sheet_name, converters={"Pond": str, "Pond ID": str}
                )
                for sheet_name in excel_file.sheet_names
            }

        # extracted allowed columns from the DB table
        # Use these columns in the database of daily data for each pond
        # will need to rebuild the database if altering columns

        # daily_data_cols = ['Date', 'Time Sampled', 'Depth', 'Temp', 'Handheld pH', 'PLC pH', 'Afternoon PLC', 'Calibration Needed', 'CO2 psi', 'Pond Color', 'ppm UAN-32 Added', 'Volume UAN-32 Added',
        #                    'ppm 10-34 Added', 'Volume 10-34 Added', 'Volume Trace Added', 'Volume Iron Added', 'ppm Bleach Added', 'Volume Bleach Added', 'ppm Cal Hypo Added', 'kg Cal Hypo Added',
        #                    'Media Only', 'Column', 'Fo', 'Fm', 'Yield', 'Salinity', 'TOC-AFDW', 'TOC', 'IC', 'TC', 'Filter DW', 'Filter AFDW', '% Ash', 'Alkalinity', 'TN', 'PO4', 'Notes-QAQC',
        #                    'Rotifers', '% Nanno', 'Morphology', 'Dead Cells', 'Flocculation', 'Detritus', 'Attached FD111', 'Free Floating FD111', 'Golden Flagellates', 'Diatoms', 'Tetra', 'Green Algae',
        #                    'Chlamy', 'Ciliates', 'Flagellates', 'Segmented Bacteria', 'Rod Shaped Bacteria', 'Spirochetes', 'Amoeba', '% Ash Relative to DW', 'Notes-Crop Protection', 'ppm Benzalkonium Added',
        #                    'Volume Benzalkonium Added', 'Sodium Chloride', 'Sodium Bicarb', 'Magnesium Sulfate', 'Potassium Chloride', 'Calcium Chloride']

        allowed_cols = get_db_table_columns(self.db_engine, self.OUT_TABLE_NAME)
        # keep track of used columns and the sheet that they appear on (for checking whether there is duplicate data or inaccurate data (such as same column name between sheets with different values...shouldn't happen and might indicate bad data)
        used_cols = {}
        sheets_data = {}
        all_removed_cols = []
        # extract data from each sheet create a composite dataframe for each pond
        for idx, (sheet_name, df) in enumerate(excel_dataframes.items()):
            print("Processing", sheet_name, "...")
            ### Column name cleaning
            df = df.loc[
                :, ~df.columns.str.contains("^Unnamed")
            ]  # drop columns without a header, assumed to be empty or unimportant
            # remove information in parentheses from column names (some contain UOM info like "(g/L)")...maybe extract and store in a separate table for column UOM info...but just remove for now
            # additionally, keep everything if "(filter)" is in the name since that is a way of differentiating some measurements...may potentially need to add more exceptions and make this more robust
            # defin exemptions for column tags to exempt from removing / keep these in the column name if they exist
            tag_strip_exemptions = ("(filter)", "(filtered)")
            df.columns = [
                re.sub(r"\([^)]*\)", "", colname)
                if not any(
                    [substr in colname.lower() for substr in tag_strip_exemptions]
                )
                else colname
                for colname in df.columns
            ]
            # strip leading or trailing whitespace from column names
            df.columns = df.columns.str.strip()
            # append sheet_name to Notes and Comments fields since those will be distinct per each sheet, and rename "Pond ID" to "Pond" if necessary
            df = df.rename(
                columns={
                    "Notes": f"Notes-{sheet_name}",
                    "Comments": f"Comments-{sheet_name}",
                    "Pond": "PondID",
                    "Pond ID": "PondID",
                    "AFDW (Filter)": "Filter AFDW",
                }
            )
            ### Skip sheet if there isn't a "PondID" column present
            # - Do this after cleaning column names due to renaming "Pond" and "Pond ID" columns to "PondID"
            # - might have issues if any sheets don't have the headers starting on the first row
            if "PondID" not in df.columns:
                print("Not a valid data sheet?? Skipping...")
                continue

            ### Clean up "PondID" column values
            # drop rows with NaN values in the 'PondID' column
            df = df.dropna(subset="PondID")
            # set Pond to string since some fields may be numerical, but need to perform string operations on this field
            df["PondID"] = df["PondID"].astype(str)
            # extract "Media Only" field from the Pond label (for ex, might be: "1204 (Media Only)" in the source spreadsheet file)
            df["Media Only"] = df["PondID"].apply(
                lambda label: True if "media only" in label.lower() else False
            )
            # remove the "Media Only" tag from pond labels if it's present
            df["PondID"] = df["PondID"].apply(
                lambda val: (
                    re.sub(r"\(media only\)", "", val, flags=re.IGNORECASE)
                ).strip()
            )

            ### Fill in rows with missing PondID's
            # Check that each Pond is listed, add it as a blank row if not
            [
                df.append({"PondID": pond_name}, ignore_index=True)
                for pond_name in self.ponds_list
                if pond_name not in df["PondID"]
            ]

            ### Miscellaneous Column Cleaning
            # get column number as a string (last digit of pond name)
            df["Column"] = df["PondID"].str[-1]
            # overwrite date column regardless if it exists already
            df["Date"] = specify_date
            # format "Time Sampled" column as a string if present (due to potential typos causing issues with some values reading as Datetime and some as strings)
            if "Time Sampled" in df:
                df["Time Sampled"] = df["Time Sampled"].apply(
                    lambda x: str(x) if not pd.isna(x) else None
                )
            if "Filter AFDW" in df:
                # replace zero values in AFDW column with None, as zero values may cause issues with other calculated fields
                df["Filter AFDW"] = df["Filter AFDW"].replace(0, None)

            ### Set "PondID" as index column of each sheet (required for properly merging sheets) and sort the df by PondID
            df = df.set_index("PondID").sort_values(by="PondID")

            # for scorecard sheet, drop everything but the "Comments-Scorecard" column, since every other data point in this sheet is coming from another sheet via excel formula
            if sheet_name == "Scorecard":
                df = df[["Comments-Scorecard"]]
            else:
                ### Drop columns not in allowed_cols list
                removed_cols = df.columns.difference(allowed_cols)
                check_cols = [
                    "Sodium Chloride",
                    "Sodium Bicarb",
                    "Magnesium Sulfate",
                    "Potassium Chloride",
                    "Calcium Chloride",
                ]
                if any([check_name in removed_cols for check_name in check_cols]):
                    print(
                        "\n\n*************************************************************************************** FOUND ONE OF  'Sodium Chloride', 'Sodium Bicarb', 'Magnesium Sulfate', 'Potassium Chloride', 'Calcium Chloride' *******************************************\n\n"
                    )
                [
                    all_removed_cols.append(col)
                    for col in removed_cols
                    if col not in all_removed_cols
                ]
                df = df[df.columns.intersection(allowed_cols)]

                # Handle duplicate columns and drop duplicates, but raise an exception when mismatching data is found for the same column name
                for col in df.columns:
                    # print('Checking column', col, '... col already seen?', col in used_cols)
                    if col not in used_cols:
                        used_cols[col] = sheet_name
                    else:
                        # get other column data (as a Pandas Series object) with 0's and NaN values removed for easier comparison
                        other_ = sheets_data[used_cols[col]][col][
                            lambda x: (x != 0) & (x.notna())
                        ]
                        # get current column data with 0's and NaN values removed for easier comparison
                        current_ = df[col][lambda x: (x != 0) & (x.notna())]

                        # if the other column is completely empty, drop the other column and set the used_column to the current one being checked (sheet_name of the column used)
                        if other_.empty:
                            sheets_data[used_cols[col]] = sheets_data[
                                used_cols[col]
                            ].drop([col], axis=1)
                            # set the current sheet name for the "used_cols" dict which tracks which sheet a column of data is being sourced from
                            used_cols[col] = sheet_name
                        # if both columns are equal or current column is empty, then drop the current column
                        elif other_.equals(current_) or current_.empty:
                            df = df.drop([col], axis=1)
                        else:
                            print(
                                f"ERROR: mismatching column data between sheets: {used_cols[col]} & {sheet_name}, column: {col}"
                            )
                            print("removing shorter column...")
                            if len(other_) >= len(current_):
                                # if the other column data is longer or same as current col, then drop the current col
                                # this will drop the current col if both sets of data are the same length (but mismatching)
                                # unfortunately there's no way to determine which data would be "correct" if they don't match but are same length, so just assume the first column processed was the correct
                                df = df.drop([col], axis=1)
                            else:
                                # else drop the "other" column
                                sheets_data[used_cols[col]] = sheets_data[
                                    used_cols[col]
                                ].drop([col], axis=1)
                                # set the current sheet name for the "used_cols" dict which tracks which sheet a column of data is being sourced from
                                used_cols[col] = sheet_name

            sheets_data[sheet_name] = df

        print("Columns ignored from file:", all_removed_cols)

        # compute a column length check value in case of bugs from handling and dropping duplicate columns
        # raise an exception in case of an error
        unique_columns = []
        [
            [
                unique_columns.append(label)
                for label in list(df.columns)
                if label not in unique_columns
            ]
            for df in sheets_data.values()
        ]
        column_length_check_ = len(unique_columns)

        # use functools.reduce to iteratively merge sheets, keeping all unique columns from each sheet (outer join), join on the 'PondID' column as primary key
        joined_df = functools.reduce(
            lambda sheet1, sheet2: pd.merge(sheet1, sheet2, on="PondID", how="outer"),
            sheets_data.values(),
        )

        if column_length_check_ != len(joined_df.columns):
            raise Exception("ERROR WITH MERGING DAILY DATA SHEETS!")

        joined_df = joined_df.reset_index()
        return joined_df, all_removed_cols

    def _download_daily_data_file(self, specify_date: str) -> pd.ExcelFile | None:
        """
        Find and download the "Daily Data" .xlsx file by date (yyyy-mm-dd format)
        This method looks through "Daily Data" sharepoint directory organized by: YEAR --> MM_MONTH (ex: '04-April') --> FILE ("yyyymmdd Daily Data.xlsx")

        params:
        --------
        - specify_date: - date to get daily data for (defaults to self.run_date if not specified)
                        - must be in "yyyy-mm-dd" format

        RETURNS -> pd.ExcelFile object (if successful download) or None (if failure)
        """
        # convert specify_date to datetime value
        specify_date = datetime.strptime(specify_date, "%Y-%m-%d")

        folder_id = load_setting("folder_ids")["daily_data"]
        for year_dir in self.account.get_sharepoint_file_by_id(folder_id).get_items():
            # first look for the "year" directory
            if year_dir.name == str(specify_date.year):
                # look within the year subdirectory
                for month_dir in year_dir.get_items():
                    if month_dir.name.lower() == specify_date.strftime("%m_%B").lower():
                        for daily_file in month_dir.get_items():
                            # search for filename with regex: case-insensitive: ["yyyymmdd" (1 or more whitespace chars) "daily data" (zero or more whitespace chars) ".xlsx"]
                            # use re.search to find 3 groups in filename: date formatted as "yyyymmdd", "daily data" (case insensitive), ".xlsx" (must be at end)
                            file_search = re.search(
                                r"(?i)({})\s+(Daily Data).*(\.xlsx$)".format(
                                    specify_date.strftime("%y%m%d")
                                ),
                                daily_file.name,
                            )
                            if file_search:
                                print("FOUND FILE:", daily_file.name)
                                dl_path_dir = Path("data_sources/tmp/")
                                dl_file_path = dl_path_dir / daily_file.name
                                download_attempts = 5
                                while True:
                                    daily_file.download(to_path=dl_path_dir)
                                    if dl_file_path.is_file():
                                        # load excel file
                                        excel_file = pd.ExcelFile(
                                            dl_file_path.as_posix()
                                        )
                                        # delete file after loading
                                        dl_file_path.unlink()
                                        return excel_file
                                    else:
                                        download_attempts -= 1
                                        if download_attempts > 0:
                                            print(
                                                "Waiting 5 seconds to try download again..."
                                            )
                                            import time

                                            time.sleep(5)
                                            continue
                                        else:
                                            raise Exception(
                                                f'ERROR: located but could not download daily data file for {datetime.strftime(specify_date, "%m/%d/%Y")}!'
                                            )

                        print(
                            f'COULD NOT FIND DAILY DATA FILE FOR {datetime.strftime(specify_date, "%m/%d/%Y")}!'
                        )
                        return None


class ScorecardDataLoad(DBColumnsBase):
    """
    Load scorecard data into 'ponds_data' table
    """

    OUT_TABLE_NAME = "ponds_data"
    DEPENDENCY_CLASSES = None

    def __init__(self, *args, **kwargs):
        # init as base class
        super().__init__(*args, **kwargs)

    def run(self):
        # Get data from scorecard file and add/update the database table for those entries ("Comments-Scorecard" & "Split Innoculum" [to indicate when a pond is harvested or split])
        # Reload data for past 5 days (in case any of it changed)
        sc_df = self._load_scorecard_data(
            begin_date=self.run_date_start, end_date=self.run_date_end
        )

        # update the db
        update_table_rows_from_df(
            db_engine=self.db_engine,
            db_table_name=self.OUT_TABLE_NAME,
            update_data_df=sc_df,
        )

    def _load_scorecard_data(self, begin_date: datetime, end_date: datetime):
        # save loaded file as a class attribute in case of using multiple successive calls to this method
        if not hasattr(self, "_sc_df"):
            print("Loading scorecard file...")
            file_id = load_setting("file_ids").get("scorecard")
            sc_file = M365ExcelFileHandler(
                file_object_id=file_id,
                load_data=True,
                data_get_method="DL",
                ignore_sheets=[
                    "Analysis",
                    "Template",
                    "Notes",
                    "SUBSTRING[(old)]",
                    "SUBSTRING[HRP]",
                ],
            )

            # define the columns to return from when loading
            include_columns = ["Date", "Comments", "Split Innoculum"]

            # because data for each pond is on a separate sheet, iterate through sheets (named by PondID) and get a df for each
            # then append each df to all_df_list, and finally concatenate into a single df by row
            all_df_list = []
            for pond_id in self.ponds_list:
                try:
                    pond_df = sc_file.sheets_data.get(pond_id).df
                except AttributeError:
                    print(
                        f"No scorecard data available for PondID: {pond_id}. Skipping..."
                    )
                    continue
                # rename the first column to date, since it's always the pond id listed here on these sheets (but it's the date index col)
                pond_df.columns.values[0] = "Date"
                # datetime values should already exist, but this converts any errors into a NaT value with errors='coerce'
                pond_df["Date"] = pd.to_datetime(pond_df["Date"], errors="coerce")
                # drop na values from Date column (after forcing error values into a NaT value)
                pond_df = pond_df.dropna(subset=["Date"])
                # filter columns
                pond_df = pond_df[include_columns]
                # add PondID column for DB table indexing
                pond_df["PondID"] = pond_id
                # rename Comments column to be consistent with DB
                pond_df = pond_df.rename(columns={"Comments": "Comments-ScorecardFile"})
                # combine dataframes for each PondID into one large df (combine by rows)
                all_df_list.append(pond_df)
            ScorecardDataLoad._sc_df = pd.concat(all_df_list, axis=0)

        out_df = self._sc_df.copy()
        out_df = out_df[out_df["Date"].between(begin_date, end_date)]
        return out_df


class CO2UsageLoad(DBColumnsBase):
    """
    Load co2 consumption data into 'co2_usage' table
    """

    OUT_TABLE_NAME = "co2_usage"
    DEPENDENCY_CLASSES = None

    def __init__(self, *args, **kwargs):
        # init as base class
        super().__init__(*args, **kwargs)

    def run(self):
        # load data
        co2_df = self._load_data_file()

        # calculate cost per day
        co2_df = self._calc_co2_cost(co2_df)

        # update the db
        update_table_rows_from_df(
            db_engine=self.db_engine,
            db_table_name=self.OUT_TABLE_NAME,
            update_data_df=co2_df,
            pk_cols=["Date"],
        )

    def _load_data_file(self) -> pd.DataFrame:
        print("Loading co2 consumption file...")
        file_id = load_setting("file_ids").get("co2_consumption")
        co2_df = M365ExcelFileHandler(
            file_object_id=file_id,
            load_data=True,
            data_get_method="DL",
            ignore_sheets=["2017", "Troubleshooting"],
            concat_sheets=True,
        ).concat_df

        # filter to date range provided to init Class
        co2_df = co2_df[co2_df["Date"].between(self.run_date_start, self.run_date_end)]

        # filter columns to just Date and Daily Consumption (lbs)
        co2_df = co2_df[["Date", "Daily Consumption (lbs)"]].dropna()
        co2_df = co2_df.rename(columns={"Daily Consumption (lbs)": "total_co2_usage"})

        return co2_df

    def _calc_co2_cost(self, co2_consumption_df) -> pd.DataFrame:
        # co2 cost / lb ... need a lookup table at some point to handle variation
        co2_cost_per_lb = 0.15
        co2_consumption_df["total_co2_cost"] = (
            co2_consumption_df["total_co2_usage"] * co2_cost_per_lb
        )
        return co2_consumption_df


class EPALoad(DBColumnsBase):
    OUT_TABLE_NAME = "epa_data"
    # keep track of columns to output? not using anywhere else yet...
    OUT_COLUMNS = [
        "epa_val",
        "epa_val_total_fa",
        "measurement_date_actual",
        "epa_actual_measurement",
    ]
    # define class dependency that needs to be run prior to an instance of this class
    DEPENDENCY_CLASSES = ["GetActiveStatus"]
    # override default daily update lookback days (from 5 to 15)
    MIN_LOOKBACK_DAYS = 15

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self):
        ## TODO: use data available from M365 api to check if file has been updated, then just store local file and re-download when necessary
        if not hasattr(self, "_epa_df"):
            EPALoad._epa_df = self._download_and_merge_epa_data()
        epa_df = self._epa_df.copy()

        # filter epa_df based on lookback_days parameter value (lookback_days is days in addition to the run_date, so num of rows returned will be lookback_days+1
        epa_df = epa_df[epa_df["Date"].between(self.run_date_start, self.run_date_end)]

        # update the db (calling a function from .db_utils.py)
        update_table_rows_from_df(
            db_engine=self.db_engine,
            db_table_name=self.OUT_TABLE_NAME,
            update_data_df=epa_df,
        )

    def _download_and_merge_epa_data(self) -> pd.DataFrame:
        print("Loading EPA data...")
        # download excel files and get downloaded file paths (pathlib.Path) as a list
        epa_data_file_paths = self._dl_epa_data()

        # process excel files (extract Date, PondID, 'epa_val' and 'epa_fa_val' as columns from file)
        tmp_list_of_dataframes = []
        # process_excel_file() returns a pandas DataFrame
        [
            tmp_list_of_dataframes.append(self._process_excel_file(excel_filename))
            for excel_filename in epa_data_file_paths
        ]

        # concat dataframes (using functools.reduce so number of files can vary)
        merged_epa_df = functools.reduce(
            lambda df1, df2: pd.concat([df1, df2], ignore_index=True),
            tmp_list_of_dataframes,
        )

        # group by Date and PondID to get averaged EPA values (when more than one value exists for a pond/date), then reset_index() to reset grouping back to a normal df
        merged_epa_df = (
            merged_epa_df.groupby(by=["Date", "PondID"])
            .mean()
            .reset_index()
            .sort_values(by=["Date", "PondID"])
        )

        # add a column to keep track of "actual" epa_values (days when measured) versus days in-betweeen measurements that are filled in
        merged_epa_df["epa_actual_measurement"] = True
        # setup column to forward fill date, so that row with filled data will have a reference to the date of the actual measurement
        merged_epa_df["measurement_date_actual"] = merged_epa_df["Date"]

        # construct a DF with combination of all possible Date and PondID columns for date range according to earliest and latest dates for EPA data
        epa_date_range = pd.date_range(
            start=merged_epa_df["Date"].min(), end=self.run_date_end
        )
        lp_date, lp_pondid = pd.core.reshape.util.cartesian_product(
            [epa_date_range, self.ponds_list]
        )
        dates_df = pd.DataFrame(
            list(zip(lp_date, lp_pondid)), columns=["Date", "PondID"]
        )

        # merge the dates_df with epa data df; using a left join so if any epa data is invalid (either date or PondID) then it will not be included in the merge,
        # and days with missing data will just be empty (except for Date and PondID fields)
        out_df = pd.merge(dates_df, merged_epa_df, on=["Date", "PondID"], how="left")

        # get list of all columns that should be forward filled by checking exclusion list
        fill_columns = [
            col
            for col in out_df.columns
            if col not in ("Date", "PondID", "epa_actual_measurement")
        ]

        # query 'active_status' to get dates when ponds were completely emptied / harvested completely, so to use as hard "reset points" in epa_data when forward filling in for dates in between readings
        active_pond_dates_df = query_data_table_by_date_range(
            db_name_or_engine=self.db_engine,
            table_name="ponds_data_calculated",
            # use the first date from the epa data / this will likely error out, but query should get first available data with "check_safe_date=True"
            query_date_start=out_df["Date"].min(),
            query_date_end=self.run_date_end,
            # Date and PondID columns automatically included in output
            col_names=["active_status"],
            check_safe_date=True,
        )

        # merge "active status" df with output df, and set all values, for "inactive" ponds, on columns that were forward filled,
        # to "n.a." temporarily, so that these rows don't get forward filled
        # do this instead of just filtering on 'active_status' when forward filling, because these rows act as a block for forward filling
        # i.e., one day a pond is active, then it is emptied with two days of inactivity, then restarted, the prior measurement wont carry forward after restarted
        out_df = pd.merge(
            out_df, active_pond_dates_df, on=["Date", "PondID"], how="left"
        )
        out_df.loc[out_df["active_status"] == False, fill_columns] = "n.a."

        # forward fill epa data to empty rows (dates between measurements) for active ponds
        for pond_id in out_df["PondID"].unique():
            # get temp copy of df filtered by PondID
            tmp_pond_id_df = out_df[out_df["PondID"] == pond_id].copy()
            tmp_pond_id_df[fill_columns] = tmp_pond_id_df[fill_columns].ffill()
            out_df.loc[out_df["PondID"] == pond_id, fill_columns] = tmp_pond_id_df[
                fill_columns
            ]

        # replace the temporary 'inactive' blocks used to stop forward filling for periods of inactivity with Null values
        out_df = out_df.replace("n.a.", None)

        # drop the 'active_status' column as it should not be included in output
        out_df = out_df.drop("active_status", axis=1)

        # set 'measurement_date_actual' as datetime to ensure that pd.dtypes recognizes it as datetime
        # this is necessary because the database update function will convert the column to string format to store it
        # but it will not work if the col dtype isn't datetime (col name including substring "date" is also required)
        out_df["measurement_date_actual"] = pd.to_datetime(
            out_df["measurement_date_actual"]
        )

        return out_df

    def _dl_epa_data(self) -> list[Path]:
        # get file IDs from settings file, get only the values from the dict that's returned
        epa_data_file_ids = {
            k: v for (k, v) in load_setting("file_ids").items() if "epa_data" in k
        }

        epa_file_paths = []
        for label, file_id in epa_data_file_ids.items():
            # returns file path, else returns None if failed
            tmp_file_path = self.account.download_sharepoint_file_by_id(
                object_id=file_id,
                name=f"{label}.xlsx",
                to_path=Path("data_sources/tmp/"),
            )
            if tmp_file_path is not None:
                epa_file_paths.append(tmp_file_path)
            else:
                raise Exception(
                    f"ERROR: failure downloading EPA data for {label}: {file_id}!"
                )
        return epa_file_paths

    def _process_excel_file(self, excel_filename) -> pd.DataFrame:
        # load epa_data spreadsheet
        epa_df = pd.read_excel(excel_filename, sheet_name="Sheet1", header=None)

        # set up an empty dict to store kwargs for options of processing rows in each data file
        row_process_kwargs = {}

        # check which file is being loaded (2 different sources currently)
        # checking if this is the primary data source
        if "epa_data_primary" in excel_filename.__repr__():
            print("Processing primary epa file")
            # parse the first 4 columns to update header (since excel file col headers are on different rows/merged rows)
            # backfill empty header rows in the first 4 header rows, results in header info copied to first row for all cols
            epa_df.iloc[0:4] = epa_df.iloc[0:4].fillna(method="bfill", axis=0)
            # remove extra spaces from the strings in the first column
            epa_df.iloc[0] = epa_df.iloc[0].apply(lambda x: " ".join(x.split()))
            # set header row
            epa_df.columns = epa_df.iloc[0]
            # delete the now unnecessary/duplicate rows of header data
            epa_df = epa_df.iloc[4:]
            epa_df = epa_df[["Sample type", "EPA % AFDW", "EPA % of Total FA"]]

        # check if it's the secondary data source
        elif "epa_data_secondary" in excel_filename.__repr__():
            # not currently using a secondary source
            pass
        else:
            print(
                'Filename wrong (expected "epa_data_primary.xlsx" or "epa_data_secondary.xlsx"...Wrong file provided?'
            )
            # return empty dataframe, as next concatenation steps expect dfs as input
            return pd.DataFrame()

        # use .apply() to generate new df for output, applying self._process_epa_row() to each row, with row_process_kwargs containing optional args
        out_epa_df = pd.DataFrame()
        out_epa_df[["Date", "PondID", "epa_val", "epa_val_total_fa"]] = epa_df.apply(
            lambda x: self._process_epa_row(x[0], x[1], x[2], **row_process_kwargs),
            axis=1,
            result_type="expand",
        )

        # drop n/a rows (rows containing N/A values will result from using epa_df.apply(), and self._process_epa_row() returning None for rows with invalid data)
        out_epa_df = out_epa_df.dropna()

        return out_epa_df

    def _process_epa_row(
        self,
        sample_label,
        epa_val,
        epa_val_total_fa,
        debug_print=False,
        convert_val_from_decimal_to_percentage=False,
    ) -> tuple:
        return_vals = {
            "date": None,
            "pond_id": None,
            "epa_val": None,
            "epa_val_total_fa": None,
        }
        return_on_error = [None] * len(return_vals)
        if debug_print:
            print("Printing output of row processing for debugging!")
            print(sample_label, end=" | ")

        if not isinstance(sample_label, str):
            if debug_print:
                print("ERROR: sample label not a string")
            return return_on_error

        # search for pond name in sample_label with regex (looking for 4 digits surrounded by nothing else or whitespace)
        # regex ref: https://stackoverflow.com/questions/45189706/regular-expression-matching-words-between-white-space
        pondname_search = re.search(r"(?<!\S)\d{4}(?!\S)", sample_label)
        if pondname_search:
            return_vals["pond_id"] = pondname_search.group()
        else:
            # check for pond name with alternate data source (epa_data2.xlsx), where some of the pond names are represented with only 3 digits, missing a leading 0 (e.g., 301 - means '0301')
            pondname_search = re.search(r"(?<!\S)\d{3}(?!\S)", sample_label)
            if pondname_search:
                tmp_pond_id = pondname_search.group()
                return_vals["pond_id"] = "0" + tmp_pond_id
            else:
                if debug_print:
                    print("ERROR: no pond name found in sample label")
                return return_on_error

        # check if pond id number exists as a key in master ponds list, ignore this data line if not
        if return_vals["pond_id"] not in self.ponds_list:
            print("EPA KEY ERROR:", return_vals["pond_id"])
            return return_on_error

        # search for date in sample_label with regex (looking for 6 digits surrounded by nothing else or whitespace)
        date_search = re.search(r"(?<!\S)\d{6}(?!\S)", sample_label)
        if date_search:
            return_vals["date"] = datetime.strptime(date_search.group(), "%y%m%d")
        else:
            if debug_print:
                print("ERROR: no date found in sample label")
            return return_on_error

        try:
            # return None if epa_val is zero (but not for epa_val_total_fa as it is secondary)
            if epa_val == 0:
                return return_on_error
            return_vals["epa_val"] = float(epa_val)
            return_vals["epa_val_total_fa"] = float(epa_val_total_fa)
        except ValueError:
            if debug_print:
                print('ERROR: "epa_val" or "epa_val_total_fa" is not a valid number')
            return return_on_error

        # convert epa val from decimal to percentage (i.e., 0.01 -> 1.00)
        if convert_val_from_decimal_to_percentage:
            return_vals["epa_val"] *= 100
            return_vals["epa_val_total_fa"] *= 100

        if debug_print:
            print("SUCCESS", return_vals)

        if None in return_vals.values():
            raise Exception(
                "ERROR: epa data all good but 'None' still in return values, missing a value assignment in dict?"
            )
        return return_vals.values()


class GetActiveStatus(DBColumnsBase):
    """
    Get the "active status" of each pond. Normally, if it has a 'Filter AFDW' and 'Fo' measurement, it would be considered 'Active'
    However, to handle occasional lags in data reporting, this status query determines if a pond should be considered active by checking previous dates
    If any rows within the 'max_check_days' int var have 'Filter AFDW' and 'Fo' measurements, then they will still be considered 'Active'.
    If any row, in the 'Split Innoculum' column, is preceded by 'HC' for the previous date (or 'I' for the current date and 'H'
    for the previous date [old way of reporting]), then that row is noted as 'Inactive'
    """

    OUT_TABLE_NAME = "ponds_data_calculated"
    # define class dependency that needs to be run prior to an instance of this class
    DEPENDENCY_CLASSES = ["DailyDataLoad"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self) -> None:
        # gets df with 'active_status' col
        active_df = self._get_ponds_active_status(
            begin_date=self.run_date_start, end_date=self.run_date_end
        )

        # update the db (calling a function from .db_utils.py)
        update_table_rows_from_df(
            db_engine=self.db_engine,
            db_table_name=self.OUT_TABLE_NAME,
            update_data_df=active_df,
        )

    def _get_ponds_active_status(
        self, begin_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        # maximum number of days to look back from a given date to determine if it has activity
        max_check_days = 5

        def _check_row(row_data):
            row_pond_id = row_data["PondID"]
            row_date = row_data["Date"]

            # filter the df to only rows being checked (current row, and the 5 days preceding it) for specific row_pond_id
            check_df = ref_df[
                (ref_df["PondID"] == row_pond_id)
                & (ref_df["Date"] >= row_date - pd.Timedelta(days=max_check_days))
                & (ref_df["Date"] <= row_date)
            ]

            # reverse sort the df by date (so newest to oldest)
            # do this instead of just .iloc[::-1] in case of the reference data being out of order
            check_df = check_df.sort_values(by="Date", ascending=False)
            # reset index so that values relative each row can be checked (next row values)
            check_df = check_df.reset_index(drop=True)

            # iterate through check_df rows in reverse order
            for row_idx, row_data in check_df.iterrows():
                # first check if there's any value in the 'Time Sampled' column
                # next, check if there is a value in either of the 'Filter AFDW' or 'Fo' columns (and that they are not zero)
                # if so, then consider the pond as active
                if not pd.isna(row_data["Time Sampled"]):
                    if (
                        not pd.isna(row_data["Filter AFDW"])
                        and row_data["Filter AFDW"] != 0
                    ) or (not pd.isna(row_data["Fo"]) and row_data["Fo"] != 0):
                        return True
                # else, if this is not the last item being iterated, check the next index value of 'Split Innoculum' column
                # for the prior day (but checking next index because of reversed order): "HC" value indicates complete harvest,
                # meaning that the pond is inactive if iteration reached this point
                elif (
                    row_idx + 1 != len(check_df)
                    and check_df.iloc[row_idx + 1]["Split Innoculum"] == "HC"
                ):
                    return False
                # also check for "H" on prior row and "I" in current row (discontinued way of noting full harvests)
                elif (
                    row_data["Split Innoculum"] == "I"
                    and row_idx + 1 != len(check_df)
                    and check_df.iloc[row_idx + 1]["Split Innoculum"] == "H"
                ):
                    return False
            # if all rows in the 'check_df' are iterated through and didn't find any 'Fo' values
            # then assume inactive status, and return False
            return False

        ref_df = query_data_table_by_date_range(
            db_name_or_engine=self.db_engine,
            table_name="ponds_data",
            query_date_start=begin_date - pd.Timedelta(days=max_check_days),
            query_date_end=end_date,
            col_names=["Time Sampled", "Fo", "Split Innoculum", "Filter AFDW"],
        )
        out_df = ref_df.loc[ref_df["Date"] >= begin_date, :].copy()

        for pond_id in out_df["PondID"].unique():
            pond_id_mask = out_df["PondID"] == pond_id
            out_df.loc[pond_id_mask, "active_status"] = out_df.loc[
                pond_id_mask, :
            ].apply(lambda x: _check_row(x), axis=1)

        return out_df[["Date", "PondID", "active_status"]]


class MassVolumeHarvestableCalculations(DBColumnsBase):
    OUT_TABLE_NAME = "ponds_data_calculated"
    # define class dependency that needs to be run prior to an instance of this class
    DEPENDENCY_CLASSES = ["DailyDataLoad"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self):
        # Load data necessary for calculations from the ref data table
        # (query_data_table_by_date_range function returns a DataFrame)
        self.ref_data_df = query_data_table_by_date_range(
            db_name_or_engine=self.db_engine,
            table_name="ponds_data",
            query_date_start=self.run_date_start,
            query_date_end=self.run_date_end,
            # Date and PondID automatically included in column output
            col_names=["Filter AFDW", "Depth", "% Nanno", "Split Innoculum"],
        )

        # Run calculations and collect in a df
        calc_df = self._base_calculations(self.ref_data_df)
        update_table_rows_from_df(
            db_engine=self.db_engine,
            db_table_name=self.OUT_TABLE_NAME,
            update_data_df=calc_df,
        )

    def _base_calculations(self, ref_df: pd.DataFrame) -> pd.DataFrame:
        """
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
             - 'harvestable_mass_nanno_corrected': harvestable_mass multiplied by "% Nanno" measurement
        """

        # helper function to convert density & depth to mass (in kilograms)
        def afdw_depth_to_mass(afdw: int | float, depth: int | float, pond_id: str):
            """
            params:
            - afdw: density in g/L
            - depth: depth in inches
            - pond_id: id number of pond (i.e., '0401')
            """
            depth_to_liters = (
                35000 * 3.78541
            )  # save conversion factor for depth (in inches) to liters
            # for the 6 and 8 columns, double the depth to liters conversion because they are twice the size
            if pond_id[-2:] == "06" or pond_id[-2:] == "08":
                depth_to_liters *= 2
            # calculate and return total mass (kg)
            return round((afdw * depth_to_liters * depth) / 1000, 2)

        # variables for calculating harvestable mass
        # based on min target density to begin harvesting, target density after harvesting and topping off with water, and target depth after topping off with water
        TARGET_TOPOFF_DEPTH = 13
        TARGET_DENSITY_AFTER_TOPOFF = 0.4
        MIN_HARVEST_DENSITY = 0.5

        calcs_df = pd.DataFrame()
        for date, pond_id, afdw, depth, pct_nanno in self.ref_data_df.loc[
            :, ["Date", "PondID", "Filter AFDW", "Depth", "% Nanno"]
        ].values.tolist():
            # initialize calculations dict with all values empty
            calcs_dict = {
                "calc_mass": None,
                "calc_mass_nanno_corrected": None,
                "harvestable_depth_inches": None,
                "harvestable_gallons": None,
                "harvestable_mass": None,
                "harvestable_mass_nanno_corrected": None,
            }

            if any(pd.isna(x) or x in (0, None) for x in (afdw, depth)):
                # skip calculations because data is missing
                pass
            else:
                calcs_dict["calc_mass"] = afdw_depth_to_mass(afdw, depth, pond_id)

                # calculate harvestable depth of pond (in inches), rounding down to nearest 1/8 inch
                if afdw >= MIN_HARVEST_DENSITY:
                    harvestable_depth_inches = (
                        math.floor(
                            (
                                (
                                    (depth * afdw)
                                    - (
                                        TARGET_TOPOFF_DEPTH
                                        * TARGET_DENSITY_AFTER_TOPOFF
                                    )
                                )
                                / afdw
                            )
                            * 8
                        )
                        / 8
                    )
                    if harvestable_depth_inches < 0:
                        harvestable_depth_inches = 0
                else:
                    harvestable_depth_inches = 0
                calcs_dict["harvestable_depth_inches"] = harvestable_depth_inches

                # calculate harvestable volume (in gallons) based on harvestable depth and conversion factor (35,000) to gallons.
                calcs_dict["harvestable_gallons"] = (
                    calcs_dict["harvestable_depth_inches"] * 35000
                )
                # Double for the '06' and '08' column (last two digits of PondID) ponds since they are double size
                if pond_id[-2:] == "06" or pond_id[-2:] == "08":
                    calcs_dict["harvestable_gallons"] *= 2

                # calculate harvestable mass using the harvestable_depth_inches calculation
                calcs_dict["harvestable_mass"] = afdw_depth_to_mass(
                    afdw, calcs_dict["harvestable_depth_inches"], pond_id
                )

                if type(pct_nanno) in (float, int, complex):
                    calcs_dict["calc_mass_nanno_corrected"] = round(
                        calcs_dict["calc_mass"] * (pct_nanno / 100), 2
                    )
                    calcs_dict["harvestable_mass_nanno_corrected"] = round(
                        calcs_dict["harvestable_mass"] * (pct_nanno / 100), 2
                    )
                else:
                    calcs_dict["calc_mass_nanno_corrected"] = 0
                    calcs_dict["harvestable_mass_nanno_corrected"] = 0

            # use dict comprehension to filter: to set <=0 values equal to None in numeric fields
            append_dict = {
                col_name: (
                    value
                    if (type(value) not in (int, float, complex) or value > 0)
                    else None
                )
                for (col_name, value) in {
                    **{"Date": date, "PondID": pond_id},
                    **calcs_dict,
                }.items()
            }
            calcs_df = calcs_df.append(append_dict, ignore_index=True)

        return calcs_df


class CalcMassGrowthPerPond(DBColumnsBase):
    OUT_TABLE_NAME = "ponds_data_calculated"
    DEPENDENCY_CLASSES = [
        "DailyDataLoad",
        "MassVolumeHarvestableCalculations",
        "GetActiveStatus",
        "CalcHarvestedSplitMass",
    ]  # define class dependency that needs to be run prior to an instance of this class

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self):
        growth_df = self._calc_growth(self.run_date_start, self.run_date_end)

        # update the db (calling a function from .db_utils.py)
        update_table_rows_from_df(
            db_engine=self.db_engine,
            db_table_name=self.OUT_TABLE_NAME,
            update_data_df=growth_df,
        )

    def _calc_growth(self, start_date, end_date) -> pd.DataFrame:
        """
        minimum date range is 26 days of data to calc growth:
            - extra 5 days padding for forward filling missing data (weekends, etc)
            - week-to-week growth: 7 days lookback
            - daily rolling average: 14 days of week-to-week growth data
        """
        # check if date range provided is at least 26 days, else override start_date param
        _param_start_date = start_date
        if (end_date - start_date).days < 26:
            start_date = end_date - pd.Timedelta(days=26)

        # query data to calculate 'liters' from 'Depth' (which val is in inches),
        # and calculate 'normalized liters' by converting liters with conversion factor (ratio of 'Filter AFDW' value compared to 0.50)
        ref_df1 = query_data_table_by_date_range(
            db_name_or_engine=self.db_engine,
            table_name="ponds_data",
            query_date_start=start_date,
            query_date_end=end_date,
            col_names=["Filter AFDW", "Depth"],
            check_safe_date=True,
        )

        # if either 'Filter AFDW' or 'Depth' value is missing for any date, then replace both with None
        # because without both the row is not a good reference for measuring growth
        def _none_if_afdw_or_depth_empty(row):
            if (pd.isna(row["Filter AFDW"]) or row["Filter AFDW"] == 0) or (
                pd.isna(row["Depth"]) or row["Depth"] == 0
            ):
                return [None, None]
            else:
                return [row["Filter AFDW"], row["Depth"]]

        ref_df1[["Filter AFDW", "Depth"]] = ref_df1.apply(
            lambda row: _none_if_afdw_or_depth_empty(row), axis=1, result_type="expand"
        )

        # calc liters and 'normalized' liters (multiplied by a factor equal to afdw/0.50)
        inches_to_liters = (
            35000 * 3.78541
        )  # save conversion factor for depth (in inches) to liters
        ref_df1["liters"] = ref_df1["Depth"].apply(
            lambda depth_val: inches_to_liters * depth_val
        )
        ref_df1["afdw_norm_factor"] = ref_df1["Filter AFDW"] / 0.50
        ref_df1["normalized_liters"] = ref_df1["liters"] * ref_df1["afdw_norm_factor"]

        # query calculated fields further used in this calculation
        ref_df2 = query_data_table_by_date_range(
            db_name_or_engine=self.db_engine,
            table_name="ponds_data_calculated",
            query_date_start=start_date,
            query_date_end=end_date,
            col_names=[
                "active_status",
                "est_harvested_mass",
                "est_split_in_mass",
                "est_split_out_mass",
                "calc_mass_nanno_corrected",
            ],
            check_safe_date=True,
        )

        # join the queried dataframes
        calc_df = pd.merge(ref_df1, ref_df2, on=["Date", "PondID"], how="outer")

        # set index as Date so that df.rolling() and df.shift() functions will work with days (to handle gaps in ponds being active)
        calc_df = calc_df.set_index("Date")
        for pond_id in calc_df["PondID"].unique():
            # set mask only on PondID field
            mask = calc_df["PondID"] == pond_id

            # compute a column as a check to whether growth can be computed for any date
            # this step is necessary in cases where a pond is emptied and "restarted" -> periods of 'inactive' status between periods of 'active' status
            # so this column determines if there are 7 continuous prior days of a pond being in an 'active' status
            def _check_growth_valid(row):
                row_date_idx = row.name
                date_check_range = (
                    pd.date_range(
                        start=row_date_idx - pd.Timedelta(days=7), end=row_date_idx
                    )
                    .to_pydatetime()
                    .tolist()
                )
                # if check range is outside the limits of the data, then return False
                if not all([d in calc_df.loc[mask].index for d in date_check_range]):
                    # print('some check dates missing in df, returning false')
                    return False
                chech_series = calc_df.loc[
                    (calc_df["PondID"] == pond_id)
                    & (calc_df.index.isin(date_check_range))
                ]["active_status"]
                if False in chech_series.values:
                    return False
                else:
                    return True

            calc_df.loc[mask, "_growth_valid"] = calc_df.loc[mask, :].apply(
                lambda row: _check_growth_valid(row), axis=1
            )

            # fill liters and mass data for missing days (weekends, etc)
            # .bfill() and ffill() doesn't have an option for date-aware filling, so ensure that any dates aren't filtered out with forward filling so ensure that 5 day limit is applied
            # first, for any missing vals where a harvest or split has been calculated: replace those vals with a temporary placeholder string
            # # only check mass for checking the columns to fill, since a missing mass would also indicate missing liters/normalized liters
            tmp_placeholder_mask = (
                (calc_df["PondID"] == pond_id)
                & (
                    (calc_df["est_harvested_mass"] > 0)
                    | (calc_df["est_split_out_mass"] > 0)
                )
                & (
                    (pd.isna(calc_df["calc_mass_nanno_corrected"]))
                    | (calc_df["calc_mass_nanno_corrected"] == 0)
                )
            )
            calc_df.loc[
                tmp_placeholder_mask,
                ["liters", "normalized_liters", "calc_mass_nanno_corrected"],
            ] = "_tmp_placeholder_for_ffill"

            # second, backfill values (filling stops when it hits a row with the placeholder string)
            calc_df.loc[
                mask, ["liters", "normalized_liters", "calc_mass_nanno_corrected"]
            ] = calc_df.loc[
                mask, ["liters", "normalized_liters", "calc_mass_nanno_corrected"]
            ].bfill(
                limit=5
            )

            # third, replace placeholder strings, then forward fill values
            calc_df.loc[
                mask, ["liters", "normalized_liters", "calc_mass_nanno_corrected"]
            ] = calc_df.loc[
                mask, ["liters", "normalized_liters", "calc_mass_nanno_corrected"]
            ].replace(
                "_tmp_placeholder_for_ffill", None
            )
            calc_df.loc[
                mask, ["liters", "normalized_liters", "calc_mass_nanno_corrected"]
            ] = calc_df.loc[
                mask, ["liters", "normalized_liters", "calc_mass_nanno_corrected"]
            ].ffill(
                limit=5
            )

            # update mask to include 'active' ponds only
            # this is necessary because further steps perform row lookbacks, and need to ensure that data filled onto an inactive day isn't factored in
            mask = (calc_df["PondID"] == pond_id) & (calc_df["active_status"] == True)

            # get "harvest corrected (hc)" values for liters, normalized_liters, and calc_mass_nanno_corrected (next day val, if day has been noted as a harvest with a calculated est_harvested val > 0)
            # per Kurt, this is so that variance in pond levels and density is factored out of growth calcs
            def _get_harvest_corrected_val(row, col_name):
                if (
                    not pd.isna(row["est_harvested_mass"])
                    and row["est_harvested_mass"] > 0
                ) or (
                    not pd.isna(row["est_split_out_mass"])
                    and row["est_split_out_mass"] > 0
                ):
                    rval_df = calc_df.loc[mask, col_name].shift(-1, freq="D")
                    if row.name in rval_df:
                        return rval_df.loc[row.name]
                    else:
                        return None
                else:
                    return row[col_name]

            calc_df.loc[mask, "hc_liters"] = calc_df.loc[mask, :].apply(
                lambda row: _get_harvest_corrected_val(row, "liters"), axis=1
            )
            calc_df.loc[mask, "hc_normalized_liters"] = calc_df.loc[mask, :].apply(
                lambda row: _get_harvest_corrected_val(row, "normalized_liters"), axis=1
            )
            calc_df.loc[mask, "hc_calc_mass_nanno_corrected"] = calc_df.loc[
                mask, :
            ].apply(
                lambda row: _get_harvest_corrected_val(
                    row, "calc_mass_nanno_corrected"
                ),
                axis=1,
            )

            # fill in n/a values in 'est_harvested_mass', 'est_split_in_mass', and 'est_split_out_mass' so that .rolling().sum() to calculate rolling sums (otherwise NaN values will cause it to not work)
            calc_df.loc[mask, "est_harvested_mass"] = calc_df.loc[
                mask, "est_harvested_mass"
            ].fillna(0)
            # TODO: update with 'split_in_mass' once its in the db
            calc_df.loc[mask, "est_split_in_mass"] = 0
            calc_df.loc[mask, "est_split_out_mass"] = calc_df.loc[
                mask, "est_split_out_mass"
            ].fillna(0)

            # calculate rolling harvested amount for past 7-days
            # since the 'calc_mass_nanno_corrected' value on any day should be inclusive of the prior 6-days of activity (harvests/splits),
            # and the calc_mass_nanno_corrected val does not reflect harvests/splits on that day (i.e., measurements are "generally" taken before these activities on any given day)
            # then get a 6-day rolling sum of harvests and net splits which does not include the current day (by shifting by 1 day)
            # without doing this, then harvests/splits would be double counted in the overall mass comparison
            calc_df.loc[mask, "rolling_7d_harvested_mass_out"] = (
                calc_df.loc[mask, "est_harvested_mass"].shift(1).rolling("6d").sum()
            )
            calc_df.loc[mask, "rolling_7d_split_net_mass_out"] = (
                calc_df.loc[mask, "est_split_out_mass"].shift(1).rolling("6d").sum()
                - calc_df.loc[mask, "est_split_in_mass"].shift(1).rolling("6d").sum()
            )

            # get the 'liters', 'normalized liters' from 7-days ago, using "harvest corrected" columns
            calc_df.loc[mask, "growth_ref_prev_liters_7d"] = calc_df.loc[
                mask, "hc_liters"
            ].shift(7, freq="D")
            calc_df.loc[mask, "growth_ref_prev_norm_liters_7d"] = calc_df.loc[
                mask, "hc_normalized_liters"
            ].shift(7, freq="D")

            # get 'calc_mass' from 7-days ago for mass change calculation, using 'harvest corrected' values
            calc_df.loc[mask, "mass_7d_prev"] = calc_df.loc[
                mask, "hc_calc_mass_nanno_corrected"
            ].shift(7, freq="D")

            # calculate the net 7-day mass change in kg
            calc_df.loc[mask, "growth_ref_mass_change_kg_7d"] = (
                calc_df.loc[mask, "rolling_7d_harvested_mass_out"]
                + calc_df.loc[mask, "rolling_7d_split_net_mass_out"]
                + calc_df.loc[mask, "calc_mass_nanno_corrected"]
                - calc_df.loc[mask, "mass_7d_prev"]
            )
            # filter out negative mass change (more than likely this is due to measurement errors)
            calc_df.loc[mask, "growth_ref_mass_change_kg_7d"] = calc_df.loc[
                mask, "growth_ref_mass_change_kg_7d"
            ].apply(lambda x: x if x > 0 else 0)
            # convert kilograms to grams for growth calc
            calc_df.loc[mask, "growth_ref_mass_change_grams_7d"] = (
                calc_df.loc[mask, "growth_ref_mass_change_kg_7d"] * 1000
            )

            # clear out the rows where growth cannot be valid (does not have enough continuous days as 'active' status)
            for column in calc_df.columns:
                if column not in ["Date", "PondID", "_growth_valid"]:
                    calc_df.loc[mask, column] = calc_df.loc[mask].apply(
                        lambda row: row[column]
                        if row["_growth_valid"] == True
                        else None,
                        axis=1,
                    )

            # calculate running average growth
            calc_df.loc[mask, "running_avg_growth_5d"] = (
                calc_df.loc[mask, "growth_ref_mass_change_grams_7d"]
                .rolling("5d", min_periods=5)
                .sum()
                / calc_df.loc[mask, "growth_ref_prev_liters_7d"]
                .rolling("5d", min_periods=5)
                .sum()
            )
            calc_df.loc[mask, "running_avg_norm_growth_5d"] = (
                calc_df.loc[mask, "growth_ref_mass_change_grams_7d"]
                .rolling("5d", min_periods=5)
                .sum()
                / calc_df.loc[mask, "growth_ref_prev_norm_liters_7d"]
                .rolling("5d", min_periods=5)
                .sum()
            )
            calc_df.loc[mask, "running_avg_growth_14d"] = (
                calc_df.loc[mask, "growth_ref_mass_change_grams_7d"]
                .rolling("14d", min_periods=14)
                .sum()
                / calc_df.loc[mask, "growth_ref_prev_liters_7d"]
                .rolling("14d", min_periods=14)
                .sum()
            )
            calc_df.loc[mask, "running_avg_norm_growth_14d"] = (
                calc_df.loc[mask, "growth_ref_mass_change_grams_7d"]
                .rolling("14d", min_periods=14)
                .sum()
                / calc_df.loc[mask, "growth_ref_prev_norm_liters_7d"]
                .rolling("14d", min_periods=14)
                .sum()
            )

        # reset index so that 'Date' is a column again
        calc_df = calc_df.reset_index()

        # filter output to include rows only greater than the parameter start_date (since it is overridden to a min of 14 days for calculations to work)
        calc_df = calc_df[calc_df["Date"] >= _param_start_date]

        calc_df.to_excel("test_growth_per_pond_rev3.xlsx", index=False)

        # filter to output columns
        calc_df = calc_df[
            [
                "Date",
                "PondID",
                "growth_ref_mass_change_grams_7d",
                "growth_ref_prev_liters_7d",
                "growth_ref_prev_norm_liters_7d",
                "running_avg_growth_5d",
                "running_avg_norm_growth_5d",
                "running_avg_growth_14d",
                "running_avg_norm_growth_14d",
            ]
        ]

        return calc_df


class CalcMassGrowthAggregate(DBColumnsBase):
    OUT_TABLE_NAME = "ponds_data_aggregate"
    # define class dependency that needs to be run prior to an instance of this class
    DEPENDENCY_CLASSES = ["CalcMassGrowthPerPond"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self):
        agg_growth_df = self._calc_growth_agg(self.run_date_start, self.run_date_end)

        # update the db (calling a function from .db_utils.py)
        update_table_rows_from_df(
            db_engine=self.db_engine,
            db_table_name=self.OUT_TABLE_NAME,
            update_data_df=agg_growth_df,
            pk_cols=["Date"],
        )

    def _calc_growth_agg(self, start_date, end_date):
        """
        minimum date range span to calculate aggregate growth is 19 days:
        - 5 days for rolling sums
        - 14 days of comparison (each a calc from a rolling sum)
        """
        # check if date range provided is at least 14, else override start_date param
        _param_start_date = start_date
        if (end_date - start_date).days < 19:
            start_date = end_date - pd.Timedelta(days=19)

        # query growth data (by pond) and subtotal by date
        agg_growth_df = query_data_table_by_date_range(
            db_name_or_engine=self.db_engine,
            table_name="ponds_data_calculated",
            query_date_start=start_date,
            query_date_end=end_date,
            col_names=[
                "growth_ref_mass_change_grams_7d",
                "growth_ref_prev_liters_7d",
                "growth_ref_prev_norm_liters_7d",
            ],
        )
        agg_growth_df = agg_growth_df.groupby(by="Date").sum()

        # calculate running average growth
        agg_growth_df["agg_running_avg_growth_5d"] = (
            agg_growth_df["growth_ref_mass_change_grams_7d"]
            .rolling("5d", min_periods=5)
            .sum()
            / agg_growth_df["growth_ref_prev_liters_7d"]
            .rolling("5d", min_periods=5)
            .sum()
        )
        agg_growth_df["agg_running_avg_norm_growth_5d"] = (
            agg_growth_df["growth_ref_mass_change_grams_7d"]
            .rolling("5d", min_periods=5)
            .sum()
            / agg_growth_df["growth_ref_prev_norm_liters_7d"]
            .rolling("5d", min_periods=5)
            .sum()
        )
        agg_growth_df["agg_running_avg_growth_14d"] = (
            agg_growth_df["growth_ref_mass_change_grams_7d"]
            .rolling("14d", min_periods=14)
            .sum()
            / agg_growth_df["growth_ref_prev_liters_7d"]
            .rolling("14d", min_periods=14)
            .sum()
        )
        agg_growth_df["agg_running_avg_norm_growth_14d"] = (
            agg_growth_df["growth_ref_mass_change_grams_7d"]
            .rolling("14d", min_periods=14)
            .sum()
            / agg_growth_df["growth_ref_prev_norm_liters_7d"]
            .rolling("14d", min_periods=14)
            .sum()
        )

        # reset index so that Date is a column
        agg_growth_df = agg_growth_df.reset_index()

        agg_growth_df.to_excel("test_agg_growth.xlsx", index=False)

        # filter output to include rows only greater than the parameter start_date (since it is overridden to a min of 7 days for calculations to work)
        agg_growth_df = agg_growth_df[agg_growth_df["Date"] >= _param_start_date]

        return agg_growth_df


class CalcHarvestedSplitMass(DBColumnsBase):
    OUT_TABLE_NAME = "ponds_data_calculated"
    DEPENDENCY_CLASSES = [
        "DailyDataLoad",
        "GetActiveStatus",
        "HarvestDataLoad",
        "MassVolumeHarvestableCalculations",
    ]  # TODO add Split data class once it's added to db

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self):
        # get df with dates for harvested/split for each pond
        h_s_dates_df = self._get_harvested_split_dates(
            start_date=self.run_date_start, end_date=self.run_date_end
        )

        # get harvested and split mass estimates using primary method
        # (get the calculated mass change between two days)
        hs_df = self._calc_harvested_split_mass_differential_actual(
            hs_dates_df=h_s_dates_df,
            start_date=self.run_date_start,
            end_date=self.run_date_end,
        )

        # get harvested and split mass estimates using the 'backup' method
        # (gets calculated mass change based on logs for how much volume was harvested)
        # (inherently less accurate than the primary method due to using only one measurement point)
        backup_estimates_df = self._calc_harvested_split_from_volume_logs(
            start_date=self.run_date_start, end_date=self.run_date_end
        )
        backup_estimates_df = backup_estimates_df.rename(
            columns={
                "est_harvested_mass": "est_harvested_mass_backup",
                "est_split_out_mass": "est_split_out_mass_backup",
            }
        )

        # merge and add backup estimated mass values where needed (i.e., wherever the backup method calculated an amount but nothing exists on the primary df)
        hs_df = pd.merge(hs_df, backup_estimates_df, on=["Date", "PondID"], how="outer")

        # if there is no value (for est_harvested and est_split_out) for the current day, previous day, and next day,
        # then use the backup value (will be None unless a backup value exists)
        hs_df["est_harvested_mass"] = hs_df.apply(
            lambda row: row["est_harvested_mass_backup"]
            if (pd.isna(row["est_harvested_mass"]) & (~pd.isna(row["Harvested-Split"])))
            else row["est_harvested_mass"],
            axis=1,
        )
        hs_df["est_split_out_mass"] = hs_df.apply(
            lambda row: row["est_split_out_mass_backup"]
            if (pd.isna(row["est_split_out_mass"]) & (~pd.isna(row["Harvested-Split"])))
            else row["est_split_out_mass"],
            axis=1,
        )

        # filter to output columns
        hs_df = hs_df.loc[
            hs_df["Date"].between(self.run_date_start, self.run_date_end),
            ["Date", "PondID", "est_harvested_mass", "est_split_out_mass"],
        ]

        # update the db (calling a function from .db_utils.py)
        update_table_rows_from_df(
            db_engine=self.db_engine,
            db_table_name=self.OUT_TABLE_NAME,
            update_data_df=hs_df,
        )

    def _get_harvested_split_dates(
        self, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """Method to query dates on which a harvest or split has occured
            - Use measured harvest & split volumes (obviously)
            - Add additional check on 'Scorecard' file data where 'H' or 'S' or 'HC' is noted
            (this is because sometimes one of the other mistakenly doesn't get recorded,
            and this ensures a more robust estimate of actual harvested amounts)

        output:
          - Dataframe:
            - 'Date': datetime
            - 'PondID': string
            - 'Harvested-Split': string:
                - 'H' : if harvested
                - 'HC': if harvested completely
                - 'S': if split
                - 'SC': if split completely
        """
        df = query_data_table_by_date_range(
            db_name_or_engine=self.db_engine,
            table_name="ponds_data",
            query_date_start=start_date,
            # add one day to db query, to try checking the next day \
            # 'Split Innoculum' value for the last date in output range
            query_date_end=end_date + pd.Timedelta(days=1),
            col_names=["Split Innoculum", "harvest_volume", "split_volume_out"],
        )
        for pond_id in self.ponds_list:
            mask = df["PondID"] == pond_id
            df.loc[mask, "Split Innoculum Next"] = df.loc[
                mask, "Split Innoculum"
            ].shift(-1)

        # conditions
        conditions = [
            (pd.isna(df["harvest_volume"]))
            & (pd.isna(df["split_volume_out"]))
            & (
                (pd.isna(df["Split Innoculum"]))
                | (df["Split Innoculum"].str.upper() == "I")
            ),
            # check conditions on 'Split Innoculum' column first since it should be noted whether harvested or harvested completely
            (df["Split Innoculum"].str.upper() == "HC")
            | (
                (df["Split Innoculum Next"].str.upper() == "I")
                & (df["Split Innoculum"].str.upper() == "H")
            ),  # harvested completely condition
            (df["Split Innoculum Next"].str.upper() == "I")
            & (
                (pd.isna(df["Split Innoculum"]))
            ),  # backup harvested completely condition, when nothing noted but inactive the next day (could also be a split, but attributing to harvest is safer)
            df["Split Innoculum"].str.upper() == "H",  # harvested condition
            (df["Split Innoculum"].str.upper() == "SC")
            | (
                (df["Split Innoculum Next"].str.upper() == "I")
                & (df["Split Innoculum"].str.upper() == "S")
            ),  # split completely condition
            df["Split Innoculum"].str.upper() == "S",  # split condition
            df["harvest_volume"] > 0,  # backup 'harvest' condition
            df["split_volume_out"] > 0,  # backup 'split' condition
        ]
        choices = [None, "HC", "HC", "H", "SC", "S", "H", "S"]
        df["Harvested-Split"] = np.select(conditions, choices, default="ERROR")

        # assert that np.select worked / if any columns get the default 'ERROR' condition, then something is wrong
        assert len(df[df["Harvested-Split"] == "ERROR"]) == 0

        df = df[["Date", "PondID", "Harvested-Split"]]
        return df

    def _calc_harvested_split_mass_differential_actual(
        self, hs_dates_df: pd.DataFrame, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """
        Method to get an estimate of harvested/split mass using the delta of calculated mass between 2 days

        params:
            - hs_dates_df: pd.DataFrame: dataframe containing 'Harvested-Split' column containing either nothing, 'H', 'HC', 'S', or 'SC'
                        (from self._get_harveted_split_dates currently)
            - start_date: datetime: start of date range to output
            - end_date: datetime: end of date range to output

        output:
          - Dataframe:
            - 'Date': datetime
            - 'PondID': string
            - 'est_harvested_mass': float : estimated harvested mass based on differential in estimated mass between two dates
            - 'est_split_out_mass': float: estimated "split out" mass ""
        """

        df_mass_data = query_data_table_by_date_range(
            db_name_or_engine=self.db_engine,
            table_name="ponds_data_calculated",
            query_date_start=start_date,
            # always get one day after the end_date, as it would be necessary \
            # to get the change in mass for the last date in output range
            query_date_end=end_date + pd.Timedelta(days=1),
            col_names=["calc_mass_nanno_corrected"],
            # use check_safe_date in case of the end_date being outside \
            # the date range available in the db table
            check_safe_date=True,
        )

        df = pd.merge(df_mass_data, hs_dates_df, on=["Date", "PondID"], how="outer")

        # set Date as index so that .shift() will work properly with df masked by PondID
        df = df.set_index("Date")
        for pond_id in df["PondID"].unique():
            # get df mask with only values for specific PondID
            mask = df["PondID"] == pond_id
            # shift to get the next day mass relative to each pond/date
            df.loc[mask, "_tmp_next_day_mass"] = df.loc[
                mask, "calc_mass_nanno_corrected"
            ].shift(-1)
        df = df.reset_index()

        # apply a function to calculate the 'est_harvested_mass', 'est_split_out_mass' by getting the difference \
        # between current day estimated/calculated mass and the next day mass, and additionally filter out "bad" \
        # values for the 'Harvested-Split' column (found by having a negative value from the difference)
        def apply_h_s_mass(row):
            if row["Harvested-Split"] == "H":
                return_val = [
                    row["calc_mass_nanno_corrected"] - row["_tmp_next_day_mass"],
                    None,
                ]
            elif row["Harvested-Split"] == "HC":
                return_val = [row["calc_mass_nanno_corrected"], None]
            elif row["Harvested-Split"] == "S":
                return_val = [
                    None,
                    row["calc_mass_nanno_corrected"] - row["_tmp_next_day_mass"],
                ]
            elif row["Harvested-Split"] == "SC":
                return_val = [None, row["calc_mass_nanno_corrected"]]
            else:
                return_val = [None, None]

            # if either of the values for est_harvested_mass or est_split_out_mass are < 0 \
            # then set them to None, and also set the 'Harvested-Split' column to None \
            # because it's obviously not correct that this date is a harvest or split \
            # due to the negative mass change
            if any([(v != None and v < 0) for v in return_val]):
                return_val = [None, None, None]
            else:
                # add col to beginning of return val as-is if no negative values exist
                return_val.insert(0, row["Harvested-Split"])

            return return_val

        df[["Harvested-Split", "est_harvested_mass", "est_split_out_mass"]] = df.apply(
            lambda row: apply_h_s_mass(row), axis=1, result_type="expand"
        )

        # filter to the output (function param) date range, and output columns
        df = df.loc[
            df["Date"].between(start_date, end_date),
            [
                "Date",
                "PondID",
                "Harvested-Split",
                "est_harvested_mass",
                "est_split_out_mass",
            ],
        ]
        return df

    def _calc_harvested_split_from_volume_logs(
        self, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        # query density and harvest volume (in gallons)
        # add 5 days to beginning of query for additional data if needed to fill in missing values
        df_harvested = query_data_table_by_date_range(
            db_name_or_engine=self.db_engine,
            table_name="ponds_data",
            query_date_start=start_date - pd.Timedelta(days=5),
            query_date_end=end_date,
            col_names=["Filter AFDW", "% Nanno", "harvest_volume", "split_volume_out"],
            check_safe_date=True,
        )

        df_active = query_data_table_by_date_range(
            db_name_or_engine=self.db_engine,
            table_name="ponds_data_calculated",
            query_date_start=start_date - pd.Timedelta(days=5),
            query_date_end=end_date,
            col_names=["active_status"],
            check_safe_date=True,
        )
        df_harvested = pd.merge(
            df_harvested, df_active, on=["Date", "PondID"], how="left"
        )

        # conversion factor for gal to L
        GALLONS_TO_LITERS = 3.78541

        # add a placeholder string for 'Inactive' dates so that they don't get filled in next step
        # can't just filter out inactive dates or data will be carried over between period of inactivity
        df_harvested.loc[
            df_harvested["active_status"] == False, "Filter AFDW"
        ] = "_tmp_placeholder_for_inactive"

        # fill afdw data for missing days (weekends, etc)
        for pond_id in self.ponds_list:
            mask = df_harvested["PondID"] == pond_id
            fill_cols = ["Filter AFDW", "% Nanno"]
            # .bfill() and ffill() doesn't have an option for date-aware filling, so ensure that any dates aren't filtered out with forward filling so ensure that 5 day limit is applied
            # first, for any missing vals where a harvest or split has been calculated, then replace those vals with a temporary placeholder string
            tmp_placeholder_mask = (
                (df_harvested["PondID"] == pond_id)
                & (
                    (df_harvested["harvest_volume"] > 0)
                    | (df_harvested["split_volume_out"] > 0)
                )
                & (
                    (pd.isna(df_harvested["Filter AFDW"]))
                    | (~pd.isna(df_harvested["Filter AFDW"]))
                )
            )
            df_harvested.loc[
                tmp_placeholder_mask, fill_cols
            ] = "_tmp_placeholder_for_ffill"

            # second, backfill values (so that the placeholder string blocks filling back to the date of harvest/split
            df_harvested.loc[mask, fill_cols] = df_harvested.loc[mask, fill_cols].bfill(
                limit=5
            )

            # third, replace placeholder strings, then forward fill any remaining missing values
            df_harvested.loc[mask, fill_cols] = df_harvested.loc[
                mask, fill_cols
            ].replace("_tmp_placeholder_for_ffill", None)
            df_harvested.loc[mask, fill_cols] = df_harvested.loc[mask, fill_cols].ffill(
                limit=5
            )

        # remove placeholder for inactive dates
        df_harvested.loc[df_harvested["active_status"] == False, fill_cols] = None

        df_harvested = df_harvested[df_harvested["active_status"] == True]

        # convert % Nanno column from str to float
        df_harvested["Filter AFDW"] = pd.to_numeric(
            df_harvested["Filter AFDW"], errors="coerce"
        )
        df_harvested["% Nanno"] = (
            pd.to_numeric(df_harvested["% Nanno"], errors="coerce") / 100
        )

        # calculate est_harvested in Kg for
        # AFDW units = g/L, so convert harvested gallons to liters, then multiply by the AFDW. Divide result by 1000 for Kg
        df_harvested["est_harvested_mass"] = (
            df_harvested["Filter AFDW"]
            * df_harvested["% Nanno"]
            * df_harvested["harvest_volume"]
            * GALLONS_TO_LITERS
        ) / 1000
        # TODO add est_split_out calc after split volumes added to DB
        df_harvested["est_split_out_mass"] = None

        # filter to the output date range & output columns
        df_harvested = df_harvested[df_harvested["Date"].between(start_date, end_date)]
        df_harvested = df_harvested[
            ["Date", "PondID", "est_harvested_mass", "est_split_out_mass"]
        ]

        return df_harvested


class CalcDaysSinceHarvestedSplit(DBColumnsBase):
    OUT_TABLE_NAME = "ponds_data_calculated"
    # TODO add class for Split data once its incorporated
    DEPENDENCY_CLASSES = ["GetActiveStatus", "CalcHarvestedSplitMass"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self):
        dhs_df = self._calc_days_since_harvest_split(
            start_date=self.run_date_start, end_date=self.run_date_end
        )

        # update the db (calling a function from .db_utils.py)
        update_table_rows_from_df(
            db_engine=self.db_engine,
            db_table_name=self.OUT_TABLE_NAME,
            update_data_df=dhs_df,
        )

    def _calc_days_since_harvest_split(
        self, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        # get ref data and add an additional 45 days of padding (excessive??) for looking back
        # use check_safe_date arg for query function so that no error is thrown if start_date is prior to DB records
        ref_df = query_data_table_by_date_range(
            db_name_or_engine=self.db_engine,
            table_name="ponds_data_calculated",
            query_date_start=start_date - pd.Timedelta(days=45),
            query_date_end=end_date,
            col_names=["active_status", "est_harvested_mass", "est_split_out_mass"],
            check_safe_date=True,
        )

        out_df = ref_df[["Date", "PondID"]]
        # set index to Date so for updating data from a temp df, so that Date-index serves as "key" column to join data on
        out_df = out_df.set_index("Date")
        ref_df = ref_df.set_index("Date")

        # get a column of 1's where active_status==True and NOT harvested on day
        ref_df["_active_and_not_harvested_split"] = ref_df.apply(
            lambda row: 1
            if row["active_status"] == True
            and (
                pd.isna(row["est_harvested_mass"])
                and pd.isna(row["est_split_out_mass"])
            )
            else None,
            axis=1,
        )

        # get cumulative sum for concurrent days of active_status=True and not harvested/split
        # ref for reset df cumsum() with None row vals: https://stackoverflow.com/questions/55147225/pandas-dataframe-cumsum-reset-on-nan
        for pond_id in ref_df["PondID"].unique():
            _tmp_df = ref_df[ref_df["PondID"] == pond_id].copy()
            _tmp_df["days_since_harvested_split"] = (
                _tmp_df["_active_and_not_harvested_split"]
                .groupby(_tmp_df["_active_and_not_harvested_split"].isna().cumsum())
                .cumsum()
                .fillna(0, downcast="infer")
            )
            # update 'days_since_harvest' in out_df for pond_id
            out_df.loc[
                out_df["PondID"] == pond_id,
                ["active_status", "days_since_harvested_split"],
            ] = _tmp_df[["active_status", "days_since_harvested_split"]]

        # set days where active_status == False to None (otherwise, they are set to 0 from the last step, and that implies that the pond is active)
        out_df.loc[
            out_df["active_status"] == False, "days_since_harvested_split"
        ] = None

        # reset index of out_df so that 'Date' is a column again
        out_df = out_df.reset_index()

        # filter dates to the param dates (remove the extra 45 days of data added to beginning of query)
        out_df = out_df[out_df["Date"].between(start_date, end_date)]

        return out_df[["Date", "PondID", "days_since_harvested_split"]]


class GetPondHealthStatusCode(DBColumnsBase):
    OUT_TABLE_NAME = "ponds_data_calculated"
    # define class dependency that needs to be run prior to an instance of this class
    DEPENDENCY_CLASSES = ["DailyDataLoad", "GetActiveStatus", "EPALoad"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self):
        status_df = self._get_pond_status_color_code(
            begin_date=self.run_date_start, end_date=self.run_date_end
        )

        # update the db (calling a function from .db_utils.py)
        update_table_rows_from_df(
            db_engine=self.db_engine,
            db_table_name=self.OUT_TABLE_NAME,
            update_data_df=status_df,
        )

    def _get_pond_status_color_code(
        self, begin_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """
        status codes:
            - 0: inactive pond
            - 1: grey: incomplete data (either 'afdw' or 'epa_val' are missing)
            - 2: red: afdw less than 0.25
            - 3: brown: afdw >= 0.25; epa_val < 2.5%
            - 4: yellow: (afdw >= 0.25 and afdw < 0.50) OR (epa_val >=2.5% and epa_val < 3%)
            - 5: light green: afdw >= 0.50 and < 0.80; epa_val > 3%
            - 6: dark green: afdw >= 0.80; epa_val > 3%
            - ERROR: should not happen, possible bug in code / missing conditions / etc
        """
        # override begin_date param, need 3 days of data minimum
        orig_begin_date = begin_date
        if (end_date - begin_date).days < 3:
            begin_date = end_date - pd.Timedelta(days=2)

        afdw_df = query_data_table_by_date_range(
            db_name_or_engine=self.db_engine,
            table_name="ponds_data",
            query_date_start=begin_date,
            query_date_end=end_date,
            col_names=["Filter AFDW"],
            check_safe_date=True,
        )
        active_df = query_data_table_by_date_range(
            db_name_or_engine=self.db_engine,
            table_name="ponds_data_calculated",
            query_date_start=begin_date,
            query_date_end=end_date,
            col_names=["active_status"],
            check_safe_date=True,
        )
        epa_df = query_data_table_by_date_range(
            db_name_or_engine=self.db_engine,
            table_name="epa_data",
            query_date_start=begin_date,
            query_date_end=end_date,
            col_names=["epa_val"],
            check_safe_date=True,
            raise_exception_on_error=False,
        )
        df = functools.reduce(
            lambda df1, df2: pd.merge(df1, df2, on=["Date", "PondID"], how="outer"),
            [df for df in [afdw_df, active_df, epa_df]],
        )

        # forward fill AFDW a max of 2 days to account for weekends when measurements aren't taken
        # add a placeholder on inactive days to prevent forward filling (i.e., if a pond was drained and re-started, don't carry forward data)
        df["Filter AFDW"] = df.apply(
            lambda row: "_tmp_placeholder_for_ffill"
            if row["active_status"] == False
            else row["Filter AFDW"],
            axis=1,
        )
        for pond_id in self.ponds_list:
            mask = df["PondID"] == pond_id
            # forward fill AFDW a maximum 2 days, to account for weekends when measurements aren't taken
            df.loc[mask, "Filter AFDW"] = df.loc[mask, "Filter AFDW"].ffill(limit=2)
        df["Filter AFDW"] = df["Filter AFDW"].replace(
            "_tmp_placeholder_for_ffill", None
        )

        # conditions
        conditions = [
            df["active_status"] == False,
            (pd.isna(df["Filter AFDW"])) | (pd.isna(df["epa_val"])),
            df["Filter AFDW"] < 0.25,
            (df["Filter AFDW"] >= 0.25) & (df["epa_val"] < 2.5),
            ((df["Filter AFDW"] >= 0.25) & (df["Filter AFDW"] < 0.50))
            | ((df["epa_val"] >= 2.5) & (df["epa_val"] < 3.0)),
            (df["Filter AFDW"] >= 0.50)
            & (df["Filter AFDW"] < 0.80)
            & (df["epa_val"] >= 3.0),
            (df["Filter AFDW"] >= 0.80) & (df["epa_val"] >= 3.0),
        ]
        choices = [0, 1, 2, 3, 4, 5, 6]

        df["status_code"] = np.select(conditions, choices, default="ERROR")

        # set filter to orig date range
        df = df[df["Date"].between(orig_begin_date, end_date)]

        return df[["Date", "PondID", "status_code"]]


class ChemicalUsageLoad(DBColumnsBase):
    OUT_TABLE_NAME = "ponds_data_expenses"
    # define class dependency that needs to be run prior to an instance of this class
    DEPENDENCY_CLASSES = ["DailyDataLoad"]

    # hardcode costs - TEMPORARY??
    chem_costs = {
        "uan-32": {
            "uom": "gal",
            "cost": 2.08,
            "data_column": "Volume UAN-32 Added",
            "out_column": "uan32_cost",
        },
        "fertilizer-10-34": {
            "uom": "gal",
            "cost": 4.25,
            "data_column": "Volume 10-34 Added",
            "out_column": "fert1034_cost",
        },
        "bleach": {
            "uom": "gal",
            "cost": 3.15,
            "data_column": "Volume Bleach Added",
            "out_column": "bleach_cost",
        },
        "trace": {
            "uom": "gal",
            "cost": 0.41,
            "data_column": "Volume Trace Added",
            "out_column": "trace_cost",
        },
        "iron": {
            "uom": "gal",
            "cost": 0.60,
            "data_column": "Volume Iron Added",
            "out_column": "iron_cost",
        },
        "cal-hypo": {
            "uom": "kg",
            "cost": 3.17,
            "data_column": "kg Cal Hypo Added",
            "out_column": "cal_hypo_cost",
        },
        "benzalkonium": {
            "uom": "gal",
            "cost": 49.07,
            "data_column": "Volume Benzalkonium Added",
            "out_column": "benzalkonium_cost",
        },
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self):
        calc_df = self._chemical_cost_calculations(
            start_date=self.run_date_start, end_date=self.run_date_end
        )

        # update the db (calling a function from .db_utils.py)
        update_table_rows_from_df(
            db_engine=self.db_engine,
            db_table_name=self.OUT_TABLE_NAME,
            update_data_df=calc_df,
        )

    def _chemical_cost_calculations(
        self, start_date: datetime, end_date: datetime
    ) -> None:
        """
        Method to calculate estimated costs from consumed chemicals and store in db_table
        """
        # query the chemical usage amounts from db
        data_col_names = [v["data_column"] for v in self.chem_costs.values()]
        calc_df = query_data_table_by_date_range(
            db_name_or_engine=self.db_engine,
            table_name="ponds_data",
            query_date_start=start_date,
            query_date_end=end_date,
            col_names=data_col_names,
        )
        # fill in zeroes for NaN chemical usage values
        calc_df = calc_df.fillna(0)

        # loop through chemicals in cost dict and add to "out_dict" according to the "out_column" value in cost dict
        # init list to append with output column names to filter output df
        out_cols = ["Date", "PondID"]
        for idx, subdict in enumerate(self.chem_costs.values()):
            in_col_name = subdict["data_column"]
            out_col_name = subdict["out_column"]
            col_cost = subdict["cost"]
            out_cols.append(out_col_name)
            calc_df[out_col_name] = calc_df[in_col_name].apply(
                lambda x: round(x * col_cost, 2)
            )
        # filter df to only the output columns
        calc_df = calc_df.loc[:, out_cols]
        return calc_df


class WeatherDataLoad(DBColumnsBase):
    OUT_TABLE_NAME = "daily_weather_data"
    DEPENDENCY_CLASSES = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self):
        weather_df = self._get_weather_data(
            start_date=self.run_date_start, end_date=self.run_date_end
        )

        # update the db (calling a function from .db_utils.py)
        update_table_rows_from_df(
            db_engine=self.db_engine,
            db_table_name=self.OUT_TABLE_NAME,
            update_data_df=weather_df,
            pk_cols=["Date"],
        )

    def _get_weather_data(self, start_date, end_date):
        from meteostat import Daily, Point, units

        # use meteostat.Point() to get weather data for geo location
        # this largely uses data for Deming NM airport weather station: "KDMN0" ... plus modeled data
        point = Point(31.7947, -107.7857, alt=1239)
        point.radius = 80000  # roughly 50 mile radius in meters (default for meteostat is 35000 meters- approx 22 miles)

        # Get daily data
        data = (
            Daily(loc=point, start=start_date, end=end_date)
            .convert(units.imperial)
            .fetch()
        )
        data = data.reset_index().rename(columns={"time": "Date"})
        data = data.rename(
            columns={
                "tavg": "temp_avg",
                "tmin": "temp_min",
                "tmax": "temp_max",
                "prcp": "precipitation",
                "wspd": "wind_speed",
                "pres": "pressure",
            }
        )
        data = data[
            [
                "Date",
                "temp_avg",
                "temp_min",
                "temp_max",
                "precipitation",
                "wind_speed",
                "pressure",
            ]
        ]
        return data


class ProcessingDataLoad(DBColumnsBase):
    OUT_TABLE_NAME = "daily_processing_data"
    DEPENDENCY_CLASSES = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self):
        out_df = self._load_processing_data(
            start_date=self.run_date_start, end_date=self.run_date_end
        )

        # update the db (calling a function from .db_utils.py)
        update_table_rows_from_df(
            db_engine=self.db_engine,
            db_table_name=self.OUT_TABLE_NAME,
            update_data_df=out_df,
            pk_cols=["Date"],
        )

    def _load_processing_data(self, start_date: datetime, end_date: datetime):
        # download data file
        proc_data_path = list(self._dl_sharepoint_files("processing_data").values())[0]

        # load downloaded file
        df = pd.read_excel(proc_data_path, sheet_name="SF Harvest")
        df = df.rename(columns={df.columns[0]: "Date"})
        # convert date column from string *use .normalize method to remove potential time data
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
        # drop columns without a header, assumed to be empty or unimportant
        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

        # construct a DF with combination of all possible Date and PondID columns for date range to ensure that no dates are missing (for indexing by date in db)
        date_range = pd.date_range(start=start_date, end=end_date)
        dates_df = pd.DataFrame(date_range, columns=["Date"])

        out_df = pd.merge(df, dates_df, how="outer", on="Date").sort_values(by="Date")
        out_df = out_df[out_df["Date"].between(start_date, end_date)]
        out_columns = [
            "Date",
            "Zobi Permeate Volume (gal)",
            "SF Reported Permeate Volume (gal)",
            "Calculated SF Permeate Volume (gal)",
            "Calculated Permeate Volume (gal)",
            "DFP Level (in)",
            "SF Slurry Produced (MT)",
            "Zobi Slurry Produced (MT)",
            "SW Dryer Totes",
            "SW Dryer Biomass (MT)",
            "Drum Dryer Totes",
            "Drum Dryer Biomass (MT)",
            "Gallons dropped",
            "HF1 Run Time (hours)",
            "HF2 Run Time (hours)",
            "SF1 Run Time (hours)",
            "SF2 Run Time (hours)",
            "SF3 Run Time (hours)",
            "Notes",
            "Volume Dropped from Ponds (gallons)",
            "Lazy River AFDW (g/L)",
            "SF Feed AFDW (g/L)",
            "Zobi Run TIme (hours)",
            "DD Run TIme (hours)",
            "SW Dryer Run TIme (hours)",
        ]
        return out_df[out_columns]


class HarvestDataLoad(DBColumnsBase):
    OUT_TABLE_NAME = "ponds_data"
    DEPENDENCY_CLASSES = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self):
        data_months = self._get_months_in_date_range(
            self.run_date_start, self.run_date_end
        )
        self.data_files = {}
        for month in data_months:
            self.data_files[month] = self._download_monthly_harvest_data_files(
                data_year=month[0], data_month=month[1]
            )

        out_df = self._load_harvest_data_from_files(
            self.run_date_start, self.run_date_end
        )

        # update the db (calling a function from .db_utils.py)
        update_table_rows_from_df(
            db_engine=self.db_engine,
            db_table_name=self.OUT_TABLE_NAME,
            update_data_df=out_df,
        )

    def _load_harvest_data_from_files(self, start_date, end_date):
        # init empty df to append rows to for each date
        combined_df = pd.DataFrame()
        for d in pd.date_range(start_date, end_date):
            # get the pd.ExcelFile object for date
            date_ExcelFile = self.data_files.get((d.year, d.month))
            for size_tag in ["1.1", "2.2"]:
                try:
                    # load sheet, skip rows 0-3, and remove rows with 'n/a' vals in the 'Pond ID' field
                    # convert Pond ID to str (otherwise it strips leading zeroes)
                    # and convert Time Open/Time Closed to str (because datetime vals aren't compatible with sqlite and the columns can just be converted whenever needed)
                    date_df = date_ExcelFile.parse(
                        sheet_name=f'{d.strftime("%y%m%d")} ({size_tag})',
                        header=4,
                        converters={"Pond ID": str, "Notes": str},
                    )
                    d_success = True
                except Exception as e:
                    print(
                        f"ERROR: {e} exception: Could not load harvest data for:",
                        d.strftime("%y/%m/%d"),
                        size_tag,
                    )
                    d_success = False

                if d_success:
                    # drop columns without a header
                    date_df = date_df.loc[:, ~date_df.columns.str.contains("^Unnamed")]
                    # strip whitespace from column headers
                    date_df = date_df.rename(columns=lambda x: x.strip())
                    # rename column headers
                    date_df = date_df.rename(
                        columns={
                            "Pond ID": "PondID",
                            "Initial Depth": "harvest_initial_depth",
                            "Final Depth": "harvest_final_depth",
                            "Volume Harvested (gallons)": "harvest_volume",
                            "Notes": "harvest_notes",
                        }
                    )
                    # filter/drop unnecessary columns
                    date_df = date_df[
                        [
                            "PondID",
                            "harvest_initial_depth",
                            "harvest_final_depth",
                            "harvest_volume",
                            "harvest_notes",
                        ]
                    ]
                    # drop rows with n/a in PondID (these rows should not contain any valid data)
                    # also drop rows with n/a in harvest_volume (due to data entry error???)
                    date_df = date_df.dropna(subset=["PondID", "harvest_volume"])
                    # add Date column
                    date_df["Date"] = d
                    # append date df to the combined output df
                    combined_df = combined_df.append(date_df)

        # combine duplicate rows (i.e., when a pond is harvested multiple times in one day, then get the total volume, total harvest drop time,
        # initial depth (from the first harvest), final depth (from the last last harvest), and concatenate any notes
        combined_df = (
            combined_df.groupby(["Date", "PondID"])
            .agg(
                {
                    "harvest_initial_depth": "max",
                    "harvest_final_depth": "min",
                    "harvest_volume": "sum",
                    "harvest_notes": list,
                }
            )
            .reset_index()
        )

        # re-process the harvest_notes column, since it was only combined into a list with groupby
        # remove n/a values and join into single string with a '/' separator if there are multiple non-null values
        combined_df["harvest_notes"] = combined_df["harvest_notes"].apply(
            lambda x: " / ".join([i for i in x if not pd.isna(i)])
        )

        # sort the combined_df
        combined_df = combined_df.sort_values(by=["Date", "PondID"])
        return combined_df

    def _get_months_in_date_range(self, start_date, end_date):
        """
        Returns a list of months between two dates.
        Needed to download data files which are separate per month
        """
        month_list = []
        while start_date <= end_date:
            if (start_date.year, start_date.month) not in month_list:
                # append a tuple of (year, month)
                month_list.append((start_date.year, start_date.month))
            start_date += pd.Timedelta(days=1)
        return month_list

    def _download_monthly_harvest_data_files(
        self, data_year: int, data_month: int
    ) -> pd.ExcelFile | None:
        """
        Find and download the "Pond Harvest Log" .xlsx file for the current month (i.e., '12_December Pond Harvest Log.xlsx')
        This method looks through sharepoint directory organized by: YEAR --> FILE ("mm_Month Pond Harvest Log.xlsx")

        params:
        --------
        - data_year: integer representing year
        - data_month: integer representing month (i.e., 12 for December)

        RETURNS -> pd.ExcelFile object (if successful download) or None (if failure)
        """
        # get datetime value from data_year and data_month, with date as the 1st (doesn't really matter which date specifically since isn't used)
        # this is just to easily convert from the integer representation into a search string for the month
        specify_date = datetime(data_year, data_month, 1)

        folder_id = load_setting("folder_ids")["harvest_data"]
        for year_dir in self.account.get_sharepoint_file_by_id(folder_id).get_items():
            # first look for the "year" directory
            if year_dir.name == str(specify_date.year):
                # look within the year subdirectory
                for month_file in year_dir.get_items():
                    # search for filename with regex: case-insensitive: ["mm_MMMM (ex: 01_January)" (1 or more whitespace chars) "pond harvest log" (zero or more whitespace chars) ".xlsx"]
                    # use re.search to find 3 groups in filename: date formatted as "mm_MMMM, "pond harvest log" (case insensitive), ".xlsx" (must be at end)
                    file_search = re.search(
                        r"(?i)({})\s+(Pond Harvest Log).*(\.xlsx$)".format(
                            specify_date.strftime("%m_%B")
                        ),
                        month_file.name,
                    )
                    if file_search:
                        print("FOUND FILE:", month_file.name)
                        dl_path_dir = Path("data_sources/tmp/")
                        dl_file_path = dl_path_dir / month_file.name
                        download_attempts = 5
                        while True:
                            month_file.download(to_path=dl_path_dir)
                            if dl_file_path.is_file():
                                # load excel file
                                excel_file = pd.ExcelFile(dl_file_path.as_posix())
                                # delete file after loading
                                dl_file_path.unlink()
                                return excel_file
                            else:
                                download_attempts -= 1
                                if download_attempts > 0:
                                    print("Waiting 5 seconds to try download again...")
                                    time.sleep(5)
                                    continue
                                else:
                                    raise Exception(
                                        f'ERROR: located but could not download daily data file for {datetime.strftime(specify_date, "%B %Y")}!'
                                    )

                print(
                    f'COULD NOT FIND HARVEST DATA FILE FOR {datetime.strftime(specify_date, "%B %Y")}!'
                )
                return None

    # def load_sfdata(self, excel_filename):
    #     print('Loading SF Data')
    #     df = pd.read_excel(excel_filename, sheet_name='Customer Log')
    #     df = df.rename(columns={df.columns[0]:'date'})
    #     df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.normalize() # convert date column from string *use .normalize method to remove potential time data
    #     df = df.set_index(df['date'].name) # set date column as index
    #     df = df.loc[:, ~df.columns.str.contains('^Unnamed')] # drop columns without a header, assumed to be empty or unimportant
    #     print('SF data loaded!')
    #     return df
