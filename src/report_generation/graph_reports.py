from utils.db_utils import query_data_table_by_date, query_data_table_by_date_range
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from abc import ABC, abstractmethod
from datetime import datetime


class BaseTimeSeriesGraphReport(ABC):
    def __init__(
        self, begin_date: datetime, end_date: datetime, run: bool = True, **plot_kwargs
    ):
        if run:
            data_df = self.query_data(begin_date, end_date)
            if "Date" not in data_df.columns:
                raise Exception(
                    'ERROR: could not find a valid "Date" column in data, so can not generate plot!'
                )
            self.generate_plot(data_df, **plot_kwargs)

    @abstractmethod
    def generate_plot(self, data_df: pd.DataFrame, plot_type: str = "scatter") -> Path:
        pass

    @abstractmethod
    def query_data(self) -> pd.DataFrame:
        """
        Method to query data
        ex:
        data_df = query_data_table_by_date_range(db_name_or_engine=db_name, table_name=table_name, query_date_start=begin_date, query_date_end=end_date, col_names=col_names)
        return data_df
        """
        pass


class EPAHealthGraph(BaseTimeSeriesGraphReport):
    def __init__(self, begin_date: datetime, end_date: datetime, plot_type="scatter"):
        # init as base class which runs data query and generates report
        super().__init__(begin_date, end_date, plot_type)

    def generate_plot(self, data_df: pd.DataFrame) -> Path:
        import numpy as np
        import matplotlib.dates as mdates

        print("test df\n", data_df)

        # Initialize figure
        plt_width = 8.27  # 8.27
        plt_height = 11.69  # 11.69
        scale_factor = 2
        # self.fig = plt.figure(figsize=(plt_width*scale_factor, plt_height*scale_factor))
        # plt.figure(figsize=(plt_width*scale_factor, plt_height*scale_factor))
        fig, (ax1, ax2) = plt.subplots(
            2, figsize=(plt_width * scale_factor, plt_height * scale_factor)
        )
        fig.suptitle("EPA Growth Overview")

        # ax1.suptitle('EPA% Over Time', y=0.92)

        pond_id_columns = sorted(data_df["Column"].unique())
        # pond_id_columns = pond_id_columns.sort()

        plot_colors = ["red", "blue", "green", "orange", "purple", "yellow", "cyan"]
        legend_artists = []
        for idx, col in enumerate(pond_id_columns):
            print(f"{plot_colors[idx]} = {col}")

            x = data_df[data_df["Column"] == col]["Date"]
            y = data_df[data_df["Column"] == col]["epa_val"]
            # plt.scatter(x, y, color=plot_colors[idx], marker='x')
            ax1.plot(x, y, color=plot_colors[idx], alpha=0.2)

            # plot trend-line
            x_trendline = mdates.date2num(x)
            z = np.polyfit(x_trendline, y, 1)
            p = np.poly1d(z)
            ax1.plot(x, p(x_trendline), color=plot_colors[idx], linestyle="--")

        ax1.legend(pond_id_columns, title="Pond Column", loc="lower right")

        plt.show()

    def query_data(self, begin_date: datetime, end_date: datetime) -> pd.DataFrame:
        epa_data_df = query_data_table_by_date_range(
            db_name_or_engine="gsf_data_test3",
            table_name="epa_data",
            query_date_start=begin_date,
            query_date_end=end_date,
        )

        afdw_data_df = query_data_table_by_date_range(
            db_name_or_engine="gsf_data_test3",
            table_name="ponds_data",
            query_date_start=begin_date,
            query_date_end=end_date,
            col_names=["Column", "Filter AFDW"],
        )
        afdw_data_df = afdw_data_df.ffill().dropna()

        joined_df = epa_data_df.merge(afdw_data_df, how="left", on=["Date", "PondID"])
        joined_df = joined_df.groupby(by=["Date", "Column"]).mean().reset_index()
        joined_df["epa_fa_density_ratio"] = (
            joined_df["epa_val_total_fa"] / joined_df["Filter AFDW"]
        )
        joined_df["epa_density_ratio"] = joined_df["epa_val"] / joined_df["Filter AFDW"]

        return joined_df
