from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import warnings
#import math
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse, Rectangle
from .utils import generate_multipage_pdf
from .db_utils import query_data_table_by_date, query_data_table_by_date_range

class BaseReportProperties:
    @property
    def fontsize_base(self) -> int|float:
        if not hasattr(self, '_FONTSIZE_BASE'):
            self._FONTSIZE_BASE = 5
        return self._FONTSIZE_BASE
    
    @fontsize_base.setter
    def fontsize_base(self, base_font_size: float|int) -> None:
        # set default fontsize for matplotlib figure
        plt.rcParams.update({'font.size': base_font_size}) 
        self._FONTSIZE_BASE = base_font_size
        print('Fontsize updated to:', base_font_size)

    @property
    def fontsizes(self) -> dict:
        return {'default': self.fontsize_base,
                'small': self.fontsize_base * 0.75,
                'medium': self.fontsize_base * 1.125,
                'large': self.fontsize_base * 1.25,
                'title': self.fontsize_base * 1.75 } 

    @property
    def report_title(self) -> str:
        if not hasattr(self, '_report_title'):
            self._report_title = ''
        return self._report_title

    @report_title.setter
    def report_title(self, title_str: str) -> None:
        self._report_title = title_str

    @property
    def report_date(self) -> datetime:
        if not hasattr(self,'_report_date'):
            raise Exception(f'ERROR! Could not create instance of {self.__name__} because no "report_date" was provided!')
        return self._report_date

    @report_date.setter
    def report_date(self, report_date_: datetime) -> None:
        if not isinstance(report_date_, datetime):
            raise Exception(f'ERROR! Could not create instance of {self.__name__} because an invalid report_date was provided! It should be a datetime.datetime value!')
        # Normalize report_date to remove potential time data and prevent possible key errors when selecting date range from data
        self._report_date = pd.to_datetime(report_date_).normalize() 
        self._report_date.as_str_filename = self._report_date.strftime("%Y-%m-%d")
        self._report_date.as_str_print = self._report_date.strftime("%-m/%-d/%Y")
        
class BasePondsGridReport(ABC, BaseReportProperties):
    def __init__(self, report_date: datetime, save_output: bool = True):
        self.report_date = report_date
        self.save_output = save_output

    def run(self) -> str|None: # returns output file path if self.save_output == True, else returns None
        '''
        Method to generate the report
        '''
        # load data (abstract method, implement for each inherited class instance)
        self.load_data()
        
        N_ROWS_SMALL = 12
        N_ROWS_LARGE = 10
        N_COLS_SMALL = 4
        N_COLS_LARGE = 2
        N_EMPTY_ROWS_BOTTOM = 0

        # generate labels for PondID's in a list (with each row being a sublist)
        title_labels = []
        for row in range(1,N_ROWS_SMALL+1):
            title_labels.append([])
            for col in range(1,N_COLS_SMALL+1):
                if row < 10:
                    title_labels[row-1].append(f'0{row}0{col}')  
                else:
                    title_labels[row-1].append(f'{row}0{col}')  
        for idx_row, row in enumerate(title_labels):
            if idx_row < N_ROWS_LARGE:
                for col in [6,8]:
                    if idx_row+1 < 10:
                        row.append(f'0{idx_row+1}0{col}')
                    else:
                        row.append(f'{idx_row+1}0{col}')
            else: 
                # append blanks for the two large columns with only 10 rows 
                [row.append(f'BLANK {idx_row+1}-{c}') for c in [6,8]]
        
        # flatten title_labels for indexing from gridspec plot generation
        title_labels = [label for item in title_labels for label in item]
        
        # Initialize figure
        plt_width = 8.27 #8.27
        plt_height = 11.69 # 11.69
        scale_factor = 2
        self.fig = plt.figure(figsize=(plt_width*scale_factor, plt_height*scale_factor))
        
        # setup GridSpec for subplots in a grid
        # set left and right margins to equal roughly 0.5" margins on A4 size paper
        # leave room on top margin for title, leave room on bottom margin for additional notes, aggregations, etc.
        plot_grid = gridspec.GridSpec(nrows=max(N_ROWS_SMALL, N_ROWS_LARGE)+N_EMPTY_ROWS_BOTTOM, ncols=N_COLS_SMALL+N_COLS_LARGE, figure=self.fig, left=0.06, right=0.94, top=0.93, bottom=0.2, wspace=0.05, hspace=0.1)

        self.fig.suptitle(f'{self.report_title}\n{self.report_date.as_str_print}', fontweight='bold', fontsize=self.fontsizes['title'], y=0.97)
        
        # iterate through each subplot in the grid
        for idx, subplot_spec in enumerate(plot_grid):
            pond_id = title_labels[idx]
            self.plot_each_pond(subplot_spec, pond_id)

        #self.fig.text(0.125, 0.01, 'test figure text', ha='left', va='top')
        self.plot_annotations()
            
        #self.fig.show()
        if self.save_output:
            out_path = Path(f'output_files/{self.report_title} {self.report_date.as_str_filename}.pdf')
            self.fig.savefig(out_path.as_posix())#, bbox_inches='tight' # reduce DPI by scale factor... should result in output file actual size being accurate to A4 sheet size?
            return out_path.as_posix()
        else:
            return None

    def subplot_ax_format(self, subplot_ax, fill_color: str|None = None):
        '''
        Method to apply formatting to subplot axes, setting 
        '''
        subplot_ax.spines['top'].set_visible(False)
        subplot_ax.spines['right'].set_visible(False)
        subplot_ax.spines['bottom'].set_visible(False)
        subplot_ax.spines['left'].set_visible(False)
        subplot_ax.get_xaxis().set_ticks([])
        subplot_ax.get_yaxis().set_ticks([])
        if fill_color:
            subplot_ax.set_facecolor(fill_color)
        self.fig.add_subplot(subplot_ax)
    
    def plot_legend(self, fig, ax, legend_data, x_align, y_spacing, y_align=0.9):
        '''
        Method to plot outlined legend boxes, using a dict of labels/data, and a separate dict of x_alignments (see examples of dict structure in calls to this function)
        '''
        y_align = y_align # start upper y-coord for plotting legend
        max_coords = [1,1,0,0] #  initialize for tracking the max coordinates of legend items for printing a box around all the legend items (x0, y0, x1, y1)
        data_rows_x_bound = [1,0]
        data_rows_y_bounds = [] 
        indicator_plots = []
        for row in legend_data:
            row_y_bound = [1,0]
            row_fill_color = 'white'
            if 'fill_color' in row:
                row_fill_color = row['fill_color']

            for idx, (label, properties) in enumerate(row['labels'].items()):
                underline = False
                text_weight = 'normal'
                text_color = 'black'

                if 'weight' in properties:
                    if properties['weight'] == 'underline': # check if 'underline' option since it's not an option in matplotlib and needs to be added after text with an annotation using text bounding box coords
                        underline = True
                    elif properties['weight'] == 'bold': 
                        text_weight = 'bold'

                if 'color' in properties:
                    text_color = properties['color']

                if 'indicator_fill' in properties: # plot indicators instead of normal text
                    # Get transform info to properly plot circles, using Ellipse. Otherwise they are deformed 
                    # ref: https://stackoverflow.com/questions/9230389/why-is-matplotlib-plotting-my-circles-as-ovals
                    ax_x0, ax_y0 = ax.transAxes.transform((0, 0)) # lower left in pixels
                    ax_x1, ax_y1 = ax.transAxes.transform((1, 1)) # upper right in pixes
                    ax_dx = ax_x1 - ax_x0
                    ax_dy = ax_y1 - ax_y0
                    ax_maxd = max(ax_dx, ax_dy)
                    circle_width = .07 * ax_maxd / ax_dx
                    circle_height = .07 * ax_maxd / ax_dy

                    # plot indicator
                    t = ax.add_patch(Ellipse((x_align[properties['align']][0], y_align), circle_width, circle_height, color=properties['indicator_fill'], fill=properties['indicator_fill'], clip_on=False))
                    ax.text(x_align[properties['align']][0], y_align, label, color='white', ha='center', va='center', fontweight='bold', fontsize='x-small', linespacing=0.7, clip_on=False)

                else:
                    t = ax.text(x_align[properties['align']][0], y_align, label, ha = x_align[properties['align']][1], va = 'center', weight=text_weight, color=text_color)

                # get bounding box of plotted items
                bb = t.get_window_extent(renderer=fig.canvas.get_renderer()).transformed(ax.transAxes.inverted())

                if underline:
                    ax.annotate('', xy=(bb.x0-0.01,bb.y0), xytext=(bb.x1+0.01,bb.y0), xycoords="axes fraction", arrowprops=dict(arrowstyle="-", color='k'))

                # update the max coord boundaries for the rows that are assigned a color
                if row_fill_color != 'white':  
                    #print(label, bb)
                    if 'excl_color' not in properties: # add option of excluding a row item from being colored
                        if bb.x0 < data_rows_x_bound[0]:
                            data_rows_x_bound[0] = bb.x0
                        if bb.x1 > data_rows_x_bound[1]:
                            data_rows_x_bound[1] = bb.x1
                        if bb.y0 < row_y_bound[0]:
                            row_y_bound[0] = bb.y0
                        if bb.y1 > row_y_bound[1]:
                            row_y_bound[1] = bb.y1

                # update max_coords for bounding box around legend and for x-bounds of rows that will be filled with color
                if bb.x0 < max_coords[0]:
                    max_coords[0] = bb.x0
                if bb.y0 < max_coords[1]:
                    max_coords[1] = bb.y0
                if bb.x1 > max_coords[2]:
                    max_coords[2] = bb.x1
                if bb.y1 > max_coords[3]:
                    max_coords[3] = bb.y1

            y_align -= y_spacing # decrease y coord for each line for spacing       

            if row_fill_color != 'white':
                data_rows_y_bounds.append((row_y_bound,row_fill_color))

        # plot colored rectangle for corresponding row
        row_box_padding = 0.01
        for row_y in data_rows_y_bounds:
            ax.add_patch(Rectangle((data_rows_x_bound[0]-row_box_padding,row_y[0][0]-row_box_padding),data_rows_x_bound[1]-data_rows_x_bound[0]+row_box_padding*2,row_y[0][1]-row_y[0][0]+row_box_padding*2,linewidth=0,edgecolor=None,facecolor=row_y[1], clip_on=False))
        
        # plot color filled box around legend text
        box_padding = 0.03
        box_xy = (max_coords[0]-box_padding, max_coords[1]-box_padding)
        box_width = max_coords[1]-max_coords[0]
        ax.add_patch(Rectangle((max_coords[0]-box_padding,max_coords[1]-box_padding),max_coords[2]-max_coords[0]+box_padding*2,max_coords[3]-max_coords[1]+box_padding*2,linewidth=1,edgecolor='black',facecolor='none',clip_on=False))
        
    @abstractmethod
    def load_data(self):
        pass
        '''
        Load data needed for report as a class attribute
        '''
    
    @abstractmethod
    def plot_each_pond(self, subplot_spec: gridspec.SubplotSpec, pond_id: str, n_rows: int, n_cols: int):
        pass
        '''
        Ex: 
        subplot_grid = gridspec.GridSpecFromSubplotSpec(nrows=n_rows,ncols=n_cols,subplot_spec=subplot_spec, wspace=-0.01, hspace=-0.01) # slightly negative vert/horiz padding to eliminate spaces in subplot  
        
        title_ax = plt.Subplot(self.fig, subplot_grid[:2,:]) # access subplot grid with [rows, columns] with zero indexing, this example gets he first 2 rows (indexed as row 0 and row 1) and all columns
        self.subplot_ax_format(title_ax, fill_color='red') # apply formatting to subplot ax that removes all borders, tick marks, etc., with optional fill color
        title_ax.text(0.5, 0.5, 'test title text', ha='center')
                                    
        data_ax = plt.Subplot(self.fig,subplot_grid[2:, :]) # access subplot grid with [rows, columns] with zero indexing, this example gets he first last 3 rows rows (indexed as row 2, 3, and 4) and all columns
        self.subplot_ax_format(data_ax, fill_color='blue')
        data_ax.text(0.5, 0.5, 'test data text', ha='center')
        '''   

    @abstractmethod
    def plot_annotations(self):
        '''
        Add annotations with additional data, legend, etc.
        '''
        pass
   
class TestReport(BasePondsGridReport):
    def load_data(self):
        pass
    
    def plot_each_pond(self, subplot_spec, pond_id, n_rows=5, n_cols=3):
        if 'BLANK' in pond_id:
            pass
        else:
            subplot_grid = gridspec.GridSpecFromSubplotSpec(n_rows,n_cols,subplot_spec=subplot_spec, wspace=-0.01, hspace=-0.01) # slightly negative vert/horiz padding to eliminate spaces in subplot  
            
            title_ax = plt.Subplot(self.fig, subplot_grid[:2,:])
            self.subplot_ax_format(title_ax, fill_color='lightgrey')
            title_ax.text(0.5, 0.9, f'Pond: {pond_id}', ha='center', va='top', fontsize=self.fontsizes['large']) # override the default fontsize 
            title_ax.text(0.5, 0.5, 'test title text', ha='center', va='top', fontsize=self.fontsizes['medium']) # override the default fontsize 
                                        
            data_ax = plt.Subplot(self.fig,subplot_grid[2:, :])
            self.subplot_ax_format(data_ax, fill_color=None)
            data_ax.text(0.5, 0.5, 'test data text', ha='center', va='center')
    
    def plot_annotations(self):
        pass

class ExpenseGridReport(BasePondsGridReport):
    def __init__(self, report_date: datetime, **kwargs):
        # init as base report class first! prior to setting custom class properties
        super().__init__(report_date, **kwargs) 

        # set report properties
        self.report_title = 'Expenses by Pond'
        self.fontsize_base = 12 # sets/overrides base fontsize

    def load_data(self) -> None:
        # get start dates for both MTD and YTD expenses
        ytd_month_start = 9
        if self.report_date.month < ytd_month_start:
            self.ytd_start = datetime(self.report_date.year - 1, ytd_month_start, 1)
        else:
            self.ytd_start = datetime(self.report_date.year, ytd_month_start, 1)
        self.mtd_start = datetime(self.report_date.year, self.report_date.month, 1)
        
        # Query ytd expense & estimated harvested data from DB, returns a pandas dataframe of data for each day and each pond
        self.ytd_expense_data = query_data_table_by_date_range(db_name_or_engine= 'gsf_data', table_name='ponds_data_expenses', query_date_start=self.ytd_start, query_date_end=self.report_date)
        self.ytd_harvest_data = query_data_table_by_date_range(db_name_or_engine= 'gsf_data', table_name='ponds_data_calculated', query_date_start=self.ytd_start, query_date_end=self.report_date, col_names=['est_harvested', 'est_split'])
        
        # get mtd expense & harvested data by filtering ytd_expense data
        self.mtd_expense_data = self.ytd_expense_data[self.ytd_expense_data['Date'] >= self.mtd_start.strftime('%Y-%m-%d')]
        self.mtd_harvest_data = self.ytd_harvest_data[self.ytd_harvest_data['Date'] >= self.mtd_start.strftime('%Y-%m-%d')]
           
    def plot_each_pond(self, subplot_spec, pond_id, n_rows=5, n_cols=3):
        if 'BLANK' in pond_id:
            pass
        else:
            # Plot subplot title
            subplot_grid = gridspec.GridSpecFromSubplotSpec(n_rows,n_cols,subplot_spec=subplot_spec, wspace=-0.01, hspace=-0.01) # slightly negative vert/horiz padding to eliminate spaces in subplot
            title_ax = plt.Subplot(self.fig, subplot_grid[:2,:])
            self.subplot_ax_format(title_ax, fill_color='lightgrey')
            title_ax.text(0.5, 0.5, f'{pond_id}', ha='center', va='center', fontsize=self.fontsizes['default']) # override the default fontsize 
            
            # Prep subplot data
            # call sum(numeric_only=True) first to sum each numeric column, then call sum() again on the resulting series of column subtotals, to get the overall sum of expenses for each pond
            pond_mtd_expense = self.mtd_expense_data[self.mtd_expense_data['PondID'] == pond_id].sum(numeric_only=True).sum()
            pond_ytd_expense = self.ytd_expense_data[self.ytd_expense_data['PondID'] == pond_id].sum(numeric_only=True).sum()
            pond_mtd_harvested = self.mtd_harvest_data[self.mtd_harvest_data['PondID'] == pond_id].sum(numeric_only=True).sum()
            pond_ytd_harvested = self.ytd_harvest_data[self.ytd_harvest_data['PondID'] == pond_id].sum(numeric_only=True).sum()

            # Get data subplot grid
            data_ax = plt.Subplot(self.fig,subplot_grid[2:, :])
            self.subplot_ax_format(data_ax, fill_color=None)
            
            # Plot subplot data, if there are expenses to show
            if pond_ytd_expense > 0:
                # left justify categories
                data_ax.text(0.05, 0.9, f'MTD Expense:\nMTD Calc Harvested:\n\nYTD Expense:\nYTD Calc Harvested:', ha='left', va='top', multialignment='left', fontsize=self.fontsizes['small'])
                # right justify data
                mtd_expense_fmt = f'${int(pond_mtd_expense):,}' if pond_mtd_expense > 0 else '- '
                mtd_harvested_fmt = f"{int(pond_mtd_harvested):,} kg" if pond_mtd_harvested >= 1 else "- "
                ytd_expense_fmt = f'${int(pond_ytd_expense):,}' if pond_ytd_expense > 0 else '- '
                ytd_harvested_fmt = f"{int(pond_ytd_harvested):,} kg" if pond_ytd_harvested >= 1 else "- "
                data_ax.text(0.9, 0.9, f'{mtd_expense_fmt}\n{mtd_harvested_fmt}\n\n{ytd_expense_fmt}\n{ytd_harvested_fmt}', ha='right', va='top', ma='right', fontsize=self.fontsizes['small'])
                
    def plot_annotations(self):
        # add note next to title regarding YTD start
        self.fig.text(0.7, 0.95, f'YTD Period Beginning: {self.ytd_start.strftime("%B %Y")}', ha='left', va='top', fontsize=self.fontsizes['default'])

        # Query ytd co2 cost data
        # (co2 cost cannot currently be split out per pond, so can only report an aggregate cost)
        ytd_co2_cost_data = query_data_table_by_date_range(db_name_or_engine= 'gsf_data', table_name='co2_usage', query_date_start=self.ytd_start, query_date_end=self.report_date, col_names=['total_co2_cost'])
        mtd_co2_cost_data = ytd_co2_cost_data[ytd_co2_cost_data['Date'] >= self.mtd_start.strftime('%Y-%m-%d')]
        
        # show aggregate co2 costs
        mtd_co2_cost = mtd_co2_cost_data['total_co2_cost'].sum()
        ytd_co2_cost = ytd_co2_cost_data['total_co2_cost'].sum()
        # left justify categories
        self.fig.text(0.07, 0.17, f'MTD Total CO2 Cost:\nYTD Total CO2 Cost:', ha='left', va='top', ma='left', weight='bold', fontsize=self.fontsizes['medium'])
        # right justify data
        self.fig.text(0.23, 0.17, f'${int(mtd_co2_cost):,}\n${int(ytd_co2_cost):,}', ha='left', va='top', ma='right', fontsize=self.fontsizes['medium'])
        
        # call sum(numeric_only=True) first to sum each numeric column, then call sum() again on the resulting series of column subtotals, to get the overall sum of expenses for entire farm
        mtd_expense_total = self.mtd_expense_data.sum(numeric_only=True).sum() + mtd_co2_cost
        ytd_expense_total = self.ytd_expense_data.sum(numeric_only=True).sum() + ytd_co2_cost
        # left justify categories
        self.fig.text(0.07, 0.14, f'MTD Total Expenses:\nYTD Total Expenses:', ha='left', va='top', ma='left', weight='bold', fontsize=self.fontsizes['medium'])
        # right justify data
        self.fig.text(0.23, 0.14, f'${int(mtd_expense_total):,}\n${int(ytd_expense_total):,}', ha='left', va='top', ma='right', fontsize=self.fontsizes['medium'])

        # show aggregate estimated harvested totals
        mtd_harvested_total = self.mtd_harvest_data.sum(numeric_only=True).sum()
        ytd_harvested_total = self.ytd_harvest_data.sum(numeric_only=True).sum()
        # left justify categories
        self.fig.text(0.33, 0.14, f'MTD Total *Calculated Harvested:\nYTD Total *Calculated Harvested:', ha='left', va='top', ma='left', weight='bold', fontsize=self.fontsizes['medium'])
        # right justify data
        self.fig.text(0.575, 0.14, f'{int(mtd_harvested_total):,} kg\n{int(ytd_harvested_total):,} kg', ha='left', va='top', ma='right', fontsize=self.fontsizes['medium'])
        
        # show aggregate costs per kg
        # left justify categories
        self.fig.text(0.67, 0.14, f'MTD Avg $/kg:\nYTD Avg $/kg:', ha='left', va='top', ma='left', weight='bold', fontsize=self.fontsizes['medium'])
        # right justify data
        mtd_avg_cost = f'${mtd_expense_total/mtd_harvested_total:.2f} /kg' if mtd_harvested_total > 0 else '-   '
        ytd_avg_cost = f'${ytd_expense_total/ytd_harvested_total:.2f} /kg' if ytd_harvested_total > 0 else '-   '
        self.fig.text(0.79, 0.14, f'{mtd_avg_cost}\n{ytd_avg_cost}', ha='left', va='top', ma='right', fontsize=self.fontsizes['medium'])

        # add footnote regarding calculated mass
        self.fig.text(0.07, 0.11, "*Calculated harvest mass is based on pond 'depth', 'density', and '% nanno' measurements to estimate change in mass between the day of harvest the next day with available data", ha='left', va='top', fontsize=self.fontsizes['small'])
        
        # show ytd mass breakout by harvest and splits
        # ytd_harvest_harvested = self.ytd_harvest_data.loc[:,'est_harvested'].sum()
        # ytd_harvest_split = self.ytd_harvest_data.loc[:, 'est_split'].sum()
        # self.fig.text(0.07, 0.14, 'YTD Mass to Processing\nYTD Mass to Splits', ha='left', va='top', ma='left', weight='bold', fontsize=self.fontsizes['medium'])
        # self.fig.text(0.26, 0.14, f'{int(ytd_harvest_harvested):,} kg\n{int(ytd_harvest_split):,} kg', ha='left', va='top', ma='right', fontsize=self.fontsizes['medium'])
        
        # plot some temp info about expense calcs
        self.fig.text(0.7, 0.3, 'Expense unit costs:\n-------------------------------------------------\nUAN-32:\nFertilizer 10-34:\nBleach:\nCO2:\nTrace:\nIron:\nCal Hypo:\nBenzalkonium:', ha='left', va='top', ma='left', fontsize=self.fontsizes['default'])
        self.fig.text(0.82, 0.3, '\n\n$2.08/gal\n$4.25/gal\n$3.15/gal\n$0.15/lb\n$0.41/gal\n$0.60/gal\n$3.17/kg\n$49.07/gal', ha='left', va='top', ma='left', fontsize=self.fontsizes['default'])
        
class BaseTableReport(ABC, BaseReportProperties):
    def __init__(self, report_date: datetime, report_subtitle: str|None = None, **kwargs): 
        self.report_date = report_date
        self.report_subtitle = report_subtitle # OPTIONAL formatted string for adding subtitle/additional information below the report title
        
    @property
    def inter_table_spacing(self) -> float|int:
        if not self._inter_table_spacing:
            self._inter_table_spacing = 0.03
        return self._inter_table_spacing

    @inter_table_spacing.setter
    def inter_table_spacing(self, spacing_value: float|int) -> None:
        self._inter_table_spacing = spacing_value
    
    @abstractmethod
    def load_data(self) -> list[dict]:
        ''' 
        Load data and return a list of dicts:
            dict keys:   "title": str
                         "df": pd.DataFrame
        '''
        pass
    
    def run(self) -> str: # returns output file path
        # helper function to generate a new figure (one for each "page" required)
        def gen_fig(title_page=False):
            ''' 
            params: 
                title_page: bool:
                    - True: add title text and adjust spacing when a title is added
                    - False: no title, start table plot from the top of figure
            '''
            # Initialize plot
            plt_width = 8.5
            plt_height = 11
            scale_factor = 1
            fig, ax = plt.subplots(figsize=(plt_width*scale_factor, plt_height*scale_factor))
            ax_border_pad = 0.5 # margin padding (in inches) between ax and full fig on all sides, just to maintain a margin for printing without needing to rescale, as well as room for header/footer info to be added (page numbers, etc.)
            ax_width = (plt_width - ax_border_pad*2) / plt_width
            ax_height = (plt_height - ax_border_pad*2) / plt_height 
            ax_left_x = (1 - ax_width)/2
            ax_bottom_y = (1 - ax_height)/2
            ax.set_position([ax_left_x, ax_bottom_y, ax_width, ax_height]) # set ax to fill entire figure (except for the border padding)
            ax.axis('off')
            if title_page:
                title_text = f'{self.report_title} - {self.report_date.as_str_print}' 
                t1 = ax.text(0.5, 1, title_text, ha='center', va='top', fontsize=14, weight='bold')
                if self.report_subtitle:
                    t1 = ax.text(0.5,0.992, self.report_subtitle, ha='center', va='top', fontsize=8)  
                # get the y-coordinate where tables can begin plotting on figure (after title text, or at top of ax if title text isn't present: i.e., page 2+)
                y0 = t1.get_window_extent(renderer=fig.canvas.get_renderer()).transformed(ax.transAxes.inverted()).y0 - self.inter_table_spacing
            else:
                y0 = 1
            return fig, ax, y0

        # Helper function to plot each sub-table (provided as a pd.DataFrame) within the figure
        # if table falls outside of the figure dimensions, then the table is removed from the current figure 
        # and the string 'start_new_page' is returned to indicate that another figure should be generated for a new page
        ## TODO: implement an option of splitting a table between multiple pages, rather than forcing a table to fit on single page
        def plot_table(table_title, df, ax, y_start, data_fontsize=6.4, title_fontsize=8):
            table_title = ax.text(0.5, y_start, table_title, ha='center', va='top', fontsize=title_fontsize, weight='bold')
            table_title_dims = table_title.get_window_extent(renderer=fig.canvas.get_renderer()).transformed(ax.transAxes.inverted())

            table_y_max = table_title_dims.y0 - 0.0025 # add 0.0025 spacing (in terms of ax y-coordinate) between the title and top of table

            # plot a temporary table to get its dimensions, then remove it 
            # need to do this so that tables can be plotted on a single ax with even vertical spacing between them
            # the important dimension to capture here is the table height (after setting fontsize) because that will be variable depending on the length of the data, 
            # and the 'bbox' parameter for a matplotlib table (to plot in an exact specified location) requires a lower y-bound value, which isn't possible without knowing its height relative to where the table should start
            # (it would be more efficient to simply move this table, but that doesn't seem to be possible with matplotlib, so just removing and regenerating it instead once bounding box coordinates are calculated)
            temp_table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center')
            temp_table.auto_set_font_size(False)
            temp_table.set_fontsize(data_fontsize)
            temp_table.auto_set_column_width(col=list(range(len(df.columns))))
            temp_table_dims = temp_table.get_window_extent(renderer=fig.canvas.get_renderer()).transformed(ax.transAxes.inverted())
            temp_table.remove()

            table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', colColours=['whitesmoke']*len(df.columns), bbox=[temp_table_dims.x0, table_y_max - temp_table_dims.height, temp_table_dims.width, temp_table_dims.height])
            table.auto_set_font_size(False)
            table.set_fontsize(data_fontsize)
            table.auto_set_column_width(col=list(range(len(df.columns))))
            table_dims = table.get_window_extent(renderer=fig.canvas.get_renderer()).transformed(ax.transAxes.inverted())

            # catch if the lower y-bound of the table is less than 0 (meaning it exceeded the ax size), if so then remove the plotted data and return signal to start a new page
            if table_dims.y0 < 0: 
                table_title.remove()
                table.remove()
                return 'start_new_page'     
            else:
                return table_dims.y0 # return the minimum/bottom y_coordinate value for the table
        
        # load data
        # returns a list of dictionaries with {"title": str, "df", pd.DataFrame} for each list item
        self.data_dict_list = self.load_data()
        
        fig_list = [] # initialize list of figs (one for each output page, as necessary)   
        fig, ax, y_align = gen_fig(title=True)
        table_spacing = 0.035
        
        for curr_table in self.data_dict_list:
            # call plot_table function, which returns the next y_coordinate for plotting a table to
            # once the y_coordinates for a table are outside of the ax bound (< 0) then the function will return the string 'start_new_page'
            # in that case, append the current figure to fig_list, then generate a new figure and re-plot the table on the new page
            y_align = plot_table(table_title=curr_table.get('title'), df=curr_table.get('df'), ax=ax, y_start=y_align)
            if y_align == 'start_new_page':
                fig_list.append(fig)
                fig, ax, y_align = gen_fig()
                y_align = plot_table(table_title=curr_table.get('title'), df=curr_table.get('df'), ax=ax, y_start=y_align)
            y_align -= self.inter_table_spacing
        fig_list.append(fig) # append the last/most-recent figure to the fig list before generating output file    

        filename = f'./output_files/{self.report_title} {self.report_date.as_str_filename}.pdf'
        out_filename = generate_multipage_pdf(fig_list, filename, add_pagenum=True, bbox_inches=None)
        return out_filename

class NewPotentialHarvestsReport(BaseTableReport):
    def __init__(self, report_date: datetime, **kwargs):
        # init as base report class first! prior to setting custom class properties
        super().__init__(report_date, **kwargs) 

        # set report properties
        self.report_title = 'Expenses by Pond'
        self.fontsize_base = 12 # sets/overrides base fontsize

    def load_data(self):
        pass
        ''' 
        load: by pond:
            - drop to (depth in inches) [CALCULATE current depth - harvestable depth add to calcs db]
            - days since harvested [CALCULATE add to calcs db]
            - harvestable depth [QUERY]
            - harvestable gallons [QUERY]
            - harvestable mass [QUERY]
            - running total gallons [QUERY]
            - running total mass [QUERY]
        '''

class PotentialHarvestsReport(): 
    ''' 
    JUST A PLACEHOLDER CLASS FOR THIS CODE FOR NOW
    - implement as a tabular report base class with data provided
    - implement title text as an abstract method???
    '''
    def plot_potential_harvests(self, overall_spacing=0.03): 
        potential_harvests_dict = self.potential_harvests_dict
        select_date = self.select_date

        # add 'columns' values to potential_harvests dict, get list of keys from the first column/first pond in 'data'
        if len(potential_harvests_dict.get('data', {})) > 0:
            potential_harvests_dict['columns'] = list(potential_harvests_dict['data'][list(potential_harvests_dict['data'].keys())[0]][0].keys())
        else:
            potential_harvests_dict['columns'] = []

        def gen_fig(title=False):
            # Initialize plot
            plt_width = 8.5
            plt_height = 11
            scale_factor = 1
            fig, ax = plt.subplots(figsize=(plt_width*scale_factor, plt_height*scale_factor))
            ax_border_pad = 0.5 # margin padding (in inches) between ax and full fig on all sides, just to maintain a margin for printing without needing to rescale, as well as room for header/footer info to be added (page numbers, etc.)
            ax_width = (plt_width - ax_border_pad*2) / plt_width
            ax_height = (plt_height - ax_border_pad*2) / plt_height 
            ax_left_x = (1 - ax_width)/2
            ax_bottom_y = (1 - ax_height)/2
            ax.set_position([ax_left_x, ax_bottom_y, ax_width, ax_height]) # set ax to fill entire figure (except for the border padding)
            ax.axis('off')
            #ax.yaxis.get_ticklocs(minor=True)
            #ax.minorticks_on()
            if title:
                title_text1 = f'Potential Harvests - {str(select_date.strftime("%-m/%-d/%y"))}' 
                title_text2 = ('\nponds with' + r'$\geq$' + '0.50 AFDW and' + r'$\geq$' + '3% EPA, with estimated harvest depth to reach 0.40 AFDW after top off to 13"\n\n' + 
                              f'Total estimated potential harvest mass: {potential_harvests_dict["aggregates"]["potential total mass"]}\n' +
                              f'Total estimated potential harvest volume: {potential_harvests_dict["aggregates"]["potential total volume"]}')
                t1 = ax.text(0.5, 1, title_text1, ha='center', va='top', fontsize=14, weight='bold')
                t2 = ax.text(0.5,0.992, title_text2, ha='center', va='top', fontsize=8)  
                # get the y-coordinate where tables can begin plotting on figure (after title text, or at top of ax if title text isn't present: i.e., page 2+)
                y0 = t2.get_window_extent(renderer=fig.canvas.get_renderer()).transformed(ax.transAxes.inverted()).y0 - overall_spacing
            else:
                y0 = 1
            return fig, ax, y0

        def plot_table(table_title, ax, y_start, data_fontsize=6.4, title_fontsize=8):
            table_title = ax.text(0.5, y_start, f'Column {table_title}', ha='center', va='top', fontsize=title_fontsize, weight='bold')
            table_title_dims = table_title.get_window_extent(renderer=fig.canvas.get_renderer()).transformed(ax.transAxes.inverted())

            table_y_max = table_title_dims.y0 - 0.0025 # add 0.0025 spacing (in terms of ax y-coordinate) between the title and top of table

            # plot a temporary table to get its dimensions, then remove it 
            # need to do this so that tables can be plotted on a single ax with even vertical spacing between them
            # the important dimension to capture here is the table height (after setting fontsize) because that will be variable depending on the length of the data, 
            # and the 'bbox' parameter for a matplotlib table (to plot in an exact specified location) requires a lower y-bound value, which isn't possible without knowing its height relative to where the table should start
            # (it would be more efficient to simply move this table, but that doesn't seem to be possible in matplotlib, so just removing and regenerating it instead once bounding box coordinates are calculated)
            temp_table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center')
            temp_table.auto_set_font_size(False)
            temp_table.set_fontsize(data_fontsize)
            temp_table.auto_set_column_width(col=list(range(len(df.columns))))
            temp_table_dims = temp_table.get_window_extent(renderer=fig.canvas.get_renderer()).transformed(ax.transAxes.inverted())
            temp_table.remove()

            table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', colColours=['whitesmoke']*len(df.columns), bbox=[temp_table_dims.x0, table_y_max - temp_table_dims.height, temp_table_dims.width, temp_table_dims.height])
            table.auto_set_font_size(False)
            table.set_fontsize(data_fontsize)
            table.auto_set_column_width(col=list(range(len(df.columns))))
            table_dims = table.get_window_extent(renderer=fig.canvas.get_renderer()).transformed(ax.transAxes.inverted())

            # catch if the lower y-bound of the table is less than 0 (meaning it exceeded the ax size), if so then remove the plotted data and return signal to start a new page
            if table_dims.y0 < 0: 
                table_title.remove()
                table.remove()
                return 'start_new_page'     
            else:
                return table_dims.y0 # return the minimum/bottom y_coordinate value for the table

        fig_list = [] # initialize list of figs (one for each output page, as necessary)
        tables_list = list(potential_harvests_dict['data'].keys())

        fig, ax, y_align = gen_fig(title=True)

        table_spacing = 0.035

        while tables_list:
            table_title = tables_list.pop(0) # set 'table_title' equal to the key from data

            # create dataframe from data in dict format 
            df = pd.DataFrame(potential_harvests_dict['data'][table_title], columns=potential_harvests_dict['columns']) 

            # call plot_table function, which returns the next y_coordinate for plotting a table to
            # once the y_coordinates for a table are outside of the ax bound (< 0) then the function will return the string 'start_new_page'
            # in that case, append the current figure to fig_list, then generate a new figure and re-plot the table on the new page
            y_align = plot_table(table_title, ax, y_align)
            if y_align == 'start_new_page':
                plt.show() # show plot for testing in jupyter nb
                fig_list.append(fig)
                fig, ax, y_align = gen_fig()
                y_align = plot_table(table_title, ax, y_align)
            y_align -= overall_spacing
        plt.show() # show plot for testing in jupyter nb
        fig_list.append(fig)     

        filename = f'./output_files/Potential Harvests {select_date.strftime("%Y-%m-%d")}.pdf'
        out_filename = generate_multipage_pdf(fig_list, filename, add_pagenum=True, bbox_inches=None)
        return out_filename
