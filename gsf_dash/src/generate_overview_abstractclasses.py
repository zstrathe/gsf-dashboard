from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import warnings
import math
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse, Rectangle
from . import generate_multipage_pdf
from .db_utils import query_data_table_by_date, query_data_table_by_date_range

class BasePondsPlot(ABC):
    def __init__(self, report_date: datetime, save_output=True, db_name: str | None = None):
        self.fontsize_base = 5 
        self.report_date = pd.to_datetime(report_date).normalize() # Normalize report_date to remove potential time data and prevent possible key errors when selecting date range from data
        self._report_title = ''
        self.save_output = save_output

    @property
    def fontsizes(self) -> dict:
        return {'default': self._FONTSIZE_BASE,
                'small': self._FONTSIZE_BASE * 0.75,
                'medium': self._FONTSIZE_BASE * 1.25,
                'large': self._FONTSIZE_BASE * 1.75,
                'title': 12 } # use a constant font size for title

    @property
    def fontsize_base(self) -> int|float:
        return self._FONTSIZE_BASE
    
    @fontsize_base.setter
    def fontsize_base(self, base_font_size: float|int) -> None:
        # set default fontsize for matplotlib figure
        plt.rcParams.update({'font.size': base_font_size}) 
        self._FONTSIZE_BASE = base_font_size
        print('Fontsize updated to:', base_font_size)

    @property
    def report_title(self) -> str:
        return self._report_title

    @report_title.setter
    def report_title(self, title_str: str) -> None:
        self._report_title = title_str

    def run(self) -> None:
        '''
        Method to generate the report
        '''
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
    
        # Add empty rows for aggregations, notes, etc     
        #[title_labels.append([f'BLANK {r}-{c}' for c in range(1,7)]) for r in range(13,15)]
        
        # flatten title_labels for indexing from gridspec plot generation
        title_labels = [label for item in title_labels for label in item]
        
        # Initialize figure
        plt_width = 8.27 #8.27
        plt_height = 11.69 # 11.69
        scale_factor = 1
        self.fig = plt.figure(figsize=(plt_width*scale_factor, plt_height*scale_factor))
        
        # setup GridSpec for subplots in a grid
        # set left and right margins to equal roughly 0.5" margins on A4 size paper
        # leave room on top margin for title, leave room on bottom margin for additional notes, aggregations, etc.
        plot_grid = gridspec.GridSpec(nrows=max(N_ROWS_SMALL, N_ROWS_LARGE)+N_EMPTY_ROWS_BOTTOM, ncols=N_COLS_SMALL+N_COLS_LARGE, figure=self.fig, left=0.06, right=0.94, top=0.93, bottom=0.2, wspace=0.05, hspace=0.1)

        title_date = self.report_date.strftime('%-m/%-d/%Y')
        self.fig.suptitle(f'{self.report_title}\n{title_date}', fontweight='bold', fontsize=self.fontsizes['title'], y=0.98)
        
        # iterate through each subplot in the grid
        for idx, subplot_spec in enumerate(plot_grid):
            pond_id = title_labels[idx]
            self.plot_each_pond(subplot_spec, pond_id)

        #self.fig.text(0.125, 0.01, 'test figure text', ha='left', va='top')
        self.plot_aggregates()
            
        #self.fig.show()
        if self.save_output:
            out_path = Path(f'output_files/{self.report_title} {self.report_date.strftime("%Y-%m-%d")}.pdf')
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

        #print('rows x-bound:', data_rows_x_bound)    
        #print('row y-bounds:', data_rows_y_bounds)
        
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
    def plot_aggregates(self):
        pass
   
class TestReport(BasePondsPlot):
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
    
    def plot_aggregates(self):
        pass

class ExpenseReport(BasePondsPlot):
    def __init__(self, report_date: datetime, **kwargs):
        # init as base class first, to set custom class parameters
        super().__init__(report_date, **kwargs) 

        # set custom report class parameters
        self.report_title = 'Expenses by Pond'
        #self.fontsize_base = 6 # sets/overrides base fontsize
        
        data_col_names = ['uan32_cost', 'fert1034_cost', 'bleach_cost', 'co2_cost', 'trace_cost', 'iron_cost', 'cal_hypo_cost', 'benzalkonium_cost']
        
        # get start dates for both MTD and YTD expenses
        ytd_start = datetime(self.report_date.year, 9, 1)
        mtd_start = datetime(self.report_date.year, self.report_date.month, 1)
        
        # Query ytd expense data from DB, returns a pandas dataframe of data for each day and each pond
        self.ytd_expense_data = query_data_table_by_date_range(db_name_or_engine= 'gsf_data', table_name='ponds_data_expenses', query_date_start=ytd_start, query_date_end=report_date)
        
        # get mtd expense data by filtering ytd_expense data
        self.mtd_expense_data = self.ytd_expense_data[self.ytd_expense_data['Date'] >= mtd_start.strftime('%Y-%m-%d')]

        # get daily expense data by filtering mtd_expense_data
        self.day_expense_data = self.mtd_expense_data[self.mtd_expense_data['Date'] == report_date.strftime('%Y-%m-%d')]
        
    def plot_each_pond(self, subplot_spec, pond_id, n_rows=5, n_cols=3):
        if 'BLANK' in pond_id:
            pass
        else:
            # Plot subplot title
            subplot_grid = gridspec.GridSpecFromSubplotSpec(n_rows,n_cols,subplot_spec=subplot_spec, wspace=-0.01, hspace=-0.01) # slightly negative vert/horiz padding to eliminate spaces in subplot
            title_ax = plt.Subplot(self.fig, subplot_grid[:2,:])
            self.subplot_ax_format(title_ax, fill_color='lightgrey')
            title_ax.text(0.5, 0.5, f'{pond_id}', ha='center', va='center', fontsize=self.fontsizes['medium']) # override the default fontsize 
            
            # Prep subplot data
            # call sum(numeric_only=True) first to sum each numeric column, then call sum() again on the resulting series of column subtotals, to get the overall sum of expenses for each pond
            pond_day_expense = self.day_expense_data[self.day_expense_data['PondID'] == pond_id].sum(numeric_only=True).sum()
            pond_day_expense = round(pond_day_expense, 2) # rounding, need to fix at the database level...
            
            pond_ytd_expense = self.ytd_expense_data[self.ytd_expense_data['PondID'] == pond_id].sum(numeric_only=True).sum()
            pond_ytd_expense = round(pond_ytd_expense, 2) # rounding, need to fix at the database level...

            pond_mtd_expense = self.mtd_expense_data[self.mtd_expense_data['PondID'] == pond_id].sum(numeric_only=True).sum()
            pond_mtd_expense = round(pond_mtd_expense, 2) # rounding, need to fix at the database level...

            # Get data subplot grid
            data_ax = plt.Subplot(self.fig,subplot_grid[2:, :])
            self.subplot_ax_format(data_ax, fill_color=None)
            
            # Plot subplot data, if there are expenses to show
            if pond_ytd_expense > 0:
                # left justify categories
                data_ax.text(0.15, 0.9, f'MTD:\nYTD:', ha='left', va='top', multialignment='left', fontsize=self.fontsizes['medium'])
                # right justify data
                data_ax.text(0.8, 0.9, f'${pond_mtd_expense:,.2f}\n${pond_ytd_expense:,.2f}', ha='right', va='top', ma='right', fontsize=self.fontsizes['medium'])
                
    def plot_aggregates(self):
        # call sum(numeric_only=True) first to sum each numeric column, then call sum() again on the resulting series of column subtotals, to get the overall sum of expenses for entire farm
        day_expense_total = self.day_expense_data.sum(numeric_only=True).sum()
        ytd_expense_total = self.ytd_expense_data.sum(numeric_only=True).sum()
        mtd_expense_total = self.mtd_expense_data.sum(numeric_only=True).sum()
        # left justify categories
        self.fig.text(0.06, 0.17, f'Daily Total Expenses:\nMTD Total Expenses:\nYTD Total Expenses:', ha='left', va='top', ma='left', weight='bold', fontsize=self.fontsizes['large'])
        # right justify data
        self.fig.text(0.25, 0.17, f'{f"${day_expense_total:,.2f}" if day_expense_total > 0 else "-"}\n${mtd_expense_total:,.2f}\n${ytd_expense_total:,.2f}', ha='left', va='top', ma='right', fontsize=self.fontsizes['large'])

        # plot some temp info about expense calcs
        self.fig.text(0.67, 0.3, 'Expense unit cost assumptions:\n------------------------------\nUAN-32: $2.08/gal\nFertilizer 10-34: $4.25/gal\nBleach: $3.15/gal\nCO2: $0.15/lb\nTrace: ***$1/gal\nIron: ***$1/gal\nCal Hypo: $0.78/kg\nBenzalkonium: ***$1/gal\n\n***not accurate pricing', ha='left', va='top', ma='left', fontsize=self.fontsizes['large'])
        
        
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
