import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from dash import Dash, html, dcc, callback, Output, Input
import dash_bootstrap_components as dbc
from pandas import to_datetime

import sys
import os

# add .packages/ directory to sys.path, so that other relative modules can be imported
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR)) 

from utils.db_utils import get_available_date_range
from report_generation import PondsOverviewPlots, ExpenseGridReport


class DashApp:
    def __init__(self, debug_opt=False):
        app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

        app.layout = dbc.Container([
            
            dcc.Interval(
                id='interval-component',
                interval=300000, # 5 minutes in milliseconds
                n_intervals=0
            ),

            dbc.Row([
                dbc.Col([
                    html.H1("GSF Reports")
                    ], style={'textAlign':'center'})
            ], justify='center'),

            dbc.Row([
                dbc.Col([html.P('Date selection:'),], style={'textAlign': 'left', 'padding-top': 13}, width=1),
                dbc.Col([
                    dcc.DatePickerSingle(
                        id='dropdown-date-selection',
                        placeholder='Select a date')
                ], style={'textAlign': 'left'}, width=1)
             ], justify='center'),

            dbc.Row([
                dbc.Col([html.P('Report selection:'),], style={'textAlign': 'left', 'padding-top': 5}, width=1),
                dbc.Col([
                    dcc.Dropdown(['Scorecard Report', 'EPA Report', 'Potential Harvests Report', 'Expense Report'], 
                                 id='dropdown-report-selection',
                                 placeholder='Select a report'
                                 )
                ], style={'textAlign': 'left'}, width=1)
            ], justify='center'),

            dbc.Row([
               dbc.Col([
                    html.Br(),
                    html.Img(id='graph-content') 
               ], style={'textAlign': 'center'})
            ]),
        ], fluid=True)

        # callback to update the available date range from the database
        @callback(
            [Output('dropdown-date-selection', 'min_date_allowed'),
             Output('dropdown-date-selection', 'max_date_allowed'),
             Output('dropdown-date-selection', 'initial_visible_month')],
            Input('interval-component', 'n_intervals')
        )
        def update_date_range_options(n):
            print('Getting date options...')
            date_range_options = get_available_date_range(db_engine='gsf_data')
            return (date_range_options[0], date_range_options[-1], date_range_options[-1])

        # callback to update the selected report based on dropdowns
        @callback(
            Output('graph-content', 'src'),
            [Input('dropdown-date-selection', 'date'),
            Input('dropdown-report-selection', 'value')]
        )
        def update_graph(select_date, select_report):
            select_date = to_datetime(select_date)
            
            if any(i is None for i in [select_date, select_report]):
                return None
            
            # generate figure depending on report selection
            match select_report:
                case 'Scorecard Report':
                    fig = PondsOverviewPlots(select_date=select_date, run_all=False, save_output=False).plot_scorecard()
                case 'EPA Report':
                    fig = PondsOverviewPlots(select_date=select_date, run_all=False, save_output=False).plot_epa()
                case 'Potential Harvests Report':
                    fig = PondsOverviewPlots(select_date=select_date, run_all=False, save_output=False).plot_potential_harvests()
                case 'Expense Report':
                    fig = ExpenseGridReport(report_date=select_date, save_output=False).run()
                case _:
                    # Default case, return nothing if no report option selected
                    return None

            # Save it to a temporary buffer.
            buf = BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
	   	 	# Embed the result in the html output.
            fig_data = base64.b64encode(buf.getbuffer()).decode("ascii")
            fig_img = f'data:image/png;base64,{fig_data}'
            # close the figure to release it from memory
            plt.close(fig)
            return fig_img

        # run app, host must be '0.0.0.0' to be accessible when deployed via docker    
        app.run(debug=debug_opt, port=8051, host='0.0.0.0')

if __name__ == '__main__':
	DashApp(debug_opt=True)
