import base64
from io import BytesIO
import cairosvg
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from dash import Dash, html, dcc, callback, Output, Input, no_update
import dash_bootstrap_components as dbc
from pandas import to_datetime

import sys
import os

# add .packages/ directory to sys.path, so that other relative modules can be imported
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from utils.db_utils import get_available_date_range, load_sqlite_db
from report_generation import PondsOverviewPlots, ExpenseGridReport
from dash_app.cache_manager import FileCache

REPORT_OPTIONS = ["Scorecard Report",
                  "EPA Report",
                  "Recommended Harvests Report",
                  "Expense Report"]

def serve_layout():
    """
    Serve layout as a func so that options can be updated on each page load
    see: https://dash.plotly.com/live-updates#updates-on-page-load
    """
    return dbc.Container(
        [
            html.Div(print("Loading page..."), id="page-load-trigger", hidden=True),
            dbc.Row(
                [dbc.Col([html.H1("GSF Reports Dashboard")], style={"textAlign": "center"})],
                justify="center",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Stack(
                                [
                                html.P("Date selection:",
                                       style={'padding-top': '15px'}),
                                dcc.DatePickerSingle(
                                    id="dropdown-date-selection",
                                    placeholder="Select a date",
                                )
                            ], direction='horizontal', gap=3, className='float-end')
                        ],
                    style={'border-style': 'none', 'textAlign': 'right'}, width=2),

                    dbc.Col(
                        [
                            dbc.Stack(
                                [
                                html.Div("Report selection:", 
                                         style={'padding-top': '7px'}),
                                dcc.Dropdown(
                                    REPORT_OPTIONS,
                                    id="dropdown-report-selection",
                                    placeholder="Select a report",
                                    clearable=True,
                                    style={'width': '275px', 'height': '47px', 'padding-top': '4px'}
                                )
                                ], direction='horizontal', gap=3, className='float-start'
                            ),

                            dbc.Stack(
                                [
                                    html.Button(
                                        'Download as PDF',
                                        id='download_button',
                                        n_clicks=0, 
                                        hidden=True,
                                        style={'margin-left': '75px', 'margin-top': '12px'}
                                        ),
                                    dcc.Loading(
                                        children=dcc.Download(id='download'),
                                        style={"margin-left": "45px", "margin-top": "12px"},
                                        type='circle',
                                        ),
                                    html.Div(id='hidden-connector-dl', hidden=True)
                                ], direction='horizontal', gap=3
                            )
                        ],
                    style={'border-style': 'none', 'white-space': 'nowrap', }, width=3)
                ], justify='center'
            ),
                   
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Br(),
                            html.Div(id="hidden-connector", hidden=True),
                            dcc.Loading(
                                id="loading-report-1",
                                children=html.Img(id="graph-content"),
                                type="default",
                                fullscreen=False,
                                style={'margin-top': '50px'},
                            ),
                        ],
                        style={"textAlign": "center"},
                    )
                ]
            ),
        ],
        fluid=True,
    )

# initialize app before callbacks to define cache
app = Dash(__name__, title='GSF Reports', update_title=None, external_stylesheets=[dbc.themes.BOOTSTRAP])

# init cache class for storing & retrieving generated reports
cache = FileCache(cache_dir='cache-directory')

# callback to update the available date range from the database
# and detect when a db update has occurred (to clear report cache)
@callback(
    [
        Output("dropdown-date-selection", "min_date_allowed"),
        Output("dropdown-date-selection", "max_date_allowed"),
        Output("dropdown-date-selection", "initial_visible_month"),
        Output("dropdown-date-selection", "date")
    ],
    Input("page-load-trigger", "children"),
)
def update_date_range_options(_):
    print("Getting date options...")
    date_range_options = get_available_date_range(db_engine="gsf_data")
    min_date = date_range_options[0]
    max_date = date_range_options[-1]
    return (min_date, max_date, max_date, max_date)


# callback to clear the report when dropdowns are updated
# this is so that the loading animation will always show right below the dropdowns
# (otherwise it tends to show far down on the page since it seems to center vertically
# based on the last image displayed). Also set the loading_state['is_loading'] property of the graph-content
# obj, so that the Download button isn't displayed until loading is completed (without otherwise needing to pass the 
# entire generated report to the button callbck for keeping track of this, which adds a slight delay for processing)
@callback(
    [Output("graph-content", "src"), 
     Output("hidden-connector", "children"),
     Output('graph-content', 'loading_state')],
    [Input("dropdown-date-selection", "date"),
    Input("dropdown-report-selection", "value")],
    prevent_initial_call=True
)
def init_new_report_output(selected_date, selected_report):
    loading_state = {'is_loading': True}
    return None, [selected_date, selected_report], loading_state


# callback to update the selected report based on dropdowns
@callback(
    [Output("graph-content", "src", allow_duplicate=True),
     Output('graph-content', 'loading_state', allow_duplicate=True)],
    Input("hidden-connector", "children"),
    prevent_initial_call=True,
)
def update_graph(options):

    if options is None:
        return no_update
    elif None in options:
        return no_update

    select_date = to_datetime(options[0])
    select_report = options[1]

    # get and return cached image for report if one exists
    cache_options = f'{select_date} {select_report}'
    cache_data = cache.get_cache_item(cache_options)
    if cache_data is not None:
        print('DISPLAY: Found cached report!')
        cache_data = base64.b64encode(cache_data).decode()
        cache_fig = f"data:image/svg+xml;base64,{cache_data}"
        
        # set loading state for figure before returning
        loading_state = {'is_loading': False}
        return cache_fig, loading_state

    # generate figure depending on report selection
    match select_report:
        case "Scorecard Report":
            fig = PondsOverviewPlots(
                select_date=select_date, run_all=False, save_output=False
            ).plot_scorecard()
        case "EPA Report":
            fig = PondsOverviewPlots(
                select_date=select_date, run_all=False, save_output=False
            ).plot_epa()
        case "Recommended Harvests Report":
            fig = PondsOverviewPlots(
                select_date=select_date, run_all=False, save_output=False
            ).plot_potential_harvests()
        case "Expense Report":
            fig = ExpenseGridReport(report_date=select_date, save_output=False).run()
        case _:
            # Default case, raise Exception
            # since report options are hardcoded, this case should never occur
            raise Exception(
                f'ERROR: report option specified: {select_report}, but no report generation method is defined!'
                )

    # Save it to a temporary buffer.
    buf = BytesIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    # close the figure to release it from memory
    plt.close(fig)
   
    # Embed the result in the html output.
    bytes_data = buf.getbuffer()
    fig_data = base64.b64encode(bytes_data).decode()
    fig_img = f"data:image/svg+xml;base64,{fig_data}"

    # add report to the report cache
    cache.add_cache_item(cache_options, bytes_data)

    # set loading state before returning
    loading_state = {'is_loading': False}

    return fig_img, loading_state

# callback to unhide download button if a report is generated 
@app.callback(Output("download_button", "hidden"),
              [Input('graph-content', 'loading_state'),
               Input("dropdown-date-selection", "date"),
               Input("dropdown-report-selection", "value")],
              prevent_initial_call=True)
def unhide_download_button(loading_state, selected_date, selected_report):

    if loading_state is not None and loading_state['is_loading'] is False and selected_date is not None and selected_report is not None:
        return False # return False to unset 'hidden' property of download button
    else:
        return True

# callback for initial processing of download button, so that loading animation is only triggered when
# the button is clicked, and not when dropdown options are changed
# use 'hidden-connector-dl' as an intermediate "container" for dropdown values to pass to the generate_pdf() 
# function. 
@app.callback([Output("hidden-connector-dl", "children"),
               Output('download_button', 'n_clicks', allow_duplicate=True)],
              [Input('download_button', 'n_clicks'),
               Input("dropdown-date-selection", "date"),
               Input("dropdown-report-selection", "value")],
               prevent_initial_call=True)
def trigger_download(n_clicks, date_selection, report_selection):
    if n_clicks == 0:
        return no_update
    else:
        return [[date_selection, report_selection], 0] # return 0 to reset n_clicks to 0

# callback to download report when download button is clicked
@app.callback(Output('download', 'data'),
              Input("hidden-connector-dl", "children"),
              prevent_initial_call=True)
def generate_pdf(report_options):
    if report_options is None:
        return no_update

    select_date = to_datetime(report_options[0])
    select_report = report_options[1]

    # get and return cached image for report if one exists
    cache_options = f'{select_date} {select_report}'
    cache_data = cache.get_cache_item(cache_options)
    if cache_data is None:
        return no_update
      
    def write_pdf(bytes_io):
        bytes_io.write(cairosvg.svg2pdf(bytestring=cache_data))
    
    return dcc.send_bytes(write_pdf, f'{select_date.strftime("%Y-%m-%d")} {select_report}.pdf')


# on app load, check if the sqlite db file in 'backup' folder (intended for cloud file store) 
# is newer than the current db file in the container
# this is a stupid workaround required due to using a sqlite db rather than something better suited
# for cloud usage...because otherwise the db file wont persist through container restarts
load_sqlite_db()

app.layout = serve_layout

if __name__ == "__main__":
    # run app, host must be '0.0.0.0' to be accessible when deployed via docker
    app.run(debug=False, port=8051, host="0.0.0.0")


def regen_figures_in_cache(num_days:int=15):
    '''
    utility function for calling to update the report figures in cache
    intended use is to call from a scheduled process outside of app
    '''
    import multiprocessing
    def _regen_fig_for_cache(d, r):
        print(f'Generating report in cache for: {r} for date: {d}')
        # delete the cached item first so that it is regenerated when calling update_graph()
        cache.delete_cache_item(f'{d} {r}')
        update_graph((d,r))
    
    dates_range = get_available_date_range(db_engine="gsf_data")[-num_days:]
    # arg_list = []
    for d in dates_range[::-1]:
        # d = d.strftime("%Y-%m-%dT%H:%H:%S")
        for r in REPORT_OPTIONS:
            # arg_list.append((d,r,))
            _regen_fig_for_cache(d,r)
    # pool = multiprocessing.Pool(3)   
    # result = pool.starmap(_regen_fig_for_cache, arg_list)

