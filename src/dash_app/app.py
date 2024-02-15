import base64
from io import BytesIO
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from dash import Dash, html, dcc, callback, Output, Input
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
                [dbc.Col([html.H1("GSF Reports")], style={"textAlign": "center"})],
                justify="center",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.P("Date selection:      "),
                        ],
                        style={"textAlign": "right", "padding-top": 13},
                        width={"size": 2},
                    ),
                    dbc.Col(
                        [
                            dcc.DatePickerSingle(
                                id="dropdown-date-selection",
                                placeholder="Select a date",
                            )
                        ],
                        style={"textAlign": "left"},
                        width={"size": 2},
                    ),
                ],
                justify="center",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.P("Report selection:      "),
                        ],
                        style={"textAlign": "right", "padding-top": 5},
                        width=2,
                    ),
                    dbc.Col(
                        [
                            dcc.Dropdown(
                                REPORT_OPTIONS,
                                id="dropdown-report-selection",
                                placeholder="Select a report",
                                clearable=True,
                            )
                        ],
                        style={"textAlign": "left"},
                        width=2,
                    ),
                ],
                justify="center",
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
                                style={"textAlign": "top"},
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
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = 'GSF Daily Reports'
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
# based on the last image displayed).
@callback(
    [Output("graph-content", "src"), 
     Output("hidden-connector", "children")],
    [Input("dropdown-date-selection", "date"),
    Input("dropdown-report-selection", "value")],
    prevent_initial_call=True
)
def clear_report_output_when_generating_new(selected_date, selected_report):
    return None, [selected_date, selected_report]


# callback to update the selected report based on dropdowns
@callback(
    Output("graph-content", "src", allow_duplicate=True),
    Input("hidden-connector", "children"),
    prevent_initial_call=True,
)
def update_graph(options):

    if options is None:
        return None

    select_date = to_datetime(options[0])
    select_report = options[1]

    # get and return cached image for report if one exists
    cache_options = f'{select_date} {select_report}'
    #cache_fig = cache.get(cache_options)
    cache_data = cache.get_cache_item(cache_options)
    if cache_data is not None:
        print('Found cached report!')
        cache_data = cache_data.decode("ascii")
        cache_fig = f"data:image/png;base64,{cache_data}"
        return cache_fig

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
            # Default case, return nothing if no report option selected
            return None

    # Save it to a temporary buffer.
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    # Embed the result in the html output.
    bytes_data = base64.b64encode(buf.getbuffer())
    fig_data = bytes_data.decode("ascii")
    fig_img = f"data:image/png;base64,{fig_data}"
    # close the figure to release it from memory
    plt.close(fig)

    # add report to the report cache
    cache.add_cache_item(cache_options, bytes_data)

    return fig_img


# on app load, check if the sqlite db file in 'backup' folder (intended for cloud file store) 
# is different from the current db file in the container
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

