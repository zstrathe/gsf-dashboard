import schedule
import time
from datetime import datetime

# add .packages/ directory to sys.path, so that other relative modules can be imported
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR)) 

from dataloader import Dataloader
from dash_app.app import regen_figures_in_cache
from .email_notify import RunWithEmailNotification

def catch_exceptions(cancel_on_failure=False):
    ''' decorator from: 
    https://schedule.readthedocs.io/en/stable/exception-handling.html '''
    import functools
    def catch_exceptions_decorator(job_func):
        @functools.wraps(job_func)
        def wrapper(*args, **kwargs):
            try:
                return job_func(*args, **kwargs)
            except:
                import traceback
                print(traceback.format_exc())
                if cancel_on_failure:
                    return schedule.CancelJob
        return wrapper
    return catch_exceptions_decorator

@catch_exceptions(cancel_on_failure=True)
def job():
    today = datetime.today()
    
    # run an instance of Dataloader class and generate/email reports
    #Dataloader(run_date=today)
    RunWithEmailNotification(date=today, run_dataloader=True, email_reports=True, test_run=True)
    
    # regenerate the reports in the cache for the dash app
    # this is to ensure that if any data has been updated, then the reports will reflect that
    regen_figures_in_cache()

schedule.every().monday.at("20:00").do(job)
schedule.every().tuesday.at("20:00").do(job)
schedule.every().wednesday.at("20:00").do(job)
schedule.every().thursday.at("20:00").do(job)
schedule.every().friday.at("20:00").do(job)

while True:
    schedule.run_pending()
    time.sleep(1)