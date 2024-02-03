import schedule
import time
from datetime import datetime
from dataloader import Dataloader

# add .packages/ directory to sys.path, so that other relative modules can be imported
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR)) 

def job():
    today = datetime.today()
    # run an instance of Dataloader class
    Dataloader(run_date=today)

schedule.every().monday.at("20:00").do(job)
schedule.every().tuesday.at("20:00").do(job)
schedule.every().wednesday.at("20:00").do(job)
schedule.every().thursday.at("20:00").do(job)
schedule.every().friday.at("20:00").do(job)

while True:
    schedule.run_pending()
    time.sleep(1)