import schedule
import time
from dataloader import Dataloader

# add .packages/ directory to sys.path, so that other relative modules can be imported
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR)) 

def job():
    # run an instance of Dataloader class
    Dataloader(run_automated=True)

schedule.every().day.at("20:00").do(job)

while True:
    schedule.run_pending()
    time.sleep(1)