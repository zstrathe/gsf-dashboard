import sys
import argparse
import traceback
from datetime import datetime
from src import Dataloader, PondsOverviewPlots, load_setting, send_email
import functools
# set print function to always flush buffer for running in terminal (i.e., running from a bash script) for ease of realtime monitoring
print = functools.partial(print, flush=True)

def main(argv):   
    
    def failure_notify_email_exit(failure_reason, traceback=None):
        print(failure_reason)
        email_msg_info = load_setting('email_failure_msg')
        send_email(recipients = email_msg_info['recipients'], 
                    subject = f'{email_msg_info["subject"]} - {datetime.strptime(args.date,"%Y-%m-%d").strftime("%a %b %-d, %Y")}', # add date to the end of the email subject
                    msg_body = f'{failure_reason}{f"<br><br>Traceback:<br>{traceback}" if traceback else ""}'
                  )
        sys.exit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("-d","--date", required=True, help="Date to generate report for in 'yyyy-mm-dd' format; year must be from 2020 to 2023")
    parser.add_argument("-t","--test_run", type=bool, default=False, help="OPTIONAL: set to True to run as a test to send to alternate email")
    #parser.add_argument("-t", "--target_density", type=float, default=0.4, help="OPTIONAL: target density of AFDW (Default = 0.40)")
    args = parser.parse_args()
    
    if args.test_run:
        print('RUNNING AS TEST')
    
    # check if date argument is valid, use try/except clause with datetime.strptime because it will generate an error if an invalid date is provided
    try:
        # store date as a datetime object 
        ### TODO convert all functions to use datetime, only parse string in this main function!!
        date_dt = datetime.strptime(args.date, '%Y-%m-%d')
    except Exception as ex:
        tb = ''.join(traceback.TracebackException.from_exception(ex).format())
        invalid_date = args.date
        args.date = '9999-01-01' # set to something ridiculous that's still a valid date so the error email will still generate, since date is added to the subject line
        failure_notify_email_exit(f'Invalid date specified: {invalid_date}', tb)
    
    # check if optional target_density argument is valid (between 0 and 1)
    # if args.target_density:
    #     if not (args.target_density > 0 and args.target_density < 1):
    #         failure_notify_email_exit("ERROR: target density (AFDW) should be between 0 and 1")
    
    '''
    LOAD DATA
    '''
    try:
        datadict = Dataloader(args.date).outdata
    except Exception as ex:
        tb = ''.join(traceback.TracebackException.from_exception(ex).format())
        failure_notify_email_exit(f'Error downloading or loading data', tb)

    # TEMPORARY INTEGRATION OF NEW DATABASE LOADING METHODS
    try:
        from src.dataloader_NEW import Dataloader as Dataloader_NEW
        db_dataloader = Dataloader_NEW(args.date) # NEED TO UPDATE TO USE datetime as date argument!!!
        db_dataloader.load_daily_data_prev_n_days(prev_num_days_to_load=5, specify_date=date_dt)
    except Exception as ex:
        tb = ''.join(traceback.TracebackException.from_exception(ex).format())
        failure_notify_email_exit(f'Error with loading daily data into db!', tb)
    
    '''
    GENERATE PLOTS
     - returns output filenames
    '''
    try:
        output_filenames = PondsOverviewPlots(args.date, **datadict).output_filenames
    except Exception as ex:
        tb = ''.join(traceback.TracebackException.from_exception(ex).format())
        failure_notify_email_exit(f'Error running pond overview script', tb)

    # TEMPORARY INTEGRATION OF NEW PLOT OF EXPENSES USING MORE MODULAR CLASS METHODS
    try:
        from src.generate_overview_abstractclasses import ExpenseReport
        output_filenames.insert(1, ExpenseReport(report_date=date_dt).run())
    except Exception as ex:
        tb = ''.join(traceback.TracebackException.from_exception(ex).format())
        failure_notify_email_exit(f'Error running expense report!', tb)
    
    print('Emailing message with attachment...')
    
    if args.test_run == False:
        email_msg_info = load_setting('email_msg')
    else:
        email_msg_info = load_setting('test_msg')
        
    send_email(recipients = email_msg_info['recipients'], 
               # split recipients on ',' and remove whitespace because ConfigParser imports as a single string, but needs to be a list of each email string 
                subject = f'{email_msg_info["subject"]} - {datetime.strptime(args.date,"%Y-%m-%d").strftime("%a %b %-d, %Y")}', # add date to the end of the email subject
                msg_body = email_msg_info['body'],
                attachments = output_filenames) 
    sys.exit(0) # exit with status 0 to indicate successful execution
    
if __name__ == '__main__':   
    main(sys.argv[1:])
