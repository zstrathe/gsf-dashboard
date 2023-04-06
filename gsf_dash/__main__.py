import sys
import argparse
import traceback
from datetime import datetime
from src import Dataloader, PondsOverviewPlot, load_setting, send_email

def main(argv):   
    
    def failure_notify_email_exit(failure_reason, traceback=None):
        print(failure_reason)
        email_msg_info = load_setting('email_failure_msg')
        send_email(recipients = [x.strip() for x in email_msg_info['recipients'].split(',')], 
                   # split recipients on ',' and remove whitespace because ConfigParser will import it as a single string, but should be a list if more than 1 address
                    subject = f'{email_msg_info["subject"]} - {datetime.strptime(args.date,"%Y-%m-%d").strftime("%a %b %-d, %Y")}', # add date to the end of the email subject
                    msg_body = f'{failure_reason}{f"<br><br>Traceback:<br>{traceback}" if traceback else ""}'
                  )
        sys.exit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("-d","--date", required=True, help="Date to generate report for in 'yyyy-mm-dd' format; year must be from 2020 to 2023")
    #parser.add_argument("-t", "--target_density", type=float, default=0.4, help="OPTIONAL: target density of AFDW (Default = 0.40)")
    args = parser.parse_args()
    
    # check if date argument is valid, use try/except clause with datetime.strptime because it will generate an error with invalid date
    try:
        date_check = datetime.strptime(args.date, '%Y-%m-%d')
    except Exception as ex:
        tb = ''.join(traceback.TracebackException.from_exception(ex).format())
        invalid_date = args.date
        args.date = '9999-01-01' # set to something ridiculous that's still a valid date so the error email will still generate, since date is added to the subject line
        failure_notify_email_exit(f'Invalid date specified: {invalid_date}', tb)
    
    # check if optional target_density argument is valid (between 0 and 1)
    # if args.target_density:
    #     if not (args.target_density > 0 and args.target_density < 1):
    #         failure_notify_email_exit("ERROR: target density (AFDW) should be between 0 and 1")
    
    # load data
    try:
        datadict = Dataloader(args.date).outdata
        scorecard_dataframe = datadict['scorecard_dataframe']
        epa_dict = datadict['epa_dict']
        active_dict = datadict['active_dict']
    except Exception as ex:
        tb = ''.join(traceback.TracebackException.from_exception(ex).format())
        failure_notify_email_exit(f'Error downloading or loading data', tb)
        
    # plot Ponds Overview and get output filename
    try:
        out_filename = PondsOverviewPlot(args.date, scorecard_dataframe, epa_dict, active_dict).out_filename
    except Exception as ex:
        tb = ''.join(traceback.TracebackException.from_exception(ex).format())
        failure_notify_email_exit(f'Error running pond overview script', tb)
        
    print('Emailing message with attachment...')
    email_msg_info = load_setting('email_msg')
    send_email(recipients = [x.strip() for x in email_msg_info['recipients'].split(',')], 
               # split recipients on ',' and remove whitespace because ConfigParser imports as a single string, but needs to be a list of each email string 
                subject = f'{email_msg_info["subject"]} - {datetime.strptime(args.date,"%Y-%m-%d").strftime("%a %b %-d, %Y")}', # add date to the end of the email subject
                msg_body = email_msg_info['body'],
                attachment_path = out_filename) 
    sys.exit(0) # exit with status 0 to indicate successful execution
    
if __name__ == '__main__':   
    main(sys.argv[1:])
