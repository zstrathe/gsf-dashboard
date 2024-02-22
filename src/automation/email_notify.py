import sys
from io import BytesIO
import traceback
from datetime import datetime
from matplotlib import pyplot as plt

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# add .packages/ directory to sys.path, so that other relative modules can be imported
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR)) 

from dataloader import Dataloader
from utils.utils import load_setting
from o365_connect import EmailHandler
from report_generation import PondsOverviewPlots, ExpenseGridReport


class RunWithEmailNotification:
    def __init__(self, date:datetime, run_dataloader:bool, email_reports:bool, test_run:bool):
        self.date = date
        
        if test_run:
            print('''*`*`*`*`*` RUNNING AS TEST '*`*`*`*`*`''')

        if run_dataloader:
            self._run_dataloader()
        
        if email_reports:
            if test_run:
                self._email_reports(test_run=True)
            else:
                self._email_reports(test_run=False)


    def _failure_notify_email_exit(self, failure_reason, err_traceback:str|None=None):
        print(failure_reason)
        email_msg_info = load_setting("email_failure_msg")
        EmailHandler().send_email(
            recipients=email_msg_info["recipients"],
            # add date to the end of the email subject
            subject=f'{email_msg_info["subject"]} - {datetime.strptime(self.date,"%Y-%m-%d").strftime("%a %b %-d, %Y")}',
            msg_body=f'{failure_reason}{f"<br><br>Traceback:<br>{err_traceback}" if err_traceback else ""}',
        )
        raise Exception("Stopped process due to error")
        
    def _run_dataloader(self):
        try:
            Dataloader(run_date=self.date)
        except Exception as ex:
            tb = "".join(traceback.TracebackException.from_exception(ex).format())
            self._failure_notify_email_exit(f"Error with loading daily data into db!", tb)


    def _email_reports(self, test_run:bool):

        report_callables = [
            {'name': f'Pond Health Overview', 'callable':
                PondsOverviewPlots(
                    select_date=self.date, run_all=False, save_output=False
                ).plot_scorecard()
            },
            
            {'name': f'EPA Report', 'callable':
                PondsOverviewPlots(
                    select_date=self.date, run_all=False, save_output=False
                ).plot_epa()
            },
            
            {'name': 'Recommended Harvests', 'callable':
                PondsOverviewPlots(
                select_date=self.date, run_all=False, save_output=False
                ).plot_potential_harvests()
            },
            
            {'name': 'Expense Report', 'callable':
                ExpenseGridReport(report_date=self.date, save_output=False).run()
            }]

        # gather reports in memory 
        report_attachments = []
        for report in report_callables:
            try:
                # generate the figure by calling the callable for each report
                fig = report['callable']

                # init a BytesIO memory obj to save report to
                rep_buffer = BytesIO()
                fig.savefig(rep_buffer, format='pdf', bbox_inches='tight')
                
                # add to the list of attachments
                # the O365 module message.add() function can take a list or tuple of:
                # (file_like_object (instance of BytesIO), name of file attachment (string))
                report_attachments.append((rep_buffer, f"{report['name']} {self.date.strftime('%Y-%m-%d')}.pdf"))

                # close the figure to release it from memory
                plt.close(fig)

            except Exception as ex:
                tb = "".join(traceback.TracebackException.from_exception(ex).format())
                self._failure_notify_email_exit(f"Error generating report for: {report['name']}!", tb)

        print("Emailing message with attachments...")

        if test_run:
            email_msg_info = load_setting("test_msg")
        else:
            email_msg_info = load_setting("email_msg")

        EmailHandler().send_email(
            # split recipients on ',' and remove whitespace because ConfigParser imports as a single string, but needs to be a list of each email string
            recipients=email_msg_info["recipients"],
            # add date to the end of the email subject
            subject=f'{email_msg_info["subject"]} - {self.date.strftime("%a %b %-d, %Y")}',
            msg_body=email_msg_info["body"],
            attachments=report_attachments,
        )