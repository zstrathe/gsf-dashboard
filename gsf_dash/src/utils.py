from configparser import ConfigParser
from O365 import Account, FileSystemTokenBackend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# load auth credentials & settings from settings.cfg file
def load_setting(specified_setting):
    cp = ConfigParser()
    cp.read('./settings/settings.cfg')
    return dict(cp.items(specified_setting))  

def send_email(recipients, subject, msg_body, attachments=[]):
    email_settings = load_setting('email_cred')
    credentials = (email_settings['client_id'], email_settings['client_secret'])
    tenant = email_settings['tenant']
    token_backend = FileSystemTokenBackend(token_path='./settings/auth_data/', token_filename='outlook_auth_token.txt') # save token for email auth to re-use 
    account = Account(credentials, auth_flow_type='authorization', tenant_id=tenant, token_backend=token_backend)
    if not account.is_authenticated:  # will check if there is a token and has not expired
        # ask for a login 
        account.authenticate(scopes=['basic', 'message_all'])
    message = account.mailbox().new_message()
    message.to.add(recipients) 
    message.subject = subject
    message.body = msg_body
    if attachments:
        for attachment_path in attachments:
            message.attachments.add(attachment_path)
    message.send()
    print('Email successfully sent!')

def generate_multipage_pdf(fig_list, pdf_filename, add_pagenum=True, bbox_inches='tight'):
    with PdfPages(pdf_filename) as pdf:
        for idx, fig in enumerate(fig_list, start=1):
            plt.figure(fig)
            if add_pagenum:
                fig.text(0.5, 0.0275, f'Page {idx} of {len(fig_list)}', ha='center', va='bottom', fontsize='small') 
            pdf.savefig(bbox_inches=bbox_inches)  # saves the current figure into a pdf page
            plt.close()
    return pdf_filename
        
def open_email_attachment():
    email_settings = load_setting('email_cred')
    credentials, tenant = (email_settings['client_id'], email_settings['client_secret']), email_settings['tenant']
    token_backend = FileSystemTokenBackend(token_path='./settings/auth_data/', token_filename='outlook_auth_token.txt') # save token for email auth to re-use 
    account = Account(credentials, auth_flow_type='authorization', tenant_id=tenant, token_backend=token_backend)
    if not account.is_authenticated:  # will check if there is a token and has not expired
        # ask for a login 
        account.authenticate(scopes=['basic', 'message_all'])
    data_folder = account.mailbox().get_folder(folder_name='Smartflow')   
    message = account.mailbox().new_message()
    message.to.add(recipients) 
    message.subject = subject
    message.body = msg_body
    if attachments:
        for attachment_path in attachments:
            message.attachments.add(attachment_path)
    message.send()
    print('Email successfully sent!')
    
class EmailHandler:
    def __init__(self):
        self.account = self.authenticate()
        
    def _auth_func(account, scopes):
        account.authenticate(scopes)
    
    def authenticate(self):
        @Decorators.timeout()
        def _auth_func(account):
            account.authenticate(scopes=['basic', 'message_all'])
        
        email_settings = load_setting('email_cred')
        credentials, tenant = (email_settings['client_id'], email_settings['client_secret']), email_settings['tenant']
        token_backend = FileSystemTokenBackend(token_path='./settings/auth_data/', token_filename='outlook_auth_token.txt') # save token for email auth to re-use 
        account = Account(credentials, auth_flow_type='authorization', tenant_id=tenant, token_backend=token_backend)
        if not account.is_authenticated:  # will check if there is a token and has not expired
            # ask for a login 
            ## TODO add a timer to this in case login is called for when running automated from command line (so script will stop execution)
            account.authenticate(scopes=['basic', 'message_all'])
            #_auth_func(account)
        print('Authenticated email account successfully!')
        return account
    
    def send_email(self, recipients, subject, msg_body, attachments=[]):
        message = self.account.mailbox().new_message()
        message.to.add(recipients) 
        message.subject = subject
        message.body = msg_body
        if attachments:
            for attachment_path in attachments:
                message.attachments.add(attachment_path)
        message.send()
        print('Email sent successfully!')

    def extract_attachment_from_email_file(self, email_filename):
        import email
        good_attachment_types = ['xlsx', 'xls', 'xlsm']
        
        msg = email.message_from_file(open(email_filename))
        attachments=msg.get_payload()
        for attachment in attachments:
            fnam = attachment.get_filename()
            if fnam != None and fnam.split('.')[-1] in good_attachment_types: 
                print(f'Saving {fnam}...')
                save_fnam= f'data_sources/{fnam}'
                print(save_fnam, 'test')
                try:
                    # save_fnam= f'data_sources/{f_nam}'
                    # print(save_fnam, 'test')
                    f=open(save_fnam, 'wb').write(attachment.get_payload(decode=True,))
                    f.close()
                except Exception as detail:
                    #print detail
                    pass
            else:
                print(f'Skipping {fnam}, not a good filetype!')
        return save_fnam
        
    def get_latest_email_attachment_from_folder(self, folder_name):
        from datetime import datetime
        data_folder = self.account.mailbox().get_folder(folder_name='SFData')
        latest_message = data_folder.get_message(query="", download_attachments=True)
        print('Sent date:', latest_message.sent.date())
        print('Has attachments:', latest_message.has_attachments)
        attachments = latest_message.attachments
        good_attachment_types = ['xlsx', 'xls', 'xlsm']
        
        for a in attachments:
            print(dir(a))
            # check if the attachment has a filename extension, len(name.split('.') should equal > 1 if it has one.
            # if the attachment has no filename extension, len(name.split('.') should equal exactly 1.
            # assuming that an attachment with no filename, then it is an email '.eml' filetype
            if len(a.name.split('.')) == 1: # if the attachment is an email
                print('Email attachment found!', end=' ', flush=True)
                print(f'Saving as {a.name}.eml')
                tmpfile_path = f'data_sources/tmp/{a.name}.eml'
                latest_message.attachments.save_as_eml(a, to_path=tmpfile_path)
                filename = self.extract_attachment_from_email_file(tmpfile_path)
            elif len(a.name.split('.')) > 1:
                print('Non-email attachment found')
                print(a.name)
                if a.name.split('.')[-1] in good_attachment_types:
                    print('GOOD ATTACHMENT TYPE!')
                    a.save(location=f'data_sources/')
                else:
                    print('BAD ATTACHMENT TYPE! Skipping...')

class Decorators:
    def __init__(self):
        pass
   
    def timeout(seconds=15):
        import _thread
        import threading
        import sys
        '''
        The intention with this timeout decorator is to utilize with a function to monitor and stop the program execution 
        if the timeout seconds expire. Specificially, the intention is to use it for the m365 email authentication method, which requires
        user input to initially setup an auth token. Since that auth process could cause an automated process to hang if it's called, 
        then ideally that process should be stopped after a specified timeout interval.
        Unfortunately this doesn't seem to be working as intended...
        
        Utilizing solution from: https://stackoverflow.com/a/31667005 (modified slightly)
        use as decorator to exit process if 
        function takes longer than s seconds
        '''
        def quit_function(fn_name):
            print(f'ERROR: process ({fn_name}) timed out! ({seconds} seconds)', flush=True)
            #sys.exit()
            _thread.interrupt_main() # raises KeyboardInterrupt
        
        def outer(fn): # outer AKA "decorator"
            # the "wrapper" function which calls the target function and a separate timed thread,
            # the timer will run for the specified number of seconds, before calling the quit_function which will / 
            # raise a KeyboardInterrupt on the main python thread 
            def inner(*args, **kwargs): # inner AKA "wrapper"
                timer = threading.Timer(seconds, quit_function, args=[fn.__name__])
                print('timer daemon', timer.daemon)
                timer.daemon = True
                timer.start()
                print('timer daemon', timer.daemon)
                try:
                    result = fn(*args, **kwargs)
                finally:
                    timer.cancel()
                return result
            return inner
        return outer
