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

def generate_multipage_pdf(fig_list, pdf_filename):
    with PdfPages(pdf_filename) as pdf:
        for fig in fig_list:
            plt.figure(fig)
            pdf.savefig(bbox_inches='tight')  # saves the current figure into a pdf page
            plt.close()
    return pdf_filename
        