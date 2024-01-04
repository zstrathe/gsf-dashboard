from configparser import ConfigParser
from O365 import Account, FileSystemTokenBackend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import email
from datetime import datetime
from pathlib import Path
import sys

# load auth credentials & settings from settings.cfg file
def load_setting(specified_setting):
    cp = ConfigParser()
    cp.read('./settings/settings.cfg')
    setting = dict(cp.items(specified_setting))
    # iterate through the returned setting to look for any keys that end in '.dtype'
    # these key/value pairs specify the data type that the specific field (in 'field.dtype') should be forced to
    for idx, (key, val) in enumerate(setting.copy().items()):
        if '.dtype' in key:
            enforced_setting = key.split('.dtype')[0]
            enforced_dtype = val
            enforced_value = setting[enforced_setting]
            # case statement to determine transformation of data depending on specified dtype
            match enforced_dtype:
                case 'list':
                    enforced_value = [x.strip() for x in enforced_value.split(',')]
                case 'int':
                    enforced_value = int(enforced_value)
                case 'float':
                    enforced_value = float(enforced_value)
                case _: 
                    print(f'ERROR: Could not enforce dtype: {enforced_dtype} for setting: {enforced_setting}')        
            setting[enforced_setting] = enforced_value
            del setting[key] # delete the setting entry for the .dtype specifier
    return setting 

# def send_email(recipients, subject, msg_body, attachments=[]):
#     email_settings = load_setting('email_cred')
#     credentials = (email_settings['client_id'], email_settings['client_secret'])
#     tenant = email_settings['tenant']
#     token_backend = FileSystemTokenBackend(token_path='./settings/auth_data/', token_filename='outlook_auth_token.txt') # save token for email auth to re-use 
#     account = Account(credentials, auth_flow_type='authorization', tenant_id=tenant, token_backend=token_backend)
#     if not account.is_authenticated:  # will check if there is a token and has not expired
#         # ask for a login 
#         account.authenticate(scopes=['basic', 'message_all'])
#     message = account.mailbox().new_message()
#     message.to.add(recipients) 
#     message.subject = subject
#     message.body = msg_body
#     if attachments:
#         for attachment_path in attachments:
#             message.attachments.add(attachment_path)
#     message.send()
#     print('Email successfully sent!')

def generate_multipage_pdf(fig_list, pdf_filename, add_pagenum=True, bbox_inches='tight'):
    with PdfPages(pdf_filename) as pdf:
        for idx, fig in enumerate(fig_list, start=1):
            plt.figure(fig)
            if add_pagenum:
                fig.text(0.5, 0.0275, f'Page {idx} of {len(fig_list)}', ha='center', va='bottom', fontsize='small') 
            pdf.savefig(bbox_inches=bbox_inches)  # saves the current figure into a pdf page
            plt.close()
    return pdf_filename
    
''' Decorator to temporarily redirect stdout to a file '''
def redirect_logging_to_file(log_file_directory: Path, log_file_name: str):
    def decorator(function):
        def func_wrapper(*args, **kwargs):
            log_file = log_file_directory / log_file_name
            log_file_path = log_file.as_posix() # convert log_file into a string of the file path
            print(f'Redirecting stdout log to: {log_file_path}...')
            from contextlib import redirect_stdout
            with open(log_file_path, "w") as f:
                with redirect_stdout(f):
                    output = function(*args, **kwargs)
            print(f'Stdout log saved...')
            return output
        return func_wrapper
    return decorator

# class EmailHandler:
#     def __init__(self):
#         self.account = self.authenticate()
        
#         # set permission list of file extensions to look for when extracting files from emails
#         # currently only looking for excel files
#         self.good_attachment_types = ['xlsx', 'xls', 'xlsm']
        
#     def _auth_func(account, scopes):
#         account.authenticate(scopes)
    
#     def authenticate(self):
#         def _auth_func(account):
#             account.authenticate(scopes=['basic', 'message_all'])
        
#         email_settings = load_setting('email_cred')
#         credentials, tenant = (email_settings['client_id'], email_settings['client_secret']), email_settings['tenant']
#         token_backend = FileSystemTokenBackend(token_path='./settings/auth_data/', token_filename='outlook_auth_token.txt') # save token for email auth to re-use 
#         account = Account(credentials, auth_flow_type='authorization', tenant_id=tenant, token_backend=token_backend)
#         if not account.is_authenticated:  # will check if there is a token and has not expired
#             # ask for a login 
#             account.authenticate(scopes=['basic', 'message_all'])
#             #_auth_func(account)
#         print('Authenticated email account successfully!')
#         return account
    
#     def send_email(self, recipients, subject, msg_body, attachments=[]):
#         message = self.account.mailbox().new_message()
#         message.to.add(recipients) 
#         message.subject = subject
#         message.body = msg_body
#         if attachments:
#             for attachment_path in attachments:
#                 message.attachments.add(attachment_path)
#         message.send()
#         print('Email sent successfully!')
  
#     def get_latest_email_attachment_from_folder(self, folder_name, save_filename, dl_attachment=True):
#         '''
#         This method retrieves the most-recent email attachment from a specified email folder. If the email contains an attachment, then 
#         if the attachment is on the permission list (self.good_attachment_types), then it is downloaded to ./data_sources/ and the file path is then returned.
        
#         If the attachment is an email file (and attachment_type will be a 'file'), then self.extract_attachemt_from_email_file will be called, 
#         and the first valid attachment will be saved to ./data_sources/ and its file path returned.
        
#         NOTE: a potential shortcoming of this method is that it will return only the first valid attachment of the most-recent email. Therefore it must be modified if there 
#         is ever a need to extract multiple attachments from an email, or needs to look through multiple emails, etc
#         '''
#         fldr = self.account.mailbox().get_folder(folder_name=folder_name)
#         latest_message = fldr.get_message(query="", download_attachments=True) 
#         attachments = latest_message.attachments
#         if len(attachments) > 0:
#             for a in attachments:
#                 a_type = a.attachment_type
#                 # check if the attachment type is an 'item' (aka hopefully a .eml file) or a 'file' (aka hopefully an excel file, but could be a pdf or anything else)
#                 # there is potential for error, where an 'item' is a different attachment such as a meeting invite but self.extract_attachment_from_email_file should 
#                 # still be able to extract files attached to those
#                 if a_type == 'item':
#                     print('Email (.eml) attachment found...', end=' ', flush=True)
#                     print(f'Saving as {a.name}.eml')
#                     tmpfile_path = f'data_sources/tmp/{a.name}.eml'
#                     latest_message.attachments.save_as_eml(a, to_path=tmpfile_path)
#                     return self.extract_attachment_from_email_file(tmpfile_path, save_filename)
#                 elif a_type == 'file':
#                     print('Non-email attachment found')
#                     print(a.name)
#                     if a.name.split('.')[-1] in self.good_attachment_types:
#                         print('GOOD ATTACHMENT TYPE!')
#                         a.save(location=f'data_sources/', custom_name=save_filename)
#                         return f'data_sources/{a.name}'
#                     else:
#                         print('BAD ATTACHMENT TYPE! Skipping...')
#                         pass
                        
#         else:
#             print('Did not find any file attachments!')
            
#     def extract_attachment_from_email_file(self, email_filename, save_filename):
#         msg = email.message_from_file(open(email_filename))
#         attachments=msg.get_payload()
#         for attachment in attachments:
#             fnam = attachment.get_filename()
#             f_extension = fnam.split('.')[-1]
#             if fnam != None and f_extension in self.good_attachment_types: 
#                 print(f'Saving {fnam}...as {save_filename}.{f_extension}')
#                 save_fnam= f'data_sources/{save_filename}.{f_extension}'
#                 with open(save_fnam, 'wb') as f:
#                     f.write(attachment.get_payload(decode=True))
#                 print(f'Successfully saved to {save_fnam}!')
#                 return save_fnam
#             else:
#                 print(f'Skipping {fnam}, not a good filetype!')