from configparser import ConfigParser
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

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

def generate_multipage_pdf(fig_list, pdf_filename, add_pagenum=True, bbox_inches='tight'):
    with PdfPages(pdf_filename) as pdf:
        for idx, fig in enumerate(fig_list, start=1):
            plt.figure(fig)
            if add_pagenum:
                fig.text(0.5, 0.0275, f'Page {idx} of {len(fig_list)}', ha='center', va='bottom', fontsize='small') 
            pdf.savefig(bbox_inches=bbox_inches)  # saves the current figure into a pdf page
            plt.close()
    return pdf_filename
    
def redirect_logging_to_file(log_file_directory: Path, log_file_name: str):
    ''' Decorator to temporarily redirect stdout to a file '''
    def decorator(function):
        def func_wrapper(*args, **kwargs):
            log_file = log_file_directory / log_file_name
            log_file_path = log_file.as_posix() # convert log_file into a string of the file path
            print(f'Redirecting stdout log to: {log_file_path}...')
            from contextlib import redirect_stdout
            with open(log_file_path, "w") as f:
                with redirect_stdout(f):
                    output = function(*args, **kwargs)
            print('Stdout log saved...')
            return output
        return func_wrapper
    return decorator