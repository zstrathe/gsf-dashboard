#!/bin/bash
currentdatetime=`date +%A,\ %b\ %d,\ %Y\ %I:%M\ %p`
currentdate=`date +%a\ %b\ %-d,\ %Y`
echo '****************************** LOG:' ${currentdatetime} '******************************'
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR; cd ../
echo "Running Python script"
source venv/bin/activate
script_date_str=`date +%Y-%m-%d`
python_output=$(python3 generate_overview.py -i "data_sources/epa_data.xlsx" -d $script_date_str 2>&1)
python_exec_code=$?
echo "$python_output"
if [[ $python_exec_code -eq 0 ]]; then
    output_filename=$(echo "$python_output" | tail -n 1)
    email_subjectline="$(cat automation/email_data/email_subjectline.txt) - ${currentdate}"
    mutt -s "$email_subjectline" -a "$output_filename" -- $(cat automation/email_data/email_addresses.txt) < automation/email_data/email_text.txt
    if [[ $? -eq 0 ]]; then
        echo "Successfully generated and emailed file"
	exit 0
    else
        echo "File generated but failed to send email"
    	exit 1
    fi
else
    echo "Dashboard generation failed for ${currentdatetime}."$'\n\nPython log:\n\n'"${python_output}" | \
    mutt -s "FAILURE: dashboard generation - ${currentdate}" -- $(cat automation/email_data/failure_email_address.txt)
    if [[ $? -eq 0 ]]; then
	echo "Failed generating file but successfully sent failure notification email"
    else
        echo "Failed generating file AND failed to send failure notification email"
    fi
    exit 1
fi




