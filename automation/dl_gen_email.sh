#!/bin/bash
currentdatetime=`date +%A,\ %b\ %d,\ %Y\ %I:%M\ %p`
echo '****************************** LOG:' ${currentdatetime} '******************************'
while [ true ]
do
    SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
    cd $SCRIPT_DIR; cd ../
    curl_download="$(cat automation/data_curl_download.txt)"' -w %{http_code} --output data_sources/datafile.xlsx'
    http_status=$(eval "$curl_download")
    connection_status=$?
    sleep 10
    if [[ $http_status -eq 200 ]] && [[ $connection_status -eq 0 ]]; then
	userid=$(id -u)
	DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/$userid/bus"
	export DBUS_SESSION_BUS_ADDRESS
	export DISPLAY=:0
	data_filename="data_sources/datafile.xlsx"
	source venv/bin/activate
        script_date_str=`date +%Y-%m-%d`
        python_output=$(python3 generate_overview.py -i $data_filename -d $script_date_str 2>&1)
	python_exec_code=$?
	echo "$python_output"
	if [[ $python_exec_code -eq 0 ]]; then
		currentdate=`date +%A\ %b\ %d,\ %Y`
		output_filename=$(echo "$python_output" | tail -n 1)
		email_subjectline="$(cat automation/email_data/email_subjectline.txt) - ${currentdate}"
		mutt -s "$email_subjectline" -a "$output_filename" -- $(cat automation/email_data/email_addresses.txt) < automation/email_data/email_text.txt
		echo Success
		/usr/bin/notify-send -u critical "SUCCESS" "Daily pond overview generated for ${currentdatetime}"
	else
		echo Failure
		/usr/bin/notify-send -u critical "FAILURE" "Daily pond overview failed for ${currentdatetime}"
		echo "Dashboard generation failed for ${currentdatetime}."$'\n\nPython log:\n\n'"${python_output}" | \
		mutt -s "FAILURE: dashboard generation" -- $(cat automation/email_data/failure_email_address.txt)
	fi
	exit 0
    fi
done



