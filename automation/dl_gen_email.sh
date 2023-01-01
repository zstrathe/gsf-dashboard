#!/bin/bash
while [ true ]
do 
    currentdatetime=`date +%A,\ %b\ %d,\ %Y\ %I:%M\ %p`
    echo '****************************** LOG:' ${currentdatetime} '******************************'
    cd /home/zack/Documents/Python-Projects/dashboard
    http_status=$(bash ./automation/scorecard_curl_download.txt)
    sleep 10
    if [[ $http_status -eq 200 ]]; then
	userid=$(id -u)
	DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/$userid/bus"
	export DBUS_SESSION_BUS_ADDRESS
	export DISPLAY=:0
	filename="./data-sources/dailyscorecard.xlsx"
	source /home/zack/Documents/Python-Projects/dashboard/venv/bin/activate
        printf -v date_str '%(%Y-%m-%d)T\n' -1
        python3 generate_overview.py -i $filename -d $date_str > output_data.tmp
	python_exec_code=$?
	python_output=$(cat output_data.tmp)
	rm output_data.tmp
	echo "$python_output"
	if [[ $python_exec_code -eq 0 ]]; then
		currentdate=`date +%A\ %b\ %d,\ %Y`
		output_filename=$(echo "$python_output" | tail -n 1)
		email_subjectline="$(cat email_data/email_subjectline.txt) - ${currentdate}"
		mutt -s "$email_subjectline" -a "$output_filename" -- $(cat email_data/email_addresses.txt) < email_data/email_text.txt
		echo Success
		/usr/bin/notify-send -u critical "SUCCESS" "Daily pond overview generated for ${currentdatetime}"
	else
		echo Failure
		/usr/bin/notify-send -u critical "FAILURE" "Daily pond overview failed for ${currentdatetime}"
		echo "Dashboard generation failed for ${currentdatetime}."$'\nPython log:\n'"${python_output}" > failure_email_text.tmp
		mutt -s "FAILURE: dashboard generation" -- $(cat email_data/failure_email_address.txt) < failure_email_text.tmp
		rm failure_email_text.tmp 
	fi
	exit 0
    fi
done



