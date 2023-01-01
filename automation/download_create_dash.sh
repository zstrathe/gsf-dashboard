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
	cat output_data.tmp
	if [[ $python_exec_code -eq 0 ]]; then
		currentdate=`date +%A\ %b\ %d,\ %Y`
		output_filename=$(tail -n 1 output_data.tmp)
		email_addresses=$(cat email_data/email_addresses.txt)
		email_subjectline="$(cat email_data/email_subjectline.txt) - ${currentdate}"
		mutt -s "$email_subjectline" -a "$output_filename" -- $email_addresses < email_data/email_text.txt
		echo Success
		/usr/bin/notify-send -u critical "SUCCESS" "Daily pond overview generated for ${currentdatetime}"
	else
		echo Failure
		/usr/bin/notify-send -u critical "FAILURE" "Daily pond overview failed for ${currentdatetime}"
	fi
	rm output_data.tmp
	exit 0
    fi
done



