#!/bin/bash
currentdatetime=`date +%A,\ %b\ %d,\ %Y\ %I:%M\ %p`
currentdate=`date +%a\ %b\ %-d,\ %Y`
echo '****************************** LOG:' ${currentdatetime} '******************************'
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR; cd ../
echo "Running Python script"
source venv/bin/activate
script_date_str=`date +%Y-%m-%d`
python_output=$(python3 -m prod.run -d $script_date_str 2>&1)
python_exec_code=$?
echo "$python_output"
if [[ $python_exec_code -eq 0 ]]; then
    echo "Python script ran successfully"
    exit 0
else
    echo "Python script failed"
    exit 1
fi




