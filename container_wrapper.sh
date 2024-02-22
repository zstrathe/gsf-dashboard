#!/bin/bash

# Start the first process
python3 ./src/automation/scheduler.py &

# Start the second process
gunicorn --bind 0.0.0.0:8051 wsgi:application &

# Wait for any process to exit
wait -n

# Exit with status of process that exited first
exit $?
