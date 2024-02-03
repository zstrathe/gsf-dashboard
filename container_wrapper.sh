#!/bin/bash

# Start the first process
python3 ./src/dataloader/scheduler.py &

# Start the second process
python3 ./src/dash_app/app.py &

# Wait for any process to exit
wait -n

# Exit with status of process that exited first
exit $?