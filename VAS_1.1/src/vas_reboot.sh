#!/bin/bash

NOW=$(date)

CURPID2=$(ps -ef | grep vas_start.sh | awk '{ print $2 }')
CURPID3=$(ps -ef | grep main_IPcamera_new.py | awk '{ print $2 }')


echo "==========${NOW}=========="
echo "VAS has been rebooted!"

kill -9 ${CURPID2[0]}
kill -9 ${CURPID3[0]}
sleep 1

