#!/bin/sh

now=$(date)
echo "${now}"
CURPID1=$(ps -ef | grep vas_status.sh | awk '{ print $2 }')
CURPID2=$(ps -ef | grep vas_start.sh | awk '{ print $2 }')
CURPID3=$(ps -ef | grep main_IPcamera_new.py | awk '{ print $2 }')

kill -9 ${CURPID1[0]}
kill -9 ${CURPID2[0]}
kill -9 ${CURPID3[0]}
echo "Running VAS has been terminated"
