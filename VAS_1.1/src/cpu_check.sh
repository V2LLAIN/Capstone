#!/bin/bash
MEM_THRESHOLD='1024' # 1GB
MEM_FREE=$(free -m | awk '{print $7}')
NOW=$(date)

CURPID2=$(ps -ef | grep vas_start.sh | awk '{ print $2 }')
CURPID3=$(ps -ef | grep main_IPcamera_new.py | awk '{ print $2 }')

if [ $MEM_FREE -lt $MEM_THRESHOLD  ]
    then
        echo "==========${NOW}=========="
        echo "Memory Warning!!!"
        echo "Memory space remaining : ${MEM_FREE} MB"

        kill -9 ${CURPID2[0]}
        kill -9 ${CURPID3[0]}
        sleep 1

    else
        echo "==========${NOW}=========="
        echo "Memory OK!!!"
        echo "Memory space remaining : ${MEM_FREE} MB"
fi