#!/bin/bash
now=$(date)
echo "${now}"
#Change to use sudo permission
if [ $(id -u) -ne 0 ]; then exec sudo bash "$0" "$@"; exit; fi
#Let us kill the exist 8080 pid
CURPID=$(netstat -tnlp | grep 8080 | awk '{ print $7 }')
CURPID_SPLIT=($(echo $CURPID | tr "/" "\n"))
echo "Current pid of 8080 : ${CURPID_SPLIT[0]}"

kill -9 ${CURPID_SPLIT[0]}

sleep 1

#Now that we can re-start 8080 pid
npx http-server /home/admin/Data/VAS_TRAINSET -p 8080
echo "http-server 8080 port start"
sleep 1

exit 0
