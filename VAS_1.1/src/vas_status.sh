#!/bin/sh
#: Title	: Check restAPI status script
#: Date 	: 2021-11-12
#: Author	: 
#: Version	: 1.0
#: Description  : Checking current running process and re-start rest_start.sh script when it dead.



while [ 1 ]
do
  CNT=`ps -ex|grep main_IPcamera_new.py|grep -v grep | wc -l`
    echo "#################################################" 
    echo "  Checking BCT VAS Learning server status..."
  if [ $CNT > 0 ] # IF NO CONNECTION
    echo "Cannot find current vas connection"
  then
    /home/admin/codes/VAS_1.1/src/vas_start.sh
    echo "Restart vas_start.sh"
  fi
  sleep 10
done
echo "BCT VAS Learning server has been recovered!" 
echo "#################################################"