#! /bin/bash
### http-server
gnome-terminal -- npx http-server /home/admin/Data/VAS_TRAINSET -p 8080
sleep 1

cd /home/admin/Python/VAS
source bin/activate

echo "Execute"
python3 /home/admin/codes/VAS_1.1/src/main_IPcamera_new.py > /home/admin/codes/VAS_1.1/src/logs/out.txt

echo "Start"
exit 0