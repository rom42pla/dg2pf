#!/bin/bash
if [ ! -d $(pwd)/_condor_logs ]
then
    mkdir $(pwd)/_condor_logs
    echo "created $(pwd)/_condor_logs"
fi

printf "universe = docker
docker_image = rom42pla/rgb2imu:v6
executable = /bin/python3
arguments = $(dirname $PWD)/main.py $*
output = $(pwd)/_condor_logs/out_\$(ClusterId)
error = $(pwd)/_condor_logs/err_\$(ClusterId)
log = $(pwd)/_condor_logs/log_\$(ClusterId)
request_cpus = 1
request_gpus = 1
request_memory = 128G
request_disk = 100G
+MountData1=TRUE
+MountData2=FALSE
+MountHomes=FALSE
queue 1" > run.sub

cat run.sub
condor_submit run.sub
rm run.sub