#!/bin/bash

if [ ! -d "./result" ];
then
  mkdir "result"
fi

# read the command options
for (( i = 0 ;  i <= 5 ; i =  i + 1 ))
do

  echo "Run Job $i"

  if [ -f ./result/job_$i.txt ];
  then
    rm ./result/job_$i.txt
  fi

  if [ -f ./result/gpuTime_job_$i.txt ];
  then
    rm ./result/gpuTime_job_$i.txt
  fi

  if [ -f ./result/cpuTime_job_$i.txt ];
  then
    rm ./result/cpuTime_job_$i.txt
  fi
	
  for (( j = 1 ;  j <= 10 ; j =  j + 1 ))
  do
    ./ocl_fo -job $i >> ./result/job_$i.txt
  done

  ./get_gpuTime.sh ./result/job_$i.txt | cut -f1 | sort -nr | tail -1 >> ./result/gpuTime_job_$i.txt
  ./get_cpuTime.sh ./result/job_$i.txt | cut -f1 | sort -nr | tail -1 >> ./result/cpuTime_job_$i.txt

done

echo "Check the directory ./result"
  
