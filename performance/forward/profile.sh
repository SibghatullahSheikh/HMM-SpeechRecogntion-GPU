#!/bin/bash


# read the command options
for (( i = 0 ;  i <= 5 ; i =  i + 1 ))
do
  echo "Run Job $i"
  ./ocl_fo -job $i
done

echo "Check the directory ./profile_result"
  
