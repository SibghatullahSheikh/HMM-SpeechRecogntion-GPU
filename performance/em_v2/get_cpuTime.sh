#!/bin/bash

awk '/cpuTime/ {print $3}'  $1 

