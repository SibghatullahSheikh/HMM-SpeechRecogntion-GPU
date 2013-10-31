#!/bin/bash

awk '/oclTime/ {print $3}'  $1 

